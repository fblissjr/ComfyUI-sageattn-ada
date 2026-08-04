"""SageAttention forward for MiniMax H3's packed self-attention.

H3 runs one unmasked self-attention per DiT block over the whole packed
`[text | cond | audio | video]` sequence -- 56 heads, head_dim 128, no
mask anywhere. That is exactly the shape SageAttention's sm89 INT8-QK /
FP8-PV kernel is built for, and at the node's default canvas it lands
about 2.7x ahead of torch's flash backend.

Replacing the block's `forward` rather than going through ComfyUI's
`optimized_attention` buys two things:

  - q/k/v stay in NHD, the layout `qkv_proj` already produces, instead of
    being transposed into HND and back.
  - the float q/k/v can be handed to sage as a list it takes ownership
    of, so they are released as soon as their quantized forms exist
    rather than at the end of the call.

Everything else -- the fused RMSNorm + split-half RoPE, the output
projection -- is left exactly as the stock forward does it, including
running in place on the qkv buffer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_FALLBACK_LOGGED = False


def _log_fallback_once(exc):
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED:
        _FALLBACK_LOGGED = True
        logger.warning(
            "[sageattn-ada] sage kernel raised (%s: %s); this block and any "
            "later failure fall back to ComfyUI's attention for the rest of "
            "the run. The render continues, just without sage.",
            type(exc).__name__, exc,
        )


def reset_fallback_state():
    """Let the next run report a fallback again instead of staying quiet."""
    global _FALLBACK_LOGGED
    _FALLBACK_LOGGED = False


def _sage():
    """Import sageattention lazily so a missing install is a node-time
    error the user can read, not an import-time traceback at startup."""
    try:
        import sageattention
    except ImportError as exc:
        raise RuntimeError(
            "sageattention is not installed in this ComfyUI's environment. "
            "This node needs the Ada fork (SageAttention-ada) built from "
            "source; see the README."
        ) from exc
    return sageattention


# mode -> (attribute on sageattention, extra kwargs). "auto" lets sage's
# own dispatcher pick, which is correct on every card we support; the
# explicit entries exist so a suspected accuracy problem can be bisected
# without editing code.
MODES = {
    "auto": (None, {}),
    "fp8++ (fastest)": ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16"}),
    "fp8": ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp32"}),
    "fp16 (most accurate)": ("sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
}


def build_kernel(mode):
    """Return `(fn(qkv_list, **kw) -> NHD output, kwargs)` for `mode`."""
    sa = _sage()
    if not hasattr(sa, "sageattn_consume"):
        raise RuntimeError(
            "The installed sageattention has no sageattn_consume(). This "
            "node needs the Ada fork at a version that provides it; a stock "
            "SageAttention install will not work."
        )

    attr, extra = MODES[mode]
    base_kwargs = {
        "tensor_layout": "NHD",
        "is_causal": False,
        # H3 self-attention is unmasked, and ComfyUI passes smooth_k=False
        # on its own sage path. Keeping it off avoids a K-mean pass that
        # buys nothing measurable at these shapes.
        "smooth_k": False,
        **extra,
    }

    if attr is None:
        return sa.sageattn_consume, base_kwargs

    kernel = getattr(sa, attr)

    def consume_specific(qkv, **kw):
        q, k, v = qkv
        qkv.clear()
        return kernel(q, k, v, **kw)

    return consume_specific, base_kwargs


def make_minimax_attn_forward(kernel_fn, kernel_kwargs):
    """Build a replacement `Attention.forward` bound to one sage kernel.

    `kernel_fn(qkv_list, **kernel_kwargs)` must consume the `[q, k, v]`
    list and return an NHD-shaped output.
    """

    def forward(self, x, rope_freqs=None, transformer_options={}):
        import comfy.model_management
        import comfy.quant_ops

        s = x.shape[0]
        # One fused projection, split into three views of the same buffer.
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        v = v.view(1, s, self.heads, self.head_dim)

        if rope_freqs is not None:
            # Same fused per-head RMSNorm + partial split-half rope the stock
            # forward uses, in place on the qkv buffer.
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2,
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        qkv = [q, k, v]
        del q, k, v  # the list is now the only owner

        try:
            out = kernel_fn(qkv, **kernel_kwargs)
        except Exception as exc:
            # The kernel consumes the list, so there is nothing left to
            # retry with -- recompute from x through ComfyUI's own forward.
            # Wasteful, but this path only runs when sage has already
            # failed and the alternative is failing the render.
            _log_fallback_once(exc)
            del qkv
            return _stock_forward(self, x, rope_freqs, transformer_options)

        return self.out_proj(out.view(s, self.heads * self.head_dim))

    return forward


def _stock_forward(self, x, rope_freqs, transformer_options):
    """Re-run ComfyUI's own unpatched Attention.forward from scratch."""
    from comfy.ldm.minimax.model import Attention

    return Attention.forward(
        self, x, rope_freqs=rope_freqs, transformer_options=transformer_options
    )
