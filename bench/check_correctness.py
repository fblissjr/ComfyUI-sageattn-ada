#!/usr/bin/env python3
"""Check the patched MiniMax H3 forward against the stock one.

The node replaces a whole `Attention.forward`, not just the attention
call, so this compares the module's real output -- projections, norms and
all -- rather than a bare kernel. Both the RoPE path and the eager
q_norm/k_norm path are covered, because the node rewires both.

SageAttention quantizes Q/K to INT8 and V to FP8, so the output is not
bit-identical by design. What matters is that the error sits at the
kernel's known level (mean relative error ~0.1 on Ada) rather than at the
"wired something up wrong" level.

    python bench/check_correctness.py

Needs ComfyUI importable and ~2 GiB of free VRAM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
# Small enough to run anywhere; the failure modes this catches (wrong
# layout, wrong reshape, dropped RoPE) are shape-independent.
SEQ = 4096
TOLERANCE = 0.15


def relative_error(actual, expect):
    a, e = actual.float(), expect.float()
    denom = torch.maximum(a.abs(), e.abs()).clamp(min=torch.finfo(torch.float32).eps)
    return ((a - e).abs() / denom).mean().item()


def rope_table(seq, rot_pairs, device, dtype):
    """A [1, S, 1, rot/2, 2, 2] rotation table shaped like the model's."""
    angles = torch.randn(seq, rot_pairs, device=device, dtype=torch.float32)
    c, s = torch.cos(angles), torch.sin(angles)
    return torch.stack([c, -s, s, c], dim=-1).reshape(
        1, seq, 1, rot_pairs, 2, 2
    ).to(dtype)


def run_case(name, rope_freqs, device, dtype):
    import comfy.ops
    from comfy.ldm.minimax.model import Attention

    from attention import build_kernel, make_minimax_attn_forward

    torch.manual_seed(0)
    attn = Attention(
        HIDDEN, HEADS, HEAD_DIM, 1e-5,
        dtype=dtype, device=device, operations=comfy.ops.manual_cast,
    ).to(device)
    # manual_cast leaves weights uninitialized; give them a real distribution
    # so the comparison is not dominated by garbage magnitudes.
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)
        # comfy-kitchen's in-place RoPE refuses to touch anything that
        # requires grad; ComfyUI loads diffusion weights this way too.
        p.requires_grad_(False)

    x = torch.randn(SEQ, HIDDEN, device=device, dtype=dtype)

    expect = attn(x, rope_freqs=rope_freqs)

    kernel_fn, kernel_kwargs = build_kernel("auto")
    forward = make_minimax_attn_forward(kernel_fn, kernel_kwargs)
    attn.forward = forward.__get__(attn, attn.__class__)
    got = attn(x, rope_freqs=rope_freqs)

    assert got.shape == expect.shape, f"{name}: {got.shape} != {expect.shape}"
    assert torch.isfinite(got).all(), f"{name}: output contains NaN or Inf"
    err = relative_error(got, expect)
    status = "ok  " if err < TOLERANCE else "FAIL"
    print(f"  {status} {name:24s} mean relative error {err:.4f}")
    return err < TOLERANCE


@torch.inference_mode()
def main() -> int:
    # inference_mode, not no_grad: comfy-kitchen's in-place rms_rope kernel
    # refuses to run under autograd, and inference is the mode the node
    # actually runs in.
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.", file=sys.stderr)
        return 0
    device, dtype = torch.device("cuda"), torch.bfloat16

    print(f"MiniMax H3 Attention, seq={SEQ}, heads={HEADS}, head_dim={HEAD_DIM}")
    ok = [run_case("eager q_norm/k_norm", None, device, dtype)]

    # rot_dim comes back as rope_freqs.shape[-3] * 2; the model uses 96 of
    # the 128 head dims, so 48 pairs.
    ok.append(run_case(
        "fused RMSNorm + RoPE", rope_table(SEQ, 48, device, dtype), device, dtype
    ))

    if not all(ok):
        print("\nFAILED: patched forward diverges beyond the kernel's noise floor.")
        return 1
    print("\nAll cases within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
