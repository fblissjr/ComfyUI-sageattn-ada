"""MiniMax H3 SageAttention node.

Drop it between the model loader and the sampler. Defaults are the ones
you want; the rest is there for when something goes wrong.
"""

from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .attention import MODES, build_kernel, make_minimax_attn_forward, reset_fallback_state

logger = logging.getLogger(__name__)


def _is_minimax_h3(diffusion_model):
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


class MiniMaxH3SageAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SageAttention",
            display_name="MiniMax H3 SageAttention",
            category="sageattn-ada",
            description=(
                "Runs MiniMax H3's self-attention on SageAttention's sm89 "
                "INT8/FP8 kernel instead of torch attention. Connect between "
                "the model loader and the sampler; the defaults are the "
                "intended configuration."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode", options=list(MODES), default="auto",
                    tooltip=(
                        "Which kernel to run. 'auto' lets SageAttention pick "
                        "and is right on every supported card -- on a 4090 it "
                        "resolves to fp8++. The explicit entries are for "
                        "bisecting a suspected accuracy problem: fp16 is the "
                        "most accurate and the slowest."
                    ),
                ),
                io.Boolean.Input(
                    "patch_token_refiner", default=False, optional=True,
                    tooltip=(
                        "Also patch the 2 text token-refiner blocks. They run "
                        "over the text span only (~2k tokens vs ~40k for the "
                        "DiT blocks), so this is worth well under 1% of "
                        "attention time. Off by default."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="auto", patch_token_refiner=False) -> io.NodeOutput:
        diffusion_model = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(diffusion_model):
            raise RuntimeError(
                f"This node only patches MiniMax H3; got "
                f"{type(diffusion_model).__name__}. Remove it from the graph "
                f"or feed it an H3 model."
            )

        kernel_fn, kernel_kwargs = build_kernel(mode)
        forward = make_minimax_attn_forward(kernel_fn, kernel_kwargs)
        reset_fallback_state()

        m = model.clone()
        blocks = list(diffusion_model.blocks)
        targets = [(f"diffusion_model.blocks.{i}", b.attn) for i, b in enumerate(blocks)]
        if patch_token_refiner:
            targets += [
                (f"diffusion_model.token_refiner.blocks.{i}", b.attn)
                for i, b in enumerate(diffusion_model.token_refiner.blocks)
            ]

        for path, attn in targets:
            m.add_object_patch(
                f"{path}.attn.forward", forward.__get__(attn, attn.__class__)
            )

        logger.info(
            "[sageattn-ada] MiniMax H3 self-attention on sage (mode=%s, "
            "%d attention modules patched)", mode, len(targets),
        )
        return io.NodeOutput(m)


class SageAttnAdaExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SageAttention]


async def comfy_entrypoint() -> SageAttnAdaExtension:
    return SageAttnAdaExtension()
