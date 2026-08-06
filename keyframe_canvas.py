"""Resolve an H3 canvas from a keyframe, the way the reference pipeline does.

`MiniMaxH3ImageToVideo` takes `width`/`height` as required inputs (default
1344x768) and stretches the first keyframe onto them. The stretch itself is
faithful -- the reference stretches the geometry anchor and cover-crops any
follower, deliberately, to match the released model's arithmetic. What ComfyUI
does not have is the default that normally makes the stretch a no-op: the
reference derives the canvas from the first keyframe when no size is given
(`MiniMaxH3ResizeStep` -> `resolve_canvas_size`) and then skips the resize
entirely once the keyframe already matches.

So the reference's deliberate-override branch is ComfyUI's default branch, and
an off-16:9 keyframe is silently distorted: measured 1.75x for a square source
at the default canvas and 2.33x for 3:4 portrait, carried by every frame of the
clip. This node closes that gap. `adapt_canvas` is ComfyUI's own port of
`resolve_canvas_size` -- same 768 short edge, same 768*1344 area cap, same
round-to-32 -- and it already sits in `nodes_minimax_h3.py`, just unused on the
keyframe path.

Verified in `bench/check_keyframe_canvas.py`: feeding the fitted image plus the
derived size makes both of the stock node's resize calls bit-identical
no-ops, so this composes with it rather than replacing it.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import adapt_canvas, _resize

logger = logging.getLogger(__name__)


class MiniMaxH3KeyframeCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3KeyframeCanvas",
            display_name="MiniMax H3 Keyframe Canvas",
            category="model/conditioning/minimax",
            description=(
                "Derives the generation canvas from the first keyframe, matching the "
                "reference pipeline's default, and fits the keyframes onto it. Wire "
                "width/height and the fitted images into MiniMax H3 Image to Video: "
                "the keyframe then arrives already at canvas size, so that node's "
                "resize is a no-op and nothing is distorted."
            ),
            inputs=[
                io.Image.Input("first_frame"),
                io.Combo.Input("mode", options=["fit_to_canvas", "match_keyframe"],
                               default="fit_to_canvas", tooltip=(
                                   "fit_to_canvas: you own the geometry -- the keyframe is "
                                   "cover-cropped into the width/height you pass, so resolution "
                                   "and render cost stay where you put them. "
                                   "match_keyframe: the reference pipeline's default -- the "
                                   "canvas is derived from the keyframe's aspect and your "
                                   "width/height are ignored.")),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                             tooltip="Used by fit_to_canvas; ignored by match_keyframe."),
                io.Int.Input("height", default=768, min=32, max=16384, step=32,
                             tooltip="Used by fit_to_canvas; ignored by match_keyframe."),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="first_frame"),
                io.Image.Output(display_name="last_frame"),
                io.Float.Output(display_name="attn_cost_vs_1to1"),
            ],
        )

    @classmethod
    def execute(cls, first_frame, mode="fit_to_canvas", width=1344, height=768,
                last_frame=None) -> io.NodeOutput:
        if first_frame.shape[0] > 1:
            # the H3 node takes [:1] silently; say so rather than let a batch
            # look like it was used
            logger.warning(
                "[sageattn-ada] first_frame carries %d images; MiniMax H3 uses only "
                "the first. Batch the prompt instead if you meant several renders.",
                first_frame.shape[0],
            )
        src_h, src_w = int(first_frame.shape[1]), int(first_frame.shape[2])

        if mode == "match_keyframe":
            width, height = adapt_canvas(src_w, src_h)
            # Aspect now matches by construction, so "disabled" is a uniform
            # scale, not a stretch. Mirrors the reference's anchor path.
            anchor_crop = "disabled"
        else:
            # Round to the DiT's multiple of 32 and otherwise leave the size
            # alone. NOT adapt_canvas: that forces a 768 short edge and the area
            # cap, which would silently promote a 832x480 preview canvas to
            # 1344x768 -- a 6.7x attention increase from a node whose whole job
            # here is keeping render cost where the user put it.
            snapped_w = max(32, round(width / 32) * 32)
            snapped_h = max(32, round(height / 32) * 32)
            if (snapped_w, snapped_h) != (width, height):
                # step=32 constrains the UI, not an API submission, and the
                # API-format workflows in this repo are how benches are driven.
                # A size the DiT cannot patch would fail downstream with a
                # shape error that says nothing about where it came from.
                logger.warning(
                    "[sageattn-ada] canvas %dx%d is not a multiple of 32; "
                    "snapped to %dx%d. The H3 latent cannot grid the original.",
                    width, height, snapped_w, snapped_h,
                )
            width, height = snapped_w, snapped_h
            # The user owns the geometry, so the anchor is cover-cropped rather
            # than stretched. NOTE: this is a deliberate divergence -- the
            # reference stretches the anchor even when width/height are given,
            # and only cover-crops followers. Cropping both keeps proportions
            # honest when the aspect was chosen on purpose; it costs edge
            # framing. Reference fidelity lives in match_keyframe.
            anchor_crop = "center"

        first_out = _resize(first_frame[:1], width, height, anchor_crop)
        # The follower is cover-cropped in either mode, as in the reference.
        last_out = (_resize(last_frame[:1], width, height, "center")
                    if last_frame is not None else first_out)

        # DiT tokens go as (h//32)*(w//32) and attention as their square, so
        # report cost against the cheapest canvas H3 will resolve (768x768).
        tokens = (height // 32) * (width // 32)
        cheapest = (768 // 32) ** 2
        attn_cost = round((tokens / cheapest) ** 2, 3)

        logger.info(
            "[sageattn-ada] H3 canvas %dx%d (%s) from a %dx%d keyframe: "
            "aspect %.4f -> %.4f, attention ~%.2fx a 768x768 canvas",
            width, height, mode, src_w, src_h,
            src_w / src_h, width / height, attn_cost,
        )
        return io.NodeOutput(width, height, first_out, last_out, attn_cost)
