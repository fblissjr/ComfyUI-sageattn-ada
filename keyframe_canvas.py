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
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="first_frame"),
                io.Image.Output(display_name="last_frame"),
            ],
        )

    @classmethod
    def execute(cls, first_frame, last_frame=None) -> io.NodeOutput:
        if first_frame.shape[0] > 1:
            # the H3 node takes [:1] silently; say so rather than let a batch
            # look like it was used
            logger.warning(
                "[sageattn-ada] first_frame carries %d images; MiniMax H3 uses only "
                "the first. Batch the prompt instead if you meant several renders.",
                first_frame.shape[0],
            )
        src_h, src_w = int(first_frame.shape[1]), int(first_frame.shape[2])
        width, height = adapt_canvas(src_w, src_h)

        # Aspect already matches, so "disabled" is a uniform scale here, not a
        # stretch. This mirrors the reference's anchor path.
        first_out = _resize(first_frame[:1], width, height, "disabled")

        # The follower is cover-cropped, as in the reference. Passing the anchor
        # through again when there is no follower keeps the output wired.
        if last_frame is not None:
            last_out = _resize(last_frame[:1], width, height, "center")
        else:
            last_out = first_out

        logger.info(
            "[sageattn-ada] H3 canvas %dx%d derived from a %dx%d keyframe "
            "(aspect %.4f -> %.4f)",
            width, height, src_w, src_h, src_w / src_h, width / height,
        )
        return io.NodeOutput(width, height, first_out, last_out)
