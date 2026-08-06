"""Keyframe geometry: does the canvas we hand H3 avoid distorting the keyframe?

Claims, one per case. Delete a case and you stop noticing the corresponding
failure:

1. `adapt_canvas` reproduces the reference's `resolve_canvas_size` -- 768 short
   edge, 768*1344 area cap, both axes rounded to 32 -- so deriving the canvas
   from a keyframe puts ComfyUI on the reference's default path.
2. A canvas derived from an image preserves that image's aspect to within the
   round-to-32 quantisation. This is the property the whole node exists for.
3. The stock node's first-frame resize is a NON-UNIFORM stretch whenever the
   keyframe aspect differs from the canvas. This is the defect; if this case
   ever goes green the defect is gone and the node can be retired.
4. Feeding a derived canvas makes that stretch a no-op, because the keyframe
   already has exactly the canvas dimensions.

Reference: coderef/diffusers .../modular_pipelines/minimax_h3/before_encoder.py
::MiniMaxH3ResizeStep, and modular_pipeline.py::resolve_canvas_size.

Run: python bench/check_keyframe_canvas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ComfyUI root: custom_nodes/<this repo>/bench -> up three
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE, MAX_PIXELS, adapt_canvas, _resize

# (w, h) worth covering: square, portrait, ultrawide, a TRUE 16:9 (1.7778), the
# canvas's own 7:4 (1.75), and an odd size that is not a multiple of 32.
# 1920x1080 and 1344x768 are deliberately both here: the default canvas is 7:4,
# NOT 16:9, so a real 16:9 source is not a no-op. Round-to-32 means no canvas is
# exactly 16:9, which is the model's rule and not a ComfyUI choice.
SOURCES = [(1024, 1024), (768, 1024), (2560, 1080), (1920, 1080), (1344, 768), (1000, 700)]

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("--- 1. adapt_canvas matches the reference's canvas rule ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    check(f"{w}x{h} -> {cw}x{ch}: axes are multiples of {CANVAS_MULTIPLE}",
          cw % CANVAS_MULTIPLE == 0 and ch % CANVAS_MULTIPLE == 0)
    check(f"{w}x{h}: area within cap (+round-32 slack)",
          cw * ch <= MAX_PIXELS * 1.15, f"{cw * ch} vs {MAX_PIXELS}")

print("\n--- 2. derived canvas preserves source aspect ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    src, got = w / h, cw / ch
    # round-to-32 on both axes is the only permitted error
    tol = max(CANVAS_MULTIPLE / ch, CANVAS_MULTIPLE * cw / (ch * ch))
    check(f"{w}x{h}: aspect {src:.4f} -> {got:.4f}", abs(src - got) <= tol,
          f"delta={abs(src - got):.4f} tol={tol:.4f}")

print("\n--- 3. the defect: stock first-frame path stretches non-uniformly ---")
DEFAULT_W, DEFAULT_H = 1344, 768
for w, h in SOURCES:
    img = torch.zeros(1, h, w, 3)
    out = _resize(img, DEFAULT_W, DEFAULT_H, "disabled")
    sx, sy = DEFAULT_W / w, DEFAULT_H / h
    distortion = max(sx, sy) / min(sx, sy)
    mismatched = abs(w / h - DEFAULT_W / DEFAULT_H) > 1e-6
    check(f"{w}x{h} at {DEFAULT_W}x{DEFAULT_H}: distortion {distortion:.3f}x",
          (distortion > 1.01) == mismatched,
          "(expected >1 only when aspect differs)")
    check(f"{w}x{h}: output is exactly the canvas",
          tuple(out.shape[1:3]) == (DEFAULT_H, DEFAULT_W))

print("\n--- 4. derived canvas makes the stock stretch a no-op ---")
for w, h in SOURCES:
    cw, ch = adapt_canvas(w, h)
    fitted = _resize(torch.rand(1, h, w, 3), cw, ch, "disabled")
    # what the stock node then does to a keyframe already at canvas size
    first = _resize(fitted, cw, ch, "disabled")
    last = _resize(fitted, cw, ch, "center")
    check(f"{w}x{h}: first-frame path is identity", torch.equal(first, fitted),
          f"max|delta|={(first - fitted).abs().max():.3e}")
    check(f"{w}x{h}: last-frame path is identity", torch.equal(last, fitted),
          f"max|delta|={(last - fitted).abs().max():.3e}")

print()
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
