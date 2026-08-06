#!/usr/bin/env python3
"""Confirm the H3 chain composes and runs, after any node-pack update.

Verification comes from the log lines, not the video. Three must appear:

    sage routing: arch=sm89 ... pv_accum=fp32+fp16 -> fp8_cuda++
    [sol_attn] chaining onto an existing attention override
    [sol_attn] sparse (1, ..., 56, 128) tau=... int8 pointer

Line 1 says sage engaged on the fast kernel. Line 3 says sparse engaged at
the configured tau. **Line 2 is the order check** -- it prints only when
Sol-Attn finds sage's override already installed. Missing means the chain is
reversed and you are silently paying full price, with no error anywhere.
That seam is a protocol two third-party repos agree on and neither owns, so
it is worth re-checking on every update rather than assuming.

Two deliberate choices, both from getting them wrong first:

- **Enough steps to look like a render.** An earlier version used 4, which
  produces a smeared, incoherent clip indistinguishable from a failure --
  so the artifact it leaves behind causes exactly the alarm it was meant to
  rule out. 10 is still fast and still recognisably converging.
- **Its own filename prefix and a short clip.** The output is throwaway; it
  should not land in the middle of real renders wearing their naming.

Read the log in a terminal, or with `stdbuf -oL -eL` on the launcher.
Redirecting ComfyUI's output block-buffers it, and these lines then fail to
appear whether or not anything is wrong -- which makes an absent line
indistinguishable from a broken chain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

WF = Path(__file__).resolve().parent.parent / "workflows"

WANT = [
    ("sage engaged", "sage routing:"),
    ("node order  ", "chaining onto an existing attention override"),
    ("sparse ran  ", "] sparse ("),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--length", type=int, default=39)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--log", help="ComfyUI log file; if given, the three lines are checked here")
    args = ap.parse_args()
    base = f"http://{args.host}"

    wf = json.loads((WF / "h3_text_to_video_api.json").read_text())
    for n in wf.values():
        ct = n["class_type"]
        if ct == "MiniMaxH3ImageToVideo":
            n["inputs"]["length"] = args.length
            n["inputs"]["prompt"] = (
                "Live-action, cinematic. A woman in a dark coat walks along a "
                "rain-wet stone street past iron railings, the camera tracking "
                "with her.\n\nAudio: rain on stone, footsteps.")
        if ct == "BasicScheduler":
            n["inputs"]["steps"] = args.steps
        if ct == "SolAttnPatch":
            n["inputs"]["verbose"] = True
        if ct == "SaveVideo":
            n["inputs"]["filename_prefix"] = "video/_smoketest"

    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{base}/prompt", json.dumps({"prompt": wf, "client_id": str(uuid.uuid4())}).encode(),
        {"Content-Type": "application/json"}), timeout=60))["prompt_id"]
    print(f"submitted {pid[:8]}, {args.length} frames / {args.steps} steps", flush=True)

    # /queue rather than the log: HTTP is never buffered.
    while True:
        q = json.load(urllib.request.urlopen(f"{base}/queue", timeout=10))
        if not q["queue_running"] and not q["queue_pending"]:
            break
        time.sleep(5)

    h = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=10))
    status = h.get(pid, {}).get("status", {}).get("status_str", "missing")
    print(f"render: {status}")
    if status != "success":
        return 1

    if not args.log:
        print("\npass --log <comfyui.log> to check the three composition lines,")
        print("or read them in the terminal. The render succeeding does not")
        print("prove sage or Sol-Attn engaged -- a silent bypass also succeeds.")
        return 0

    text = Path(args.log).read_text(errors="replace")
    missing = False
    for label, needle in WANT:
        ok = needle in text
        print(f"  {label}  {'ok' if ok else 'MISSING'}")
        missing |= not ok
    if missing:
        print("\nA missing line is not proof of breakage if the log is buffered.")
        print("Confirm the log is live (byte count growing) before concluding.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
