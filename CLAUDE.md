# ComfyUI-h3-explorations

MiniMax H3 research hub for ComfyUI: attention kernels, keyframe and
provenance nodes, benchmarks, and workflows. Start with README.md for what
ships and why. This file holds only what README.md would not.

## The one rule that matters

Never rename a node's `node_id=` (in any `io.Schema`). It is baked into
every saved workflow's `type` field: this repo's `workflows/*.json` and the
owner's live workflows outside this repo. Renaming one breaks every saved
graph that uses it, silently, with no clear error in the UI. Class names,
menu `category=`, log prefixes, and package metadata are all safe to rename.

## Running things

No test suite. Verify changes against a live ComfyUI and GPU:
`bench/smoke_h3.py` for a fast sanity pass, the relevant `bench/*.py` script
for the specific claim you changed. This repo runs inside ComfyUI's own
venv, not a standalone uv project. There is no uv.lock here on purpose.

## Research notes

`internal/` (gitignored) holds prompt-writing research, session logs, and
upstream-change surveys. Not shipped, not for redistribution.
