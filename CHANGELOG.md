# Changelog

Semantic versioning. Nothing here has been tagged or published, so every
version below describes the state of the working repo rather than a release
artifact.

## 0.2.0

### Added

- `workflows/h3_image_ref_plus_text_to_video_ref_lora.json` (and its `_api`
  copy): the shipped image-ref graph with one thing changed, loading fl2va
  plus Kijai's extracted ref LoRA instead of loading ref2va. Seed, prompt,
  canvas, length, sampler, sage and every Sol-Attn setting are shared with
  its sibling by construction, so anything that differs between the two
  renders is the LoRA.
- `REF_LORA` and `REF_LORA_STRENGTH` in `workflows/h3_config.py`.
- `unet`, `lora`, `out_prefix` and `title` parameters on `build_api` and
  `build_ui`. Checkpoint and task had to come apart: the new graph wants r2v
  conditioning driven by the fl2va checkpoint, which no task name describes.

### Changed

- Both workflow-emitting loops now read one `GRAPHS` table, so the
  difference between the shipped ref graph and its ref-LoRA sibling is a
  single `extra` dict and the two cannot drift apart in a dimension nobody
  intended.
- `cross_check` compares `UNETLoader.unet_name` and
  `LoraLoaderModelOnly.strength_model`, not only node presence. Until `unet`
  and `lora` became free builder parameters the checkpoint was derived from
  `task` inside both builders and the two formats could not disagree about
  it. Now they can, and the node-set check cannot see it, because both
  formats carry a `UNETLoader` either way. Both assertions were confirmed to
  fail when the formats are deliberately desynced.
- `coderef/` is ignored. It holds machine-local symlinks to sister repos
  rather than repo content.

The three existing graphs regenerate byte-identical.

## 0.1.0

Initial state, covering everything before the repo took an interest in the
ref LoRA.

### Added

- MiniMax H3 SageAttention node. Replaces each DiT block's attention
  `forward` with SageAttention's INT8-QK / FP8-PV kernel on Ada (sm89), and
  registers itself as the `optimized_attention_override` fallback so a
  sparse-attention patch layered on top composes with it instead of
  replacing it.
- MiniMax H3 Keyframe Canvas. Fits a keyframe to the target canvas before
  it reaches `MiniMaxH3ImageToVideo`, whose own resize stretches
  non-uniformly.
- MiniMax H3 Provenance Stamp, bench only. Records what a render's settings
  resolved to rather than what was typed, which `/history` already has. The
  field it exists for is `n_sparse`, the sigma window intersected with the
  sampler's schedule, which is readable from nothing else.
- Assert Sage Attention Chain. Turns a silently bypassed attention patch
  into a hard failure instead of a render that succeeds and is quietly
  slower or numerically different than intended.
- Benchmarks under `bench/`: module-level and end-to-end sage A/B,
  correctness check, keyframe-canvas check, override-routing check, and a
  smoke pass.
- Three generated workflows in `workflows/`, in UI and API format, built
  from one description so the two formats cannot describe different graphs.
- `workflows/h3_config.py` as the single source for checkpoint names,
  sampler settings, canvas geometry and Sol-Attn knobs, shared by the
  workflow generator and the bench. Both previously carried their own copy
  and those copies drifted.
- Sol-Attn interop evaluation in `docs/SOLATTN.md`, and H3 geometry notes in
  `docs/h3_geometry_and_nodes.md`.

### Changed

- Renamed from `ComfyUI-sageattn-ada` to `ComfyUI-h3-explorations`. Node
  ids were deliberately left alone: they are baked into every saved
  workflow's `type` field, and renaming one breaks every saved graph that
  uses it with no clear error in the UI.
