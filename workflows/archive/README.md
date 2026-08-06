# Superseded workflows

Runnable snapshots of configs that used to ship. Drag any of these into
ComfyUI directly — they are complete UI graphs, not fragments.

These are here so a config change is reversible without git surgery. The
live workflows in the parent directory are generated from `h3_config.py`, so
git history is the authoritative record; this directory is the convenience
copy.

**Do not edit these.** They are frozen at the settings they shipped with. To
change a config, edit `h3_config.py` and regenerate — hand-editing a JSON
desyncs the API and UI forms, which `cross_check()` will fail the build on.

## What is here

### `2026-08-06_tau1.3_denseblocks/`

`tau 1.3`, `dense_blocks="33-35,39-42"`, everything else as shipped today.

Superseded the same day in favour of `tau 2.0` with no dense blocks, which
is ~5-10% faster per render. Both of the settings that differ here were
fixes for a reported artifact: sparse routing making a small persistent
object dissolve mid-clip and be replaced by something else.

**That artifact was not reproduced at 16:9 / 362 frames**, at tau 2.0, past
its reported onset, on a deliberately tracked object. What is visible at
that length is broad late-clip softening — a different shape, and the
ordinary long-clip DiT failure near the top of the trained range.
`dense_blocks` made no visible difference to it on a same-seed comparison
and cost 29-70 s, so it was dropped for our geometry.

Reach for this snapshot if a future run shows **object-specific dropout** —
a small high-frequency thing vanishing outright, rather than everything
getting softer. That is the failure shape the sparse mechanism predicts and
the one these settings were built for.

The block list itself comes from a measured per-block sensitivity profile
and is independent of whether the exemption fixes any particular failure.
`SOL_ARTIFACT_INSURANCE` in `h3_config.py` keeps it live in code.
