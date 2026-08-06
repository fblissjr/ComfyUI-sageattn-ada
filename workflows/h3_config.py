"""Single source of truth for the H3 node chain and its settings.

Both `workflows/build_workflows.py` (which emits the graphs you open in
ComfyUI) and `bench/bench_e2e_h3.py` (which produces the numbers) import
from here. Before this file existed they each carried their own copy of the
SolAttn settings, and those copies drifted the moment one was updated -- so
a bench arm named "sol" and the workflow you would actually render were
different configurations, and the measurement described something nobody
ran. That is the same failure as quoting a speedup for a config that was
never rendered; keep it structural rather than remembered.

Nothing here is allowed to have a second copy anywhere in the repo.
"""

# Checkpoint names are the ones ComfyUI actually offers. The bundled
# templates ask for `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, an NVFP4
# text encoder that is not present in this install (and is a
# Blackwell-oriented quant); the int8_convrot build is the one to use.
MODELS = dict(
    unet_fl2va="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    unet_ref2va="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    # Staying on fp16 pending a quality pass. The int8_convrot decoder is
    # usable (Comfy-Org/ComfyUI#15334 merged 2026-08-06, loader branch at
    # comfy/sd.py:945) and measured 2666.2 MiB resident against fp16's
    # 4966.5 -- a real 2300 MiB, engagement confirmed by allocation rather
    # than by a log line, since the loader prints `dtype: torch.float16` for
    # both builds and cannot distinguish int8 storage from a dequantized
    # fallback.
    #
    # What is not measured is quality. A quantized decoder fails in the
    # temporal axis -- flicker, or block boundaries that only move frame to
    # frame -- which is the same axis the tau 2.0 sign-off missed by
    # comparing stills. Swap this in after a video pass at real length, not
    # before. Decode speed is also unmeasured.
    video_vae="minimax_h3_video_vae_fp16.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)

# From the bundled templates, which is what keeps these comparable to the
# numbers in the README. `res_multistep` is one model eval per step (true
# multistep, reuses `old_denoised`), so it costs what euler costs -- sampler
# choice here is a quality decision, never a speed one.
SAMPLING = dict(sampler="res_multistep", scheduler="simple", steps=20, denoise=1.0)

# SolAttn knobs, pinned so neither a graph nor a bench arm inherits whatever
# the node currently defaults to. Pinning is load-bearing and has already
# nearly failed once: SolAttn changed `int8_qk`, `int8_pv` and `morton_curve`
# defaults underneath us, so an arm named "sol" would have meant different
# things before and after that release with no visible change on our side.
#
# Revised 2026-08-06 on a 4090 / 24 GB, where render time is the objective
# and VRAM headroom only counts insofar as it converts to render time. It
# mostly does not here: weight streaming is 0.6% of a 362-frame step and the
# trace shows it already hidden behind compute, and phase swapping is 2.0%,
# most of it unavoidable because the text encoder and DiT are 45.9 GB
# together and can never co-reside. So the ceiling on headroom-to-speed is
# ~2.6%, and knobs that trade launches for headroom are not worth it.
#
#   tau 2.0           The largest single lever. The often-quoted 1.64x
#                     (602.9 s against 991.0 s sage-only, tau 1.2 at
#                     714.9 s) comes from a 2026-08-05 ladder taken on
#                     Sol-Attn and KJNodes commits nobody recorded -- which
#                     is why internal/h3_stack_versions.md now exists. A
#                     2026-08-06 run of this same config measured 693.3 s
#                     total, but autotune-cold and against a different
#                     stopwatch (total, not sampler), so the two are not
#                     comparable and the gap cannot currently be
#                     decomposed. **Do not quote a speedup from this file
#                     until a warm re-run on pinned commits exists.**
#                     It was briefly set to
#                     1.3 on a report that sparse routing makes a small
#                     persistent object dissolve mid-clip above tau ~1.5.
#   dense_blocks ""   Was 33-35,39-42, the two highest-error regions on the
#                     author's per-block sensitivity profile.
#
# Why both were dropped, stated carefully because this went back and forth
# twice and the intermediate positions were both written down before being
# superseded. Three separate claims, each at its own tier:
#
# 1. MEASURED. The reported artifact -- a small persistent object losing
#    its identity mid-clip and being regenerated as something else -- was
#    **not reproduced** at 16:9 / 362 frames, at tau 2.0, past the reported
#    onset, on a deliberately tracked object.
# 2. OBSERVED. What is there instead is broad late-clip softening: zoomed-out
#    content getting less clear and drifting, mildly, at the end. That is the
#    ordinary long-clip DiT failure at 362 frames, the top of H3's trained
#    range, and it is a different shape from the reported artifact.
# 3. MEASURED. `dense_blocks` made no visible difference to (2) on a
#    same-seed comparison, and cost 29-70 s (~5-10%) per render.
#
# INFERRED, not measured: that sparse routing is not implicated. It rests on
# the drift's shape -- broad rather than small-object-specific -- where the
# sparse mechanism predicts the opposite, since the long-range temporal links
# carrying one object's identity are what a block-sparse router prunes first.
# No no-sparsity control was run; every arm in the comparison had sparse
# attention on, so none of them isolates sparsity. One sage-only 362-frame
# arm at the same seed would settle it, and was deliberately skipped as not
# worth a render window against something that reads as model-inherent. Do
# not let this harden into a fact by repetition -- it is an inference from
# one viewing, and it is also what decides whether these 5-10% are free.
#
# The block list is kept below rather than deleted, since it comes from a
# measured sensitivity profile and is independent of whether the exemption
# fixes this particular failure. Re-derive with `SolAttnBlockProbe` at the
# tau actually in use before trusting it -- a profile taken at a gentler
# setting is measured where the failure does not occur. Scope the negative
# result honestly too: it is "does not fix the drift at 16:9 / 362 frames",
# not "the blocks do not help", which we have not tested at the geometry
# they were derived on (768x1344 / 192 frames).
#   exact_kv_and_rows Runs the packed conditioning query rows dense, which
#                     is what keeps the generated audio intact. Those rows
#                     are ~250-400 in a ~38k sequence, thin enough to be
#                     exactly what a block-sparse router drops first -- the
#                     same shape as the object-dissolve artifact above.
#   morton off        Worth 1.16x alone but a net loss stacked on int8
#                     (1.34x against 1.39x), and its arm runs at 94% GPU
#                     utilisation where every other arm hits 99%.
#   int8_qk/pv on     Worth 1.16x on top of plain sparsity at 362 frames.
#
# Head chunking (KJNodes' MiniMax H3 Low VRAM Attention) is deliberately not
# in this chain: it costs ~4x the attention launches to buy headroom that
# converts to at most the ~2.6% above. Revisit only if a head_chunks 1-vs-4
# A/B says the launches are free.
SOL_RECOMMENDED = dict(
    tau=2.0, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=True, sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", int8_pv=True, verbose=False, use_tma=False,
    dense_blocks="",
)

# Kept for the day someone reproduces the artifact and wants the fix back.
# Not in the shipped config -- see the tau/dense_blocks notes above.
SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")

# The settings the 124-frame evaluation in docs/SOLATTN.md ran on. This exists so
# a bench arm can reproduce an old number, and it deliberately differs from
# SOL_RECOMMENDED above -- do not "fix" it to match. Every recorded ratio in
# docs/SOLATTN.md's frontier table was produced with these, so changing them
# silently makes old and new numbers incomparable while both still print.
#
# Keeping the two side by side is the point: before this file existed the
# bench and the workflow builder each had one of these and neither knew the
# other existed, so the difference read as a bug rather than as two things
# doing different jobs.
SOL_BASELINE_124F = dict(
    tau=1.2, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=False, int8_pv=False, sink_conditioning="exact_kv", morton=False,
    morton_curve="3d", verbose=False, use_tma=False, dense_blocks="",
)

# Our own node. `auto` resolves to fp8++ on sm89, so naming it explicitly
# would change nothing; the explicit modes exist for bisecting accuracy.
# token_refiner runs over the text span only (~2k rows against ~42k), so
# patching it is worth well under 1% of attention time.
SAGE_NODE = dict(mode="auto", patch_token_refiner=False)

# Node order is not cosmetic. Sol-Attn composes with the attention patches
# it finds, so it must come after ours; reversed, it overwrites the patch
# and you silently get sage only.
CHAIN = ["Load Diffusion Model", "MiniMax H3 SageAttention", "SolAttnPatch"]

CANVAS = dict(width=1344, height=768)
FPS = 24.0

# Frame counts snap to a 17k+5 grid and the node's tooltip puts the trained
# range at ~124-362. 124 is ~5.2s at 24fps.
LENGTH = 124
LONG_LENGTH = 362

SEED = 1

# `adapt_canvas` imposes short edge 768 and a hard area cap of 768*1344 =
# 1,032,192 px, each axis rounded to 32. That cap is why this is a list of
# aspect ratios rather than resolutions -- there is no higher one to pick.
# 21:9, 16:9 and 9:16 all land on the cap and cost the same; a square never
# reaches it, so 1:1 is a third of the attention cost of 16:9 at equal frame
# count. Landscape and portrait of a ratio are exactly equal in cost, since
# packed rows are (h//32)*(w//32).
ASPECTS = {
    "16x9":  (1344, 768),
    "9x16":  (768, 1344),
    "4x3":   (1024, 768),
    "3x4":   (768, 1024),
    "1x1":   (768, 768),
}
