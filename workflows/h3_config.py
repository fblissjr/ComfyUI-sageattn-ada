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

# Kijai's extracted fl2va -> ref2va weight difference (Kijai/MiniMax-H3-experimental,
# 2026-08-08). Rank 256 on the attention and MLP projections, rank 8 on the
# adaln projections, plus full-rank `diff`/`diff_b` deltas on every norm and
# bias -- a whole-model delta extraction, not a trained adapter. Coverage
# matches the fl2va checkpoint exactly: verified against both safetensors
# headers, and against comfy.lora.load_lora, which turns its 794 tensors into
# 530 patches with zero unmatched keys. So at strength 1.0 it should
# reconstruct ref2va, up to rank truncation and requantization error.
#
# That "should" is why `h3_image_ref_plus_text_to_video_ref_lora.json` exists
# as a sibling of the shipped ref graph rather than as a claim: upstream's own
# description is "completely experimental, I don't even know if it has a use
# case at this point". Run the two and judge.
#
# The `h3/` prefix is load-bearing: LoRAs are foldered in this install and
# ComfyUI's combo carries the subfolder in the value, so the bare filename is
# rejected by /object_info validation.
REF_LORA = "h3/minimax_h3_ref_lora_rank_256_bf16.safetensors"

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

# `res_multistep` is one model eval per step (true multistep, reuses
# `old_denoised`), so it costs what euler costs. Note that is not true of the
# whole sampler list: `heun`, `dpm_2`, and the `2s`/`3s`/`res_Ns` families are
# 2-6 evals per step, and picking one silently multiplies the ~91% of render
# time the sampler occupies.
#
# **steps 16, measured and judged 2026-08-06.** 20 was the bundled template's
# default and was never questioned until it turned out to be the largest
# untested lever in the config -- steps multiply everything, including the
# ~24% of the step that attention work cannot reach. At 362 frames:
#
#   20 steps  765.4 s      16 steps  669.2 s  (-12.6%)      12 steps  508.5 s
#
# 12 was rejected, and for a reason worth recording because two of the three
# gates we had prepared would have passed it. It is not smeared and it has no
# late-clip artifact -- it simply **stops following the prompt**. The test
# prompt specifies three shots with explicit cut times, and at 12 steps the
# third one (a pull-back into the street at 00:10) never happens. Shot
# structure is a long-range instruction and the trajectory settles into
# something simpler before reaching it. Invisible in stills, invisible to a
# convergence check, only catchable by watching to the end knowing what was
# asked for. Any future step reduction needs that as a third gate.
#
# Scheduler stays `simple`. `beta57` was tested and dropped -- not because it
# sampled worse but because the comparison was never isolating the scheduler:
# Sol-Attn's window is a *percent* band that `percent_to_sigma` resolves off
# the model's sigma curve with no knowledge of the scheduler, so a different
# scheduler puts a different number of steps inside it. beta57 ran 10 dense
# steps at 20 against simple's 6, which is why it measured slower (825.1 s
# against 765.4). It also came from a custom node pack rather than core.
SAMPLING = dict(sampler="res_multistep", scheduler="simple", steps=16, denoise=1.0)

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
#   tau 1.3           Below the onset of the moving-content artifact -- see
#                     the two-phenomena note below. Costs 82.3 s of sampler
#                     against tau 2.0, measured same-seed at 362 frames
#                     (712.1 s against 629.8 s), and worth it.
#   dense_blocks ""   Was 33-35,39-42, the two highest-error regions on the
#                     author's per-block sensitivity profile. Dropped: it
#                     does not fix the artifact tau does, and costs 39.2 s.
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
#
# **The sigma window stays .2-.9, and widening it is closed.** `.1-.95` is
# tempting -- 687.4 s against 768.2 at 20 steps, ~10%, and it passed every
# gate there including prompt adherence. It does not survive at 16 steps:
# 568.8 s, but the shot timeline drifts (the scripted 00:10 cut lands nearer
# 12-13 s) and the subject's motion stalls. Not smearing, not the late-clip
# artifact -- a fourth failure mode, structural timing.
#
# Worth keeping as a caution rather than a footnote: **both factors passed
# adherence individually and the combination failed.** 20 steps + wide hit
# the cut on time; 16 steps + narrow hit it on time; 16 + wide did not. A
# knob validated at one setting of another knob is not validated, and the
# ten minutes spent confirming that was the cheapest measurement of the day.
SOL_RECOMMENDED = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=4096,
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

# The strength the ref-LoRA graph ships at. 1.0 is the only value with a
# defined meaning -- it is where the extracted delta is supposed to reconstruct
# ref2va. Everything below it is an interpolation the LoRA was never fitted
# for, which is the interesting part but not the default.
#
# Strength 0.0 and bypassing the node are the same thing, not two baselines.
# `LoraLoader.load_lora` short-circuits when both strengths are zero (ComfyUI
# nodes.py:729) and `LoraLoaderModelOnly` always passes strength_clip=0, so
# either route hands back the untouched model and renders true plain fl2va.
#
# What neither gives you is a baseline that took the same path as the 1.0
# arm. Applying the LoRA to a quantized checkpoint is a dequantize / add /
# requantize round trip, and the zero-strength route skips it entirely -- so
# part of any 1.0-against-0.0 difference is that round trip rather than the
# delta. To see the round trip by itself, render 0.01: visually nil, but it
# does not short-circuit, so it pays the full cost.
REF_LORA_STRENGTH = 1.0
