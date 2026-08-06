# MiniMax H3: valid geometry, and which nodes to use

Last updated: 2026-08-06.

Everything here is read out of ComfyUI's own `comfy_extras/nodes_minimax_h3.py`
or measured on a 4090, not inferred from community lore.

## Resolution is an aspect-ratio choice, not a quality dial

`adapt_canvas()` gives you no say in how many pixels you get. It sets the
short edge to **768**, caps the area at **768 x 1344 = 1,032,192 px**, and
rounds each axis to a multiple of **32**. There is no higher resolution to
select — asking for 4K returns the same canvas as asking for 720p at the
same aspect.

The cap only binds on wide or tall ratios, so a square never reaches it.
That is the whole reason aspect ratio is a cost decision:

| aspect | canvas | rows/frame | packed rows at 362f | attention cost |
|---|---|---|---|---|
| 21:9 | 1536x672 | 1008 | 91,728 | 1.00x |
| **16:9** | **1344x768** | 1008 | 91,728 | 1.00x |
| 3:2 | 1152x768 | 864 | 78,624 | 0.73x |
| 4:3 | 1024x768 | 768 | 69,888 | 0.58x |
| 5:4 | 960x768 | 720 | 65,520 | 0.51x |
| **1:1** | **768x768** | 576 | 52,416 | **0.33x** |
| 4:5 | 768x960 | 720 | 65,520 | 0.51x |
| 3:4 | 768x1024 | 768 | 69,888 | 0.58x |
| 2:3 | 768x1152 | 864 | 78,624 | 0.73x |
| 9:16 | 768x1344 | 1008 | 91,728 | 1.00x |

**1:1 costs a third of 16:9 at the same frame count.** Attention is O(S²)
and dominates the step at long clip lengths, so that is the largest single
lever available anywhere — larger than any kernel or sparsity setting.

**Portrait and landscape of the same ratio cost exactly the same.** Packed
rows are `(h//32) * (w//32)`, which is symmetric, so 1344x768 and 768x1344
both pack 1008 rows per frame. Any 16:9-vs-9:16 difference is the model's
training distribution, not geometry — test those for quality, never for
speed.

Rows per frame come from the VAE's 16x spatial downsample followed by the
model's `(1, 2, 2)` patchify, i.e. `(h//16//2) * (w//16//2)`.

## Frame counts snap to a 17k+5 grid

`align_frame_count()` rounds **up** to the next `n % 17 == 5`. Ask for 200
and you get 209; ask for 300 and you get 311. The node's own tooltip puts
the trained range at **~124 to 362**, and says longer is untested.

Valid counts near the top: **… 311, 328, 345, 362**.

Duration is `frames / 24`, so 362 frames is 15.08 s and 124 is ~5.2 s.

**Two things worth knowing at the top of the range.** Attention grows as S²
while everything else grows linearly, so at 362 frames attention is ~76% of
the step against ~50% at 124 — long clips are where kernel and sparsity work
pays off most. And late-clip identity softening at 362 is the ordinary
long-clip DiT failure at the edge of the trained range; stepping down to 328
or 345 costs proportionally less attention *and* reduces it.

## Which nodes to use

### Required, all ComfyUI core

| node | notes |
|---|---|
| Load Diffusion Model (`UNETLoader`) | `fl2va` checkpoint for t2v/i2v, `ref2va` for reference-to-video |
| `CLIPLoader` | Qwen3-VL-32B text encoder, type `minimax` |
| `VAELoader` x2 | video VAE and audio VAE are separate loaders |
| `MiniMaxH3ImageToVideo` | t2v **and** i2v — `first_frame`/`last_frame` are optional, so no image wired is text-to-video |
| `MiniMaxH3ReferenceToVideo` | reference images / video / audio → conditioning |
| `RandomNoise` → `KSamplerSelect` → `BasicScheduler` → `BasicGuider` → `SamplerCustomAdvanced` | standard custom-sampler stack |
| `VAEDecode` + `VAEDecodeAudio` → `CreateVideo` → `SaveVideo` | video and audio decode separately |

`res_multistep` is one model eval per step (true multistep, reuses
`old_denoised`), so it costs exactly what euler costs — sampler choice here
is a quality decision, never a speed one.

### Use from this repo

**`MiniMax H3 SageAttention`** — the attention node. Replaces all 50 DiT
attention forwards with SageAttention's INT8-QK / FP8-PV kernel, and *also*
registers an `optimized_attention_override`. That second registration is
what lets Sol-Attn compose rather than silently bypassing sage. Defaults are
the intended config.

**`MiniMax H3 Keyframe Canvas`** — use it on every i2v and fl2v graph.
`MiniMaxH3ImageToVideo` takes `width`/`height` as required inputs defaulting
to 1344x768, and non-uniformly stretches the first keyframe onto them. That
stretch is faithful to the reference pipeline, which also stretches the
geometry anchor and cover-crops any follower. What ComfyUI lacks is the
*default* that normally makes it a no-op: the reference derives the canvas
from the first keyframe when no size is given
(`modular_pipelines/minimax_h3/before_encoder.py::MiniMaxH3ResizeStep` ->
`resolve_canvas_size`) and then skips the resize once the keyframe matches.
The reference's deliberate-override branch is ComfyUI's default branch.

Measured distortion at the default canvas, from
`bench/check_keyframe_canvas.py`:

| source | ratio | stretch |
|---|---|---|
| 768x1024 | 0.750 | **2.33x** |
| 1024x1024 | 1.000 | **1.75x** |
| 2560x1080 | 2.370 | 1.35x |
| 1000x700 | 1.429 | 1.23x |
| 1920x1080 | 1.778 (true 16:9) | 1.016x |
| 1344x768 | 1.750 | 1.00x |

**The default canvas is 7:4, not 16:9.** 1344/768 = 1.7500; 16/9 = 1.7778. A
genuine 16:9 source takes a 1.6% squeeze, not a no-op — small, but do not read
the table as "16:9 is safe". Round-to-32 on both axes means no H3 canvas is
exactly 16:9: `adapt_canvas(16, 9)` returns 1344x768. That is the model's canvas
rule, not a ComfyUI choice, and the node inherits it.

It is silent, and every frame of the clip inherits it. The node runs
`adapt_canvas` — ComfyUI's own port of `resolve_canvas_size`, sitting unused
on the keyframe path — and fits the keyframes onto the result. Wire its
`width`/`height` and both image outputs into the H3 node; the keyframe then
arrives already at canvas size and the stock resize is a bit-identical no-op
(verified, `max|delta| = 0`). With two keyframes the canvas comes from the
first and the follower is cover-cropped, as in the reference.

Cost: output resolution now follows the input's aspect. A 9:16 still renders
768x1344, the slowest canvas on the area cap. That is the reference's own
behaviour, not an extra.

**`MiniMax H3 Provenance Stamp (bench)`** — **bench graphs only, keep it out
of shipped workflows.** Writes a JSON sidecar to `output/provenance/`
recording what a render's settings *resolved to*. It deliberately records
nothing you typed: `/history/{prompt_id}` already carries the whole graph with
every widget value and the output filenames. It records only what `/history`
structurally cannot know — the resolved sigmas, the eleven Sol closure values
(what actually ran, if anything replaced the override), the node-pack HEADs and
sage build, and the snapped frame count and canvas.

The field it exists for is `n_sparse`. That is not a setting anywhere: it is the
sigma window intersected with the sampler's schedule, so two schedulers with
identical `sol_compose` bounds can run a different number of sparse steps and
nothing in the graph, the logs or `/history` says so. Wire `SIGMAS` from
`BasicScheduler` or the field cannot be computed, and pass the sampler's
`LATENT` through it — ComfyUI orders by dependency, not graph position, so
without a real data dependency it can legally run *before* sampling.

Three states, all visible in the record rather than only in the log:
`sol: absent` (nothing installed, fine), `present` (values recorded), and
`broken` (override installed but its closure unreadable), which also raises —
most likely meaning the pack renamed parameters and `SOL_CLOSURE_KEYS` needs
updating. Joins to `/history` on `graph_sha256`, since ComfyUI does not expose
`prompt_id` to nodes.

Two cautions are in the module docstring and worth repeating: a
well-provenanced number is not a verified one, and a stamp makes invented
mechanisms *more* dangerous rather than less, because a number with a full
provenance record beside it reads as more trustworthy while carrying a wrong
causal story just as well. It records what settings resolved to, never why a
number came out the way it did.

### Use from Sol-Attn (`ComfyUI-SolAttn_triton`)

**`SolAttnPatch`** — block-sparse attention. **Must come after** the sage
node; it composes with the attention patch it finds, and reversed it
overwrites ours and you silently get sage only. Settings live in
`workflows/h3_config.py`.

**`SolAttnBlockProbe`** — diagnostic only. Runs every attention call both
sparse and dense and logs per-block error worst-first. Costs roughly
dense+sparse, so remove it once you have the numbers.

### Use from KJNodes

**`ModelPreviewOverrideKJ`** — taeh3 preview during sampling. Worth more
than any kernel knob: it lets a bad seed die in ~90 s instead of costing a
full render. Deliberately kept out of the API-format workflows, since its
decodes would land in any timing run as an unattributed cost.

### Skip, with reasons

**`MiniMaxH3MemoryEfficientSageAttentionPatch`** (KJNodes) — does the same
job as our node and patches the same key, so they conflict and the last one
applied wins. Ours additionally registers the attention override. Pick one;
there is no benefit to both.

**`MiniMaxLowVRAMAttention`** (KJNodes) — head chunking. Shrinks the
kernel's internal transients by the chunk count (~1070 MiB at 4 groups), but
turns 1000 attention calls per render into 4000. On a 24 GB 4090 freed VRAM
converts to wall-clock at a ~2.6% ceiling — weight streaming is already
hidden behind compute — so it is buying headroom you cannot spend. Take it
only if you are actually hitting OOM. As of KJNodes `35e5956` it composes
with an existing attention patch rather than conflicting.

**`MiniMaxChunkFeedForward`** (KJNodes) — at 362 frames attention peaks
around 17.8 GiB against the FFN's 9-12, so chunking the FFN lowers a peak
that is not the binding one. It is a short-clip feature; at short lengths
the two peaks are close.

**`PathchSageAttentionKJ`** — the global sage switch. It sages every
attention call in the process with no per-model guard. Prefer the
per-workflow node.

**Untested here**, not a recommendation either way: `EasyCache`,
`MiniMaxH3Cache`, `MiniMaxH3SigmaShift`, `MiniMaxH3TurboLoRA`.

### On `ResolutionSelector`

Core ComfyUI's resolution helper works from a **megapixel target**, which is
not how H3 sizes a canvas. `adapt_canvas` ignores your pixel budget entirely
and applies the short-edge and area-cap rules above. Use the table here and
type the numbers, or let the conditioning node's defaults stand.

## Node order

```
Load Diffusion Model
  -> MiniMax H3 SageAttention        (ours: kernel + attention override)
  -> SolAttnPatch                    (sparse; must be after ours)
  -> BasicScheduler / BasicGuider    (MODEL forks to both -- rewire both)
```

MODEL forks to **two** consumers, `BasicScheduler.model` and
`BasicGuider.model`. Rewiring only the guider leaves the scheduler reading
sigmas off the unpatched model, and the render still succeeds — which is why
that mistake survives. The generated workflows drive both from one variable.
