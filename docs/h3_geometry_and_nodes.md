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
