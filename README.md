# ComfyUI-sageattn-ada

SageAttention kernels for consumer video DiTs on Ada (RTX 40xx / sm89).

Optimized for this sage fork code: [`https://github.com/fblissjr/SageAttention-ada/`](https://github.com/fblissjr/SageAttention-ada/)

One node today: **MiniMax H3 SageAttention**.

## What it does

MiniMax H3 runs one unmasked self-attention per DiT block over the whole
packed `[text | cond | audio | video]` sequence — 56 heads, head_dim 128,
50 blocks. At the default canvas that sequence is about 42k rows, and
attention is roughly 61% of the model's forward FLOPs. It is a good fit
for SageAttention's INT8-QK / FP8-PV kernel.

The node replaces each block's attention `forward`. Compared to going
through ComfyUI's generic attention dispatch, that also lets q/k/v stay in
the layout the fused QKV projection already produces, and lets the float
q/k/v be released as soon as their quantized forms exist.

## Measured

One `Attention` module at H3's config, packed sequence 41822 (fl2va at the
default 1344x768 canvas, 124 frames), RTX 4090, bf16. One arm per process.
Reproduce with `bench/bench_minimax_attn.py`:

| | per module call | peak allocation |
|---|---|---|
| stock ComfyUI attention | 389.15 ms | 3886 MiB |
| this node | 183.23 ms | 3451 MiB |
| | **2.12x faster** | **435 MiB lower** |

That is the whole module — QKV projection, norms, RoPE, attention, output
projection — not just the attention kernel, so it is the number that
actually applies per block. The attention kernel alone is about 2.7x.

Accuracy, via `bench/check_correctness.py`: mean relative error 0.0732
against the stock forward, on both the eager norm path and the fused
RMSNorm+RoPE path. SageAttention quantizes Q/K to INT8 and V to FP8, so
some divergence is by design; this sits at the kernel's known level.

### On a real render

The above is one module in isolation. This is a full render through a
running ComfyUI at the bundled i2v template's settings — 1344x768,
length 73, 20 steps, `res_multistep`/`simple`, `int8_convrot` weights —
warmup discarded, arms alternating, two paired runs. Reproduce with
`bench/bench_e2e_h3.py`:

| | sampler | total render |
|---|---|---|
| sage off | 141.2 s | 151.9 s |
| sage on | 82.9 s | 93.6 s |
| | **1.70x** | **1.62x** |

The two paired runs agreed to within 0.3 s on every figure. The gap
between the columns is text encode plus VAE decode, which attention
cannot touch; the sampler is 93% of total at these settings, so that gap
is small. Expect a smaller end-to-end ratio at short durations, where the
packed sequence is short enough that attention stops dominating — at
length 5 the same A/B measures 1.02x, i.e. nothing.

Peak VRAM during the render was ~20.6 GB of 24 GB, so this fits with room
to spare on a 4090 here. Longer durations or higher resolutions will
close that margin.

## Requirements

- RTX 40xx / Ada (sm89). Other architectures fall back to whatever
  SageAttention's dispatcher picks for them and are untested here.
- CUDA 12.8 or newer.
- A ComfyUI recent enough to have `comfy.ldm.minimax`.
- [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada), built from
  source, new enough to provide `sageattn_consume` (v0.7.0+ — see its
  [CHANGELOG](https://github.com/fblissjr/SageAttention-ada/blob/main/CHANGELOG.md)).
  A stock `pip install sageattention` will not work — the node checks and tells
  you so.

## Use

Drop **MiniMax H3 SageAttention** between the model loader and the
sampler. The defaults are the intended configuration; you do not need to
change anything.

Inputs:

- **model** — an H3 model. The node refuses anything else rather than
  silently doing nothing.
- **mode** (default `auto`) — `auto` lets SageAttention's dispatcher
  choose, which resolves to fp8++ on a 4090, so picking `fp8++` explicitly
  changes nothing. The explicit modes exist for bisecting a suspected
  accuracy problem. `fp16 (most accurate)` is the slowest and least lossy
  (mean relative error 0.010 vs 0.069 for the fp8 modes), and is the one
  mode that gives up the per-call memory saving — there is no consuming
  entry point for that kernel.
- **patch_token_refiner** (default off) — also patches the 2 text
  token-refiner blocks. They run over the text span only (~2k rows vs
  ~42k), so this is worth well under 1% of attention time.

If a sage call raises at runtime, the node logs once and falls back to
ComfyUI's attention for the rest of the run. The render continues.

### Longer clips are the better case

Not the worse one. Attention grows as S^2 while everything else in the
block grows as S, so the longer the clip, the larger the share of the step
sage is attacking. At length 73 the sampler speedup is 1.70x; at 124 it is
1.91x.

Measured per-call at the 1344x768 canvas, sage `fp8++` against torch's
flash backend, with q/k/v as the three views of one fused QKV buffer that a
DiT block actually produces.

Worth being explicit about whose numbers these are: **the speed below is
upstream SageAttention's kernel** — the sm89 INT8-QK / FP8-PV design from
[thu-ml](https://github.com/thu-ml/SageAttention) via
[woct0rdho](https://github.com/woct0rdho/SageAttention), which the Ada fork
ships unmodified. The fork's contribution at these lengths is that it
builds for sm89 at all and stays correct past ~99,864 rows (below), not
that it is faster.

| frames | packed rows S | sage | flash | ratio | attention share of step |
|---|---|---|---|---|---|
| 124 | 37,774 | 90.1 ms | 253.5 ms | 2.81x | ~50% |
| 209 | 63,256 | 256.0 ms | 708.2 ms | 2.77x | — |
| 311 | 93,836 | 556.5 ms | 1560.3 ms | 2.80x | — |
| 362 | 109,126 | 757.7 ms | 2107.9 ms | 2.78x | ~76% |

The kernel ratio is flat — 2.77-2.81x across a 2.9x span of sequence
length — and per-call accuracy is flat with it (mean rtol 0.0978-0.0985),
so nothing is traded for the extra length.

What changes is leverage, not the multiplier. A 362-frame render logs
49.66 s/it at 20 steps; 50 blocks times the 757.7 ms above is 37.9 s of
that, so three quarters of the clock is attention. The same step on flash
would be about 118 s/it — 39.5 min against 16.6 min for the render.

**Past 328 frames the addressing changes underneath you**, and it is worth
knowing why that is safe here. Above S=99,864 rows the element offsets in
the fused-QKV layout exceed int32, which in this layout silently zeroes the
tail of the output rather than raising.

This is an upstream defect, not one the Ada fork introduced, and it is not
obscure: kijai independently hit the same int32 wrap in Sol-Attn's own
Triton kernels and patched it on the same day, from a different direction.
Anything built on the unpatched quant kernels will hit it somewhere past
100k tokens.

Two independent fixes exist. SageAttention-ada v0.7.0 selects an int64
specialization per launch, so ordinary shapes keep int32 addressing;
v0.7.1 verified it at 362 frames, where it engages and costs 0.07% —
inside noise. KJNodes vendors its own always-int64 copy of the quant
kernels, which is why its H3 patch is safe against stock SageAttention
too. Either covers you. What you should not do is run long clips on
unpatched stock through a wrapper that does neither.

## Stacking with other attention patches

The node registers **two** things: a replacement `forward` on each of the
50 DiT attention modules, and an `optimized_attention_override`.

The forward patch is the fast path and handles every call on its own. The
override exists for patches that run ComfyUI's *stock* forward in order
to reach their own override — Kijai's
[Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton) works this
way. Without an override of ours registered, everything Sol-Attn declines
after that point (a mask, its kernel returning `None`, a kernel error)
would land on ComfyUI's default attention instead of sage. Ours chains
onto any override already present and stays in place for a later one to
chain onto, so layering is order-independent in that direction.

**Apply this node before Sol-Attn**, so Sol-Attn sees it and composes:

```
Load Diffusion Model → MiniMax H3 SageAttention → SolAttnPatch → BasicGuider
```

Worth understanding: the two do not stack per-call, they **alternate**.
Inside Sol-Attn's sigma window it runs sparse and sage is bypassed;
outside it, sage runs dense. At H3's defaults that is 14 of 20 steps
sparse.

Do **not** also enable KJNodes' *MiniMax H3 Mem Eff Sage Attention
Patch* — it patches the same keys, so whichever node runs last silently
wins and you will not know which kernel is running.

Measurements, settings worth using, and what did not hold up:
**[Experiments on sage + Sol-Attn](./docs/SOLATTN.md)**. Short version — about 1.15x on
top of sage with `int8_qk=True` and `morton=False`, no quality difference that
survived replication, and Sol-Attn renders measure consistently louder on audio
for reasons not yet established.

## Layout

```
attention.py   kernel selection, the replacement Attention.forward,
               and the optimized_attention_override
nodes.py       the ComfyUI node
bench/
  bench_minimax_attn.py      per-module speed + peak VRAM
  bench_e2e_h3.py            full render A/B against a running ComfyUI,
                             selectable arms via --arms
  check_correctness.py       patched forward vs the stock one
  check_override_routing.py  which calls the override sends to sage
                             (no CUDA needed)
```

Run bench arms one per process — peak VRAM is biased by a prior arm
training the caching allocator, and `bench_e2e_h3.py` varies the seed per
iteration because ComfyUI serves an identical graph from cache and would
otherwise report an enormous fake speedup for a render that never ran.
