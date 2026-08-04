# ComfyUI-sageattn-ada

SageAttention kernels for consumer video DiTs on Ada (RTX 40xx / sm89).

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

**Not yet measured: what this does to a full render.** These are
single-module numbers. A 21 GB checkpoint on a 24 GB card spends real time
streaming weights, and time saved in attention does not necessarily show
up as wall-clock. Treat the table above as a kernel result, not a promised
speedup, until you have run your own before/after on a real workflow.

## Requirements

- RTX 40xx / Ada (sm89). Other architectures fall back to whatever
  SageAttention's dispatcher picks for them and are untested here.
- CUDA 12.8 or newer.
- A ComfyUI recent enough to have `comfy.ldm.minimax`.
- The Ada SageAttention fork, built from source, new enough to provide
  `sageattn_consume`. A stock `pip install sageattention` will not work —
  the node checks and tells you so.

## Use

Drop **MiniMax H3 SageAttention** between the model loader and the
sampler. The defaults are the intended configuration; you do not need to
change anything.

Inputs:

- **model** — an H3 model. The node refuses anything else rather than
  silently doing nothing.
- **mode** (default `auto`) — `auto` lets SageAttention's dispatcher
  choose, which resolves to fp8++ on a 4090. The explicit modes exist for
  bisecting a suspected accuracy problem; `fp16 (most accurate)` is the
  slowest and least lossy.
- **patch_token_refiner** (default off) — also patches the 2 text
  token-refiner blocks. They run over the text span only (~2k rows vs
  ~42k), so this is worth well under 1% of attention time.

If a sage call raises at runtime, the node logs once and falls back to
ComfyUI's attention for the rest of the run. The render continues.

## Layout

```
attention.py   kernel selection + the replacement Attention.forward
nodes.py       the ComfyUI node
bench/         A/B bench and correctness check (run one arm per process)
```
