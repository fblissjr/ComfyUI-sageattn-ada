# Stacking with Sol-Attn on MiniMax H3

Measured evaluation of [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)
layered on this node, on a 4090 at H3's 1344x768 / length 124 / 20-step
configuration. Sparsity and quantization attack different things, so the
question is whether they compose usefully. They do, modestly.

Everything below is a single-machine, single-workload result. Sol-Attn is
tuned for more than H3 and the numbers here say nothing about it elsewhere.

## The frontier

Sampler time, same seed, warmup discarded, arms alternating:

| config | sampler | vs nothing | vs sage | quality |
|---|---|---|---|---|
| no sage | 342.1 s | 1.00x | — | reference |
| **sage** | **178.7 s** | **1.91x** | 1.00x | indistinguishable by eye |
| sage + sol (`int8_qk`) | 160.6 s | **2.13x** | 1.11x | different, not worse |
| sage + sol, widened window | ~150 s (projected) | ~2.3x | ~1.2x | unmeasured, expect degradation |

**Sage is the easy call**: a large win with no perceptual cost and one
node. **Sol-Attn is a judgement call**: another 11% for a second node, a
sigma window to tune, and a sample that is visibly different (not worse)
from the same seed. Below about 20k packed rows it will do less, because
attention stops dominating.

## Where the time actually goes

Profiled one forward, 50 DiT blocks, device time only:

| | sage dense | sol sparse (`int8_qk`) |
|---|---|---|
| attention kernel, all 50 blocks | 4296 ms | **2668 ms** |
| per block | 85.9 ms | **53.4 ms** |
| sol routing + quant prep | — | 161 ms (2.1%) |
| everything else | 4944 ms | 4935 ms |
| whole forward | 9240 ms | 7603 ms |

Two things worth reading off that table.

**Sol-Attn's sparse kernel is 1.61x faster than sage's dense kernel.**
That is a real, large win at the kernel level.

**Non-attention time matches to 0.2%** (4944 vs 4935 ms), which is the
control: only attention changed, so the patching surface is clean.

The end-to-end result is small because of Amdahl applied twice: attention
is only ~46% of a forward, and only 14 of 20 steps fall inside Sol-Attn's
default sigma window. All 20 steps sparse would give ~1.22x on the
sampler; the window alone accounts for roughly half the gap between that
and the 1.11x measured.

## Configuration findings

- **`int8_qk=True` is worth setting.** Sol-Attn's default kernel is
  **bf16**; sage runs INT8 QK + FP8 PV. Its own tooltip says int8 helps at
  `tau<=1.5` and we run 1.2. Measured 165.4 -> 160.6 s.
- **Morton reordering did not help here** — about 2 s *slower* than plain
  Sol-Attn once warm. It is off by default; leave it off unless a
  measurement on your own shapes says otherwise.
- **`sink_conditioning="exact_kv_and_rows"` costs ~4%**, cheaper than its
  "~20%" tooltip. It runs H3's audio query rows dense. H3's audio is only
  ~250-400 rows inside a ~38k packed sequence, thin enough for a
  block-sparse router to drop, so this is the knob to reach for if audio
  quality regresses. Whether it helps is unverified here.
- The conditioning sink forces only **9 of ~591 KV blocks** exact (1.5%),
  so it is not meaningfully inflating density.

## Ordering

```
UNETLoader -> MiniMax H3 SageAttention -> SolAttnPatch -> BasicGuider
```

Sol-Attn must come second: it walks the model's existing object patches
and composes with the attention forwards it finds. Reversed, this node
overwrites its patch and you silently get sage only.

They **alternate rather than stack**. Inside the sigma window Sol-Attn
runs sparse and sage is bypassed entirely; outside it, sage runs dense.
Confirmed engaged, rather than assumed, from its own verbose logging:

```
[sol_attn] composed with 50 object-patched attention forward(s)
[sol_attn] sparse (1, 37826, 56, 128) tau=1.2 bf16
```

with no `dense: <reason>` lines and zero kernel failures.

## Measurement traps hit while producing this

Recorded because each one produced a confident, wrong number first.

- **ComfyUI caches node outputs.** Re-submitting an identical graph
  executes nothing and returns in milliseconds. The first version of the
  e2e bench reused one seed and reported a **405x speedup** for a render
  that never ran. Vary the seed per iteration and hard-fail when the timed
  node does not execute.
- **Run 1 of any Sol-Attn arm pays Triton autotune.** Comparing a warm arm
  against a cold one made Morton look like it helped by 3 points. It does
  not; it is slightly negative. Warm every arm or compare run 2 only.
- **Reasoning from wall-clock is not profiling.** From per-step arithmetic
  it looked like routing overhead was eating the sparsity benefit. The
  profile put routing at 2.1% and the real answer at Amdahl. The
  arithmetic was arithmetically fine and the conclusion was wrong.
- **Comparing finished renders numerically measures chaos, not quality.**
  At 20 steps of a flow-matching ODE any perturbation diverges the
  trajectory, so same-seed sage-on/sage-off latents differ substantially
  while both look fine. The honest instruments are fixed-input kernel
  divergence and human judgement.
- **Synthetic inputs cannot answer questions about input distribution.**
  A `torch.randn` sweep reported `smooth_k` as having no effect, which it
  would have reported whether or not the effect were real, since
  `torch.randn` has zero mean and `smooth_k` removes a channel offset.

## Reproducing

```
python bench/bench_e2e_h3.py --arms sage,sage+sol+morton+int8qk --runs 2 --length 124
```

Arms are defined in `bench/bench_e2e_h3.py`. Add a `verbose` arm to
confirm Sol-Attn engaged; its routing decisions land in the ComfyUI log
under `[sol_attn]`.
