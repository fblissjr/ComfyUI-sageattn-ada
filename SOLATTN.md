# Stacking with Sol-Attn on MiniMax H3

Measured evaluation of [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)
layered on this node, on a 4090 at H3's 1344x768 / length 124 / 20-step
configuration. Sparsity and quantization attack different things, so the
question is whether they compose usefully. They do, modestly.

Everything below is a single-machine, single-workload result. Sol-Attn is
tuned for more than H3 and the numbers here say nothing about it elsewhere.

The sage baseline is [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada),
not stock SageAttention -- its
[CHANGELOG](https://github.com/fblissjr/SageAttention-ada/blob/main/CHANGELOG.md)
lists what differs. It has not been measured against stock, so these ratios
mean "on this fork, this model, this box".

## The frontier

Sampler time, same seed, warmup discarded, arms alternating:

| config | sampler | vs nothing | vs sage | quality |
|---|---|---|---|---|
| no sage | 342.1 s | 1.00x | — | reference |
| **sage** | **178.7 s** | **1.91x** | 1.00x | indistinguishable by eye |
| sage + sol (`int8_qk`) | 155.0 s | **2.21x** | 1.15x | no difference held up |
| sage + sol, widened window | ~150 s (projected) | ~2.3x | ~1.2x | unmeasured, expect degradation |

**Sage is the easy call**: a large win with no perceptual cost and one
node. **Sol-Attn is a judgement call**: another 15% for a second node, a
sigma window to tune, and a sample that differs from the same seed without
being better. Below about 20k packed rows it will do less, because
attention stops dominating.

### The 15% is a floor, and the length it was measured at is why

This table is length 124, where attention is ~50% of the step. Sol-Attn
only attacks attention, so its ceiling is that share — and the share climbs
with sequence length, because attention is quadratic in S and the rest of
the block is linear:

| frames | packed rows S | attention share of step |
|---|---|---|
| 124 | 37,774 | ~50% |
| 362 | 109,126 | ~76% |

Backing the 1.15x out through Amdahl at a 50% share implies Sol-Attn made
attention itself about 1.35x faster. Holding that kernel ratio and
re-applying it at a 76% share projects **~1.25x** at 362 frames, before any
credit for longer sequences having more skippable blocks to find. On a
16.6-minute render that is worth roughly three minutes.

Projection, not measurement — the arithmetic assumes the same sparsity
behaviour at 2.9x the length, which is exactly the assumption long
sequences are most likely to break. It is the strongest available reason to
re-run this evaluation at 362 frames rather than to trust the number.

### Three Sol-Attn changes postdate this evaluation

Measured Aug 4; Sol-Attn shipped these Aug 4-6, so nothing below is
reflected in the table above.

- **`int8_pv` (new, defaults on).** Runs the exact branch's P@V in INT8 as
  well as QK. Upstream's note is that PV and QK cost the same, so this is
  "the other half of the int8 win" — our 1.15x was `int8_qk` only.
- **`SolAttnBlockProbe` + `dense_blocks`.** The probe computes every
  attention call both sparse and dense and logs per-block relative error
  worst-first; the worst blocks then go in `dense_blocks` to stay exact.
  This is the direct instrument for the audio question left open below —
  it answers per block which approximations actually reach the output,
  instead of inferring it from listening.
- **`morton_curve="2d_frame"` (new default).** Z-orders within each frame
  and leaves frame order alone, motivated by H3's frame spacing being
  non-uniform. Our evaluation ran morton off, and the failure mode it
  addresses is length-dependent, so this is worth revisiting at 362 frames
  specifically.

Upstream also fixed an int32 overflow in Sol-Attn's own Triton kernels on
Aug 4 (`9cab9a0`) and a Morton corruption at certain sizes (`e353f6d`).
Both are in the size range long clips reach, so any long-sequence
Sol-Attn measurement needs a build at or after those commits to mean
anything.

## Quality

Treat Sol-Attn as a speed knob. No quality difference held up.

| seed | arm | blind | verdict |
|---|---|---|---|
| 801 | sol+morton+int8qk | no | Sol-Attn better |
| 701 | plain sol | yes | different, neither better |
| 702 | plain sol | yes | nearly identical |
| 1001 | sol+morton+int8qk | no | same |

One positive in four, and it was the first pair looked at -- before there was
any sense of how much these samples vary seed to seed. A later pair on the same
arm at a different seed came back "same", unblinded, so an expectation effect
had its chance and did not appear.

Limits of this, which are severe:

- One observer, one prompt, four pairs.
- Three of the four judgments came late in a long session. Fatigue pushes
  toward "these look the same", which is the direction the nulls point, so
  "no difference" and "stopped discriminating" are not separable here.
- The prompt is slow-camera, diffuse fog, ambient audio. A block-sparse artifact
  would more likely surface in fast motion, fine repeated detail, or on-screen
  text -- none of which this scene has.

For scale on the noise floor: the same observer called one plain-sage render
"dramatically more interesting" than two others differing only by seed.

The renders are kept, so re-judging cold is the cheap way to firm this up.

**Measured: Sol-Attn renders are consistently louder.** Noticed by ear, then
confirmed with `ffmpeg -af volumedetect` across all three same-seed pairs:

| pair | seed | mean dB (sage -> sol) | peak dB (sage -> sol) |
|---|---|---|---|
| 00031/00032 | 1001 | -40.2 -> -39.6 (+0.6) | -25.6 -> -24.6 (+1.0) |
| 00033/00034 | 1002 | -39.6 -> -38.9 (+0.7) | -24.8 -> -23.5 (+1.3) |
| 00035/00036 | 1003 | -36.9 -> -35.2 (+1.7) | -20.8 -> -14.8 (+6.0) |

Louder in 3 of 3, on both mean and peak. This does not mean *better* -- it means
Sol-Attn measurably changes the audio path, which nothing else in this session
established.

Plausible mechanism, untested: attention output is a weighted average, and
sparse attention drops low-weight blocks then renormalizes over what remains, so
the result leans harder on the strongest matches. Less averaging reads as more
dynamic range. The +6.0 dB peak against a +1.7 dB mean has that shape --
transients sharpening rather than everything rising uniformly. If that is what
is happening, "sounds better" and "is less faithful" would both be true at once:
punchier is more pleasing and less accurate.

Two things remain open and point opposite ways. H3's audio is ~250-400 rows in a
~38k sequence and Sol-Attn reports `dense query blocks (0, 0)` -- no query rows
dense -- so sparsity should *hurt* audio; yet forcing those rows dense
(`exact_kv_and_rows`) also sounded better. Both cannot be right.

What was never checked: accuracy against the sage output rather than loudness.
The files exist, so that is a listening test, not a render.

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
and the 1.15x measured.

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
