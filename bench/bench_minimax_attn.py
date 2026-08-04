#!/usr/bin/env python3
"""A/B the patched MiniMax H3 attention forward against the stock one.

Builds a single real `comfy.ldm.minimax.model.Attention` module at H3's
config and drives it directly, so this measures the thing the node
actually replaces -- including the fused qkv projection whose output q, k
and v are all views of, which is what decides how much memory can
actually be released mid-call.

Run one arm per process. Both arms allocate multi-GiB tensors, and a
prior arm trains the caching allocator in a way that biases whatever runs
second:

    python bench/bench_minimax_attn.py stock
    python bench/bench_minimax_attn.py sage

Needs ComfyUI importable (run from the ComfyUI root or with it on
PYTHONPATH) and about 8 GiB of free VRAM at the default shape.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# H3's attention config, and the packed sequence length for fl2va at the
# node's default canvas (1344x768, 124 frames).
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
SEQ_DEFAULT = 41822
MiB = 2**20


def build_attention(device, dtype):
    import comfy.ops
    from comfy.ldm.minimax.model import Attention

    return Attention(
        HIDDEN, HEADS, HEAD_DIM, 1e-5,
        dtype=dtype, device=device, operations=comfy.ops.manual_cast,
    ).to(device)


def timed(fn, warmup=1, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=["stock", "sage"])
    ap.add_argument("--seq", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--mode", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    attn = build_attention(device, dtype)
    x = torch.randn(args.seq, HIDDEN, device=device, dtype=dtype)

    if args.arm == "sage":
        from attention import build_kernel
        from attention import make_minimax_attn_forward

        kernel_fn, kernel_kwargs = build_kernel(args.mode)
        forward = make_minimax_attn_forward(kernel_fn, kernel_kwargs)
        attn.forward = forward.__get__(attn, attn.__class__)

    # rope_freqs=None exercises the eager q_norm/k_norm path, which keeps
    # the bench independent of comfy-kitchen's fused rope kernel. The
    # attention shape and the qkv aliasing -- the two things being
    # measured -- are identical either way.
    call = lambda: attn(x, rope_freqs=None)

    call()  # allocate autotune scratch before the peak is recorded
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = call()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MiB
    del out

    ms = timed(call)
    print(
        f"{args.arm:6s} seq={args.seq} mode={args.mode}  "
        f"{ms:8.2f} ms   peak {peak:7.0f} MiB"
    )


if __name__ == "__main__":
    main()
