#!/usr/bin/env python3
"""Check which calls the attention override sends to sage, and which it declines.

The override exists for one situation: another patch (Sol-Attn) runs
ComfyUI's stock attention forward to reach its own override, bypassing our
forward patch. Anything that override then declines would land on
ComfyUI's default attention rather than sage, so we register ours as the
fallback for it to chain onto.

That makes the routing decision the whole value of the code, and every
wrong decision is silent: sending a masked call to sage returns plausible
numbers computed without the mask, and declining an eligible one just runs
slower. Neither raises. So the decisions are asserted directly, with a
fake kernel and a fake fallback standing in for the real ones -- no CUDA,
no model, runnable anywhere.

Claims, i.e. what breaks if a case is deleted:
  eligible->sage        the override does anything at all
  mask->fallback        sage has no mask support on this path; sending a
                        masked call would silently drop the mask
  scale->fallback       a custom softmax scale is not plumbed through here
  fp32/head_dim         sage's preconditions; violating them raises inside
                        the kernel instead of degrading
  chains to previous    an override already on the model must still run,
                        or we clobber whatever the user layered underneath
  raises->fallback      a kernel failure must degrade, not kill the render.
                        This one caught a real bug: an earlier version
                        released q/k/v to the kernel for the memory saving,
                        which left them unbound in the fallback's closure,
                        so a kernel error became a NameError.
  NHD reshape path      the non-skip_reshape branch, which reshapes in and
                        out; an error here corrupts shape, not values

    python bench/check_override_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention import make_sage_override

B, H, S, D = 1, 4, 64, 64


class Spy:
    """Counts which of the three destinations a call reached.

    Counters are n_-prefixed so they cannot shadow the callables; an
    earlier version named a counter `func` alongside a method `func`.
    """

    def __init__(self):
        self.n_kernel = self.n_func = self.n_prev = 0

    def kernel_fn(self, qkv, **_kw):
        self.n_kernel += 1
        q = qkv[0]
        qkv.clear()
        return torch.zeros_like(q)

    def fallback(self, q, _k, _v, heads, **_kw):
        self.n_func += 1
        return torch.zeros(q.shape[0], q.shape[2], heads * q.shape[3])

    def previous(self, _func, q, _k, _v, heads, **_kw):
        self.n_prev += 1
        return torch.zeros(q.shape[0], q.shape[2], heads * q.shape[3])


def main() -> int:
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    hnd = dict(skip_reshape=True, skip_output_reshape=True)
    failures = []

    def case(name, *, want_kernel, want_func, want_prev, build=None, call=None):
        spy = Spy()
        fallback_fn = spy.fallback
        ov = make_sage_override(
            build or spy.kernel_fn, {"is_causal": False},
            previous=(spy.previous if want_prev else None))
        out = (call or (lambda o, f: o(f, q, q, q, H, **hnd)))(ov, fallback_fn)
        got = (spy.n_kernel, spy.n_func, spy.n_prev)
        want = (want_kernel, want_func, want_prev)
        ok = got == want and out is not None
        print(f"  {'ok  ' if ok else 'FAIL'} {name:24s} "
              f"kernel/func/prev={got} want={want}")
        if not ok:
            failures.append(name)

    def boom(qkv, **kw):
        raise RuntimeError("simulated kernel failure")

    case("eligible->sage", want_kernel=1, want_func=0, want_prev=0)
    case("mask->fallback", want_kernel=0, want_func=1, want_prev=0,
         call=lambda o, f: o(f, q, q, q, H, mask=torch.ones(S, S), **hnd))
    case("scale->fallback", want_kernel=0, want_func=1, want_prev=0,
         call=lambda o, f: o(f, q, q, q, H, scale=0.5, **hnd))
    case("fp32->fallback", want_kernel=0, want_func=1, want_prev=0,
         call=lambda o, f: o(f, q.float(), q.float(), q.float(), H, **hnd))
    case("head_dim>128->fallback", want_kernel=0, want_func=1, want_prev=0,
         call=lambda o, f: (lambda b: o(f, b, b, b, H, **hnd))(
             torch.randn(B, H, S, 192, dtype=torch.bfloat16)))
    case("chains to previous", want_kernel=0, want_func=0, want_prev=1,
         call=lambda o, f: o(f, q, q, q, H, mask=torch.ones(S, S), **hnd))
    case("kernel raises->fallback", want_kernel=0, want_func=1, want_prev=0,
         build=boom)
    case("NHD reshape path", want_kernel=1, want_func=0, want_prev=0,
         call=lambda o, f: (lambda t: o(f, t, t, t, H, skip_reshape=False,
                                        skip_output_reshape=False))(
             torch.randn(B, S, H * D, dtype=torch.bfloat16)))

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll routing cases behave as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
