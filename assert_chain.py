"""Fail the render if the attention chain is not composed as intended.

The problem this exists for: our node registers a replacement `forward` on
each DiT attention module *and* an `optimized_attention_override`. A sparse
attention patch applied afterwards is supposed to find that override and
chain onto it. Whether it does is negotiated between two third-party
packages through a duck-typed attribute and a delegate key that neither
formally owns, and both rewrote that seam within a minute of each other on
2026-08-06.

If either side renames something, composition silently takes a different
path. There is no error and no warning. The render succeeds, looks fine, and
is slower or numerically different than the one you meant to run -- which is
indistinguishable from success unless you go read the log.

Guarding that by hand, with a three-line log grep after every restart, only
works while someone remembers to do it. This promotes the check to a hard
gate: wire it after the last model patch and the graph refuses to run when
the contract is broken.

Scope: it asserts *our* routing contract -- that sage is installed and that
anything layered on top chained rather than overwrote. It deliberately does
not assert anything about what the other package does internally, which is
not ours to pin.

**What it proves, and what it does not.** The structural checks run at patch
time and prove *registration*: the override object exists, the forward
patches are on the keys we expect. They cannot prove the composed path is
taken when attention actually fires -- the same gap as a log line that
confirms a block list parsed rather than that an exemption fired. That
distinction has cost this project several measurements.

`exercise=True` closes it by pushing one tensor through the composed
attention and reading the sparse patch's own call counters before and after.
That is call-time evidence. It costs a fraction of a second and ~176 MiB
transiently, and it degrades to a warning when the counters are not
importable, since they belong to a third-party package that may rename them.
"""
from __future__ import annotations

import logging

from comfy_api.latest import io

logger = logging.getLogger(__name__)


class SageChainAssert(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SageChainAssert",
            display_name="Assert Sage Attention Chain",
            category="model/attention/minimax",
            description=(
                "Raises if the attention chain is not composed as intended, "
                "instead of letting a silently-bypassed patch render "
                "successfully. Place after the last node that patches "
                "attention."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "require_override", default=True,
                    tooltip="Require our optimized_attention_override to still be "
                            "installed. Fails when a later patch replaced it "
                            "outright instead of chaining onto it."),
                io.Boolean.Input(
                    "require_forward_patch", default=True,
                    tooltip="Require the per-block attention forward patches to "
                            "still be present. A later patch may legitimately "
                            "own this key and cooperate -- see the note in the "
                            "failure message before turning this off."),
                io.Boolean.Input(
                    "exercise", default=True,
                    tooltip="Push one tensor through the composed attention and "
                            "assert on what actually routed, rather than only on "
                            "what is registered. This is the difference between "
                            "install-time and call-time evidence. Costs a "
                            "fraction of a second and ~176 MiB transiently."),
                io.Boolean.Input(
                    "warn_only", default=False,
                    tooltip="Log instead of raising. Defeats the point of the "
                            "node; use only while diagnosing."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @staticmethod
    def _exercise(override, to):
        """Fire one attention call through the composed path; report what ran.

        Returns (ok, detail). Structural checks above prove registration;
        this proves routing. The counters belong to the sparse package, so
        every failure to reach them is a warning rather than an assertion --
        an absent counter means "cannot tell", not "did not fire", and
        conflating those is the error this whole node exists to prevent.
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            return None, f"torch unavailable: {exc}"
        if not torch.cuda.is_available():
            return None, "no CUDA device; skipped"

        try:
            import importlib
            sol = importlib.import_module("ComfyUI-SolAttn_triton")
            # Read a delta rather than resetting: the counters are shared
            # process state and another node may be accumulating them.
            stats_fn = sol.sol_attn_stats
        except Exception:
            return None, ("sparse-attention counters not importable, so routing "
                          "could not be confirmed at call time; registration "
                          "checks above still passed")

        # min_tokens defaults to 4096 and the kernel needs head_dim 128, so a
        # genuinely small probe would be declined for being small -- which
        # would look identical to a broken chain. Named in the [B, S, H, D]
        # layout the override actually takes, so the unpack order and the
        # construction order agree; a file whose job is being trustworthy
        # should not read like it has a transposed-axis bug.
        BATCH, SEQ, HEADS, HEAD_DIM = 1, 4608, 56, 128
        dt = torch.bfloat16

        # 3 x 66 MB of probe, plus the kernel's own output and workspace, at the
        # moment the model is being staged. On a card already oversubscribed by
        # H3's stack that is a bad time to be the allocation that fails, so give
        # up the call-time evidence rather than the render.
        need = 4 * BATCH * SEQ * HEADS * HEAD_DIM * 2
        free = torch.cuda.mem_get_info()[0]
        if free < need * 4:
            return None, (f"skipped the probe: {free / 2**20:.0f} MiB free, want "
                          f"{need * 4 / 2**20:.0f} MiB headroom for a "
                          f"{need / 2**20:.0f} MiB probe. Registration checks "
                          f"above still passed; routing was not confirmed at "
                          f"call time")

        q, k, v = (torch.randn(BATCH, SEQ, HEADS, HEAD_DIM, device="cuda", dtype=dt)
                   for _ in range(3))
        before = dict(stats_fn())
        try:
            with torch.inference_mode():
                override(lambda *a, **kw: torch.zeros_like(q),
                         q, k, v, HEADS, transformer_options=dict(to))
        except Exception as exc:
            return False, f"composed attention raised on a probe call: {exc!r}"
        finally:
            del q, k, v
            torch.cuda.empty_cache()

        after = stats_fn()
        moved = {key: after.get(key, 0) - before.get(key, 0)
                 for key in set(after) | set(before)}
        moved = {key: n for key, n in moved.items() if n}
        if not moved:
            return False, ("a probe call through the composed attention "
                           "incremented no routing counter at all, which means "
                           "the composed path was not taken")
        return True, "routed as " + ", ".join(f"{k}={n}" for k, n in sorted(moved.items()))

    @classmethod
    def execute(cls, model, require_override, require_forward_patch,
                exercise, warn_only):
        problems = []

        to = model.model_options.get("transformer_options", {})
        override = to.get("optimized_attention_override")
        if require_override and override is None:
            problems.append(
                "no optimized_attention_override is installed. Either the sage "
                "node is not in this graph, or a later patch replaced the "
                "override instead of chaining onto it. A sparse-attention patch "
                "that declines a call would then fall through to ComfyUI's "
                "default attention rather than sage.")

        patches = getattr(model, "object_patches", {}) or {}
        attn_forwards = [k for k in patches
                         if k.startswith("diffusion_model.blocks.")
                         and k.endswith(".attn.forward")]
        if require_forward_patch and not attn_forwards:
            problems.append(
                "no per-block attention forward patches are installed on "
                "diffusion_model.blocks.*.attn.forward. If a cooperating "
                "low-VRAM or sparse patch has legitimately taken ownership of "
                "that key, this check is too strict for your graph and "
                "require_forward_patch can be turned off -- but confirm from "
                "the log that sage still runs before you do.")

        if exercise and override is not None:
            ok, detail = cls._exercise(override, to)
            if ok is None:
                logger.warning("[h3] chain assert: %s", detail)
            elif ok:
                logger.info("[h3] chain assert, call-time: %s", detail)
            else:
                problems.append(detail)

        if problems:
            detail = "\n".join(f"  - {p}" for p in problems)
            msg = (f"SageChainAssert: attention chain is not composed as "
                   f"intended.\n{detail}\n"
                   f"  Node order matters: the sage node must come before any "
                   f"sparse-attention patch. Reversed, the sparse patch "
                   f"overwrites it and the render silently uses sage only.")
            if warn_only:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        else:
            logger.info(
                "[h3] chain assert ok: override installed, "
                "%d attention forward patch(es) present", len(attn_forwards))

        return io.NodeOutput(model)
