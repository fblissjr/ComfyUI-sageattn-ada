"""Bench-only: stamp what a render's settings actually RESOLVED to.

ComfyUI already records everything you *typed*. `/history/{prompt_id}` carries
the whole prompt graph with every widget value and the output filenames, for
free. So this node deliberately does not re-record any of that. It records only
what `/history` structurally cannot know:

- resolved sigmas from `sol_compose`, and the sparse-step count computed from
  them against the actual schedule
- the eleven Sol closure values, which are what really ran if anything
  downstream replaced the override after it was installed
- node-pack HEADs and the sage build, which live outside the graph entirely
- the snapped frame count and resolved canvas, because requested != actual

The single field this exists for is `n_sparse`. It is not a setting anywhere:
it is the intersection of the sigma window with the sampler's schedule, so two
schedulers with identical `sol_compose` bounds can run a different number of
sparse steps and nothing in the graph, the logs or `/history` says so.

## Two cautions, both load-bearing

**A well-provenanced number is not a verified one.** A stamp makes bookkeeping
failures visible and does nothing about invented mechanisms — and it makes the
latter *more* dangerous, because a number with a full provenance record beside
it reads as more trustworthy while being exactly as capable of carrying a wrong
causal story. Recording state and explaining a result are different jobs.

**This records what settings resolved to, never why a number came out the way
it did.** Those get confused precisely because the stamp sits next to the
number. A mechanism claim still needs both arms instrumented.

## Why bench-only

It reads another node pack's closure internals, so it breaks when that pack
changes. On the bench surface that breakage is cheap and expected. In a shipped
workflow it would break in a user's render, which is the wrong place. Wire it
in benches; keep it out of the shipped workflows.

## Joining a stamp to a render

ComfyUI does not expose `prompt_id` to nodes (`io.Hidden` has `unique_id`,
`prompt`, `extra_pnginfo`, `dynprompt` and no id), so the stamp keys on a
canonical hash of the prompt graph instead. `/history` entries carry the same
graph, so hashing each one's `prompt[2]` the same way joins the two and gets you
the graph, the timings and the output filename together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

import folder_paths
from comfy_api.latest import io

logger = logging.getLogger(__name__)

STAMP_SCHEMA_VERSION = 1

# Every Sol setting that exists only inside the override closure. Anything here
# that introspection cannot reach becomes an explicit "not detected" IN THE
# RECORD -- a reader diffing two sidecars never sees a log line, and a silently
# absent key is indistinguishable from a setting that was never on.
SOL_CLOSURE_KEYS = (
    "tau", "min_tokens", "sigma_start", "sigma_end", "verbose", "int8_qk",
    "sink_conditioning", "use_tma", "dense_blocks", "tau_profile", "int8_pv",
)

NOT_DETECTED = "not detected"


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return NOT_DETECTED
        head = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        return head + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return NOT_DETECTED


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sol_state(transformer_options, sigmas):
    """Three states, and the caller must be able to tell them apart.

    absent   -- no override installed; nothing to record, not an error
    broken   -- override installed but its settings are unreachable; raise
    present  -- resolved values recorded
    """
    override = transformer_options.get("optimized_attention_override")
    if override is None:
        return {"state": "absent"}

    state: dict[str, object] = {"state": "present"}
    compose = transformer_options.get("sol_compose")
    state["sol_compose"] = {k: _jsonable(v) for k, v in compose.items()} if compose else NOT_DETECTED
    state["morton"] = bool(transformer_options.get("sol_morton", False))
    state["morton_curve"] = _jsonable(transformer_options.get("sol_morton_curve"))

    freevars = getattr(getattr(override, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(override, "__closure__", None) or ()
    reached = {}
    for name, cell in zip(freevars, cells):
        if name in SOL_CLOSURE_KEYS:
            try:
                reached[name] = _jsonable(cell.cell_contents)
            except ValueError:  # empty cell
                pass
    # setdefault, not a plain dict build: an upstream rename drops the name out
    # of co_freevars, and this turns that into an explicit "cannot tell" rather
    # than a key that quietly disappears from the record.
    for key in SOL_CLOSURE_KEYS:
        reached.setdefault(key, NOT_DETECTED)
    state["closure"] = reached

    if all(v == NOT_DETECTED for v in reached.values()):
        state["state"] = "broken"
        return state

    # n_sparse: the whole reason this node exists. Not readable anywhere -- it
    # is the window intersected with the schedule.
    s_start, s_end = reached.get("sigma_start"), reached.get("sigma_end")
    if sigmas is None or not isinstance(s_start, (int, float)) or not isinstance(s_end, (int, float)):
        state["n_sparse"] = NOT_DETECTED
    else:
        # The model is evaluated at sigmas[0..steps-1]; the terminal sigma never
        # gets an eval. Counting the full tensor overcounts by one whenever the
        # terminal sigma falls inside the window -- which it does exactly when
        # end_percent == 1.0, since percent_to_sigma(1.0) is 0.0. That is the
        # widest-window arm, i.e. the one most likely to be measured.
        evals = sigmas[:-1]
        inside = (evals <= s_start) & (evals >= s_end)
        state["n_sparse"] = int(inside.sum())
        state["n_evals"] = int(evals.numel())
    return state


class MiniMaxH3ProvenanceStamp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ProvenanceStamp",
            display_name="MiniMax H3 Provenance Stamp (bench)",
            category="model/debug/minimax",
            description=(
                "BENCH ONLY. Writes a JSON sidecar recording what this render's "
                "settings resolved to -- sparse-step count, Sol closure values, "
                "node-pack HEADs, snapped frame count and canvas. Does not record "
                "what you typed; /history already has that. Wire the sampler's "
                "LATENT through it so it runs after sampling."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip=(
                    "Pass the sampler's output through. Required for ordering: "
                    "ComfyUI orders by dependency, not graph position, so without "
                    "a real data dependency this can legally run BEFORE sampling.")),
                io.Model.Input("model"),
                io.Sigmas.Input("sigmas", optional=True, tooltip=(
                    "From BasicScheduler. Without it n_sparse cannot be computed, "
                    "which is the one field this node exists for.")),
                io.String.Input("note", default="", multiline=False, optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
            hidden=[io.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, model, sigmas=None, note="") -> io.NodeOutput:
        to = (model.model_options or {}).get("transformer_options", {}) or {}
        sol = _sol_state(to, sigmas)

        here = Path(__file__).resolve().parent
        packs = here.parent
        record = {
            "stamp_schema_version": STAMP_SCHEMA_VERSION,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
            "sol": sol,
            "builds": {
                "h3_explorations": _git_head(here),
                "sol_attn": _git_head(packs / "ComfyUI-SolAttn_triton"),
                "comfyui": _git_head(Path(folder_paths.base_path)),
                "sageattention": cls._sage_version(),
            },
            "resolved": cls._geometry(latent),
        }

        prompt = getattr(cls.hidden, "prompt", None)
        if prompt is not None:
            blob = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
            record["graph_sha256"] = hashlib.sha256(blob).hexdigest()
        else:
            record["graph_sha256"] = NOT_DETECTED

        out_dir = Path(folder_paths.get_output_directory()) / "provenance"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Second-resolution plus graph hash is NOT unique: re-running an
        # identical graph for variance collides, and the later stamp silently
        # replaces the earlier one -- a provenance tool losing provenance.
        stem = record["utc"].replace(":", "").replace("-", "")
        base = f"stamp_{stem}_{record['graph_sha256'][:8]}"
        path = out_dir / f"{base}.json"
        serial = 0
        while path.exists():
            serial += 1
            path = out_dir / f"{base}_{serial:02d}.json"
        record["stamp_serial"] = serial
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

        if sol["state"] == "broken":
            raise RuntimeError(
                "Sol-Attn's attention override is installed but none of its settings "
                f"could be read from the closure. The stamp at {path.name} records "
                "'broken' rather than a hollow set of defaults, but treat any "
                "measurement from this render as unattributed -- most likely the "
                "pack renamed its parameters and SOL_CLOSURE_KEYS needs updating."
            )

        logger.info(
            "[h3] provenance -> %s (sol=%s, n_sparse=%s)",
            path.name, sol["state"], sol.get("n_sparse", "n/a"),
        )
        return io.NodeOutput(latent)

    @staticmethod
    def _sage_version():
        try:
            import sageattention
            ver = getattr(sageattention, "__version__", NOT_DETECTED)
            src = Path(sageattention.__file__).resolve().parent.parent
            return f"{ver}@{_git_head(src)}"
        except Exception:
            return NOT_DETECTED

    @staticmethod
    def _geometry(latent):
        """Resolved, not requested -- today's own finding is that they differ."""
        try:
            samples = latent["samples"]
            video = samples[0] if not torch.is_tensor(samples) else samples
            _, _, latent_t, lh, lw = video.shape
            # video latent frames are 5n+2 for 17n+5 pixel frames
            frames = 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 else NOT_DETECTED
            return {
                "canvas": [int(lw) * 16, int(lh) * 16],
                "latent_t": int(latent_t),
                "frame_count": frames,
                "duration_s": round(frames / 24, 4) if isinstance(frames, int) else NOT_DETECTED,
                "packed_rows_per_frame": int(lh // 2) * int(lw // 2),
            }
        except Exception as exc:
            # Type only, never the message. A sidecar is meant to be shared
            # next to a render, and an exception string is the one field here
            # that can carry a filesystem path out of the machine.
            return {"error": type(exc).__name__}
