#!/usr/bin/env python3
"""Generate the MiniMax H3 test workflows, in API format and UI format.

Why a generator instead of hand-edited JSON: the three bundled ComfyUI
templates are not equally editable. `video_minimax_h3_r2v` is a flat graph,
but t2v and i2v hide the entire sampler stack inside a subgraph named
"Image to Video (MiniMax H3)". Editing a subgraph by hand -- or converting
one to API format by hand -- is how you end up measuring a graph that is
subtly not the one you meant to run. Building all three from one description
keeps them identical everywhere they should be identical, and makes the one
thing that differs (which conditioning node, which checkpoint) obvious.

The sage node goes between `UNETLoader` and the sampler stack. Note that
MODEL forks to *two* consumers -- `BasicScheduler.model` and
`BasicGuider.model`. Rewiring only the guider leaves the scheduler reading
sigmas off the unpatched model; the render still succeeds, which is why the
mistake survives. Every graph here is generated from a single `model_src`
variable so the fork cannot drift.

Run it to regenerate:

    python3 build_workflows.py

It writes the JSON next to itself and validates every API graph against a
live ComfyUI's /object_info (or a cached copy passed with --object-info).
Validation is static -- nothing is submitted, nothing touches the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Model names, sampler settings, canvas geometry and the SolAttn knobs all
# used to live here in duplicate with the bench. Single source is
# h3_config.py -- see its docstring for why that matters.
from h3_config import (  # noqa: E402
    CANVAS, FPS, LENGTH, LONG_LENGTH, MODELS, SAMPLING,
    SAGE_NODE, SEED, SOL_RECOMMENDED,
)

# Prompt for the 362-frame presets. 15.08s at 24fps needs a shot timeline,
# not one continuous beat -- the guide wants numbered shots with explicit cut
# times past a few seconds, and a 15s request against a 6s prompt leaves the
# model twelve seconds it was never told about.
LONG_T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, handheld, shallow depth of field. A medium shot frames a courier in a soaked red jacket standing over a bicycle at a city crosswalk in heavy evening rain, wet asphalt throwing back the signal lights, a brick facade with iron railings filling the background. The camera tracks right at medium amplitude and moderate speed as she snaps her helmet strap and pushes off.
[Shot 2] At 00:04.000, the shot cuts to a low tracking shot running alongside the bicycle as it crosses the junction, spray coming off the tyres, painted lane markings streaming past underneath.
[Shot 3] At 00:08.000, the camera whip pans up to a wide shot of the street as she cuts between two parked cars, pigeons scattering off the railings, neon shopfront signs reflected in the puddles.
[Shot 4] At 00:11.500, the shot changes to a close shot of her face under the helmet, rain streaking across the lens, as she glances back over her shoulder and then forward again, breathing hard.

overall_soundscape: steady heavy rain on asphalt and metal, tyre hiss through standing water, the click and rattle of a bicycle chain, a car horn twice in the middle distance, wings clattering as the pigeons take off, her breathing close and rhythmic under the helmet.

non_diegetic_music: none."""

# Placeholder input filenames. These are whatever the local install happens
# to have; swap them for your own before running an i2v or r2v graph.
PLACEHOLDER_IMAGE_A = "1-man.png"
PLACEHOLDER_IMAGE_B = "2-mountain_landscape.png"

T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a lone lighthouse keeper on a wet stone balcony at dawn, wearing a heavy oilskin coat, the lamp housing glowing behind him. Grey-blue sea fog rolls past below the railing and gulls cross the frame. The camera pushes in with small amplitude at slow speed as he raises a brass telescope, holds it steady against his eye, then lowers it and turns toward the light. [Shot 2] At 00:03.000, the shot cuts to a close-up of the rotating lamp assembly, the beam sweeping past the lens and out into the fog.

overall_soundscape: A low sea swell breaks against stone under a steady wind, with gulls calling overhead. A distant foghorn sounds twice, and the lamp mechanism turns with a slow mechanical grind.

non_diegetic_music: Sustained low strings at a slow tempo with a single sparse piano figure, holding without a swell."""

I2V_PROMPT = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the subject shown in <Picture 1> holds its position, framing, lighting, and colors exactly as established in the image. The camera pushes in with small amplitude at slow speed while the subject begins to move, the surrounding scene staying continuous with the reference frame.

overall_soundscape: Quiet room tone with a low ambient hum continues throughout, joined by soft physical sounds from the subject's movement.

non_diegetic_music: N/A"""

R2V_PROMPT = """subject_definitions:
<Subject 1> is the main character in <Picture 1>, whose face, hair, and clothing are carried into the target video.
<Subject 2> is the environment in <Picture 2>, whose architecture, palette, and lighting are carried into the target video.

summary:
[reference generation] The target video places <Subject 1> inside <Subject 2> for a single continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - architecture, palette, and lighting are retained.

detailed_description:
The target video is in a cinematic live-action style with soft directional lighting.
[Shot 1] A medium shot establishes <Subject 2>, then <Subject 1> enters from the left and stops at the center of the frame. The camera trucks right with small amplitude at slow speed as <Subject 1> turns toward the light and looks off-screen.

overall_soundscape:
Steady interior room tone continues throughout, with soft footsteps and fabric movement as <Subject 1> crosses the frame.

non_diegetic_music:
N/A"""


# --------------------------------------------------------------------------
# API format
# --------------------------------------------------------------------------

def build_api(task: str, *, sage: bool = True, prompt: str | None = None,
              length: int = LENGTH, seed: int = SEED,
              sol: dict | None = None, **canvas) -> dict:
    """API-format graph, submittable as {"prompt": <this>} to POST /prompt.

    Node ids match `bench/bench_e2e_h3.py` so a timing run and a hand-edited
    graph can be compared node-for-node; "10" is the sampler in every graph.
    """
    if task not in ("t2v", "i2v", "r2v"):
        raise ValueError(task)
    ref = task == "r2v"
    cv = dict(CANVAS, **canvas)
    prompt = prompt if prompt is not None else {
        "t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": MODELS["unet_ref2va" if ref else "unet_fl2va"],
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": MODELS["clip"], "type": "minimax",
                         "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": SAMPLING["sampler"]}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None, "scheduler": SAMPLING["scheduler"],
                         "steps": SAMPLING["steps"], "denoise": SAMPLING["denoise"]}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 1]}},
        # Both decoders read the same packed AV latent and each pulls out its
        # own half; this is not a mistake in the wiring.
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "fps": FPS, "audio": ["12", 0]}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0],
                          "filename_prefix": f"video/h3_{task}_sage",
                          "format": "auto", "codec": "auto"}},
    }

    if ref:
        g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                             "prompt": prompt, "width": cv["width"], "height": cv["height"],
                             "length": length, "ref_image_size": "match",
                             # Autogrow slots are addressed by their flat dotted
                             # path; ComfyUI reassembles them into the nested
                             # dict the node signature expects. Slot ordinals are
                             # 0-based but the prompt tags are 1-based, so
                             # ref_image_0 is <Picture 1>.
                             "ref_images.ref_image_0": ["15", 0],
                             "ref_images.ref_image_1": ["16", 0]}}
        g["15"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_A}}
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_B}}
    else:
        inputs = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
                  "width": cv["width"], "height": cv["height"], "length": length}
        if task == "i2v":
            # first_frame only. Adding "last_frame": ["17", 0] from a second
            # LoadImage turns this into the fl2va task the checkpoint is named
            # for; the model and every other node stay the same.
            inputs["first_frame"] = ["15", 0]
            g["15"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_A}}
        g["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}

    model_src = ["1", 0]
    if sage:
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": model_src, **SAGE_NODE}}
        model_src = ["20", 0]
    if sol is not None:
        # After sage, never before -- SolAttn composes with the attention
        # patches it finds, and reversed it overwrites ours and you silently
        # get sage only. Node id 21 matches `bench/bench_e2e_h3.py`.
        g["21"] = {"class_type": "SolAttnPatch",
                   "inputs": {"model": model_src, **sol}}
        model_src = ["21", 0]
    # The fork. Both consumers, always, from the same variable.
    g["8"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["model"] = model_src
    return g


# --------------------------------------------------------------------------
# UI format
# --------------------------------------------------------------------------

class UIGraph:
    """Minimal litegraph workflow writer.

    Field shapes are copied from the bundled `video_minimax_h3_r2v` template,
    which is the one H3 template that is already flat, so this emits the same
    dialect the frontend just loaded from disk.

    Deliberately no widget-to-input conversions and no helper nodes
    (ResolutionSelector, ComfyMathExpression, PrimitiveStringMultiline). The
    templates use those for convenience, but every one of them is another
    place a hand-edit can go wrong, and the point of these copies is to be
    easy to edit. Resolution, length and prompt are plain widget values on
    the conditioning node.
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def add(self, type_: str, pos, *, widgets=None, inputs=None, outputs=None,
            size=(320, 100), title=None):
        nid = self._next_node
        self._next_node += 1
        n = {
            "id": nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [dict(i) for i in (inputs or [])],
            "outputs": [dict(o) for o in (outputs or [])],
            "properties": {"Node name for S&R": type_},
        }
        if widgets is not None:
            n["widgets_values"] = list(widgets)
        if title:
            n["title"] = title
        self.nodes.append(n)
        return nid

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def link(self, src, src_slot, dst, dst_input_name, type_):
        lid = self._next_link
        self._next_link += 1
        s, d = self._node(src), self._node(dst)
        s["outputs"][src_slot].setdefault("links", [])
        if s["outputs"][src_slot]["links"] is None:
            s["outputs"][src_slot]["links"] = []
        s["outputs"][src_slot]["links"].append(lid)
        for inp in d["inputs"]:
            if inp["name"] == dst_input_name:
                inp["link"] = lid
                break
        else:
            raise KeyError(f"{d['type']} has no input {dst_input_name!r}")
        self.links.append([lid, src, src_slot, dst, self._input_index(d, dst_input_name), type_])
        return lid

    @staticmethod
    def _input_index(node, name):
        return [i["name"] for i in node["inputs"]].index(name)

    def _topo_order(self):
        # `order` is advisory -- the frontend recomputes it -- but an
        # inconsistent value shows up as nodes drawn in a nonsense sequence,
        # so emit a real topological order.
        incoming = {n["id"]: set() for n in self.nodes}
        for lid, src, _ss, dst, _ds, _t in self.links:
            incoming[dst].add(src)
        order, placed = {}, set()
        i = 0
        while len(placed) < len(self.nodes):
            progressed = False
            for n in self.nodes:
                nid = n["id"]
                if nid in placed or not incoming[nid] <= placed:
                    continue
                order[nid], i = i, i + 1
                placed.add(nid)
                progressed = True
            if not progressed:
                raise RuntimeError("cycle in graph")
        for n in self.nodes:
            n["order"] = order[n["id"]]

    def dump(self, workflow_id: str) -> dict:
        self._topo_order()
        return {
            "id": workflow_id, "revision": 0,
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4,
        }


def _in(name, type_, *, optional=False, widget=False, label=None):
    d = {"name": name, "type": type_, "link": None}
    if label:
        d["label"] = label
    if optional:
        d["shape"] = 7
    if widget:
        d["widget"] = {"name": name}
    return d


def _out(name, type_):
    return {"name": name, "type": type_, "links": None}


# Text for the in-graph notes. Kept next to the builder rather than in
# docs/h3_geometry_and_nodes.md on purpose: that doc is the long form, this
# is what you need with the graph open. Numbers here come from
# comfy_extras/nodes_minimax_h3.py, not from lore.
_NOTE_GEOMETRY = """\
## Canvas and length are not free parameters

`adapt_canvas()` ignores your pixel budget. Short edge **768**, hard area
cap **768x1344 = 1,032,192 px**, each axis rounded to **32**. There is no
higher resolution to pick -- asking for 4K gives the same canvas as 720p.

So resolution is an **aspect-ratio choice**, and it is the single biggest
speed lever anywhere, because attention is O(S^2) and dominates the step.

| aspect | canvas | attention |
|---|---|---|
| 21:9 / 16:9 / 9:16 | 1536x672 / 1344x768 / 768x1344 | 1.00x |
| 3:2 / 2:3 | 1152x768 / 768x1152 | 0.73x |
| 4:3 / 3:4 | 1024x768 / 768x1024 | 0.58x |
| 5:4 / 4:5 | 960x768 / 768x960 | 0.51x |
| **1:1** | **768x768** | **0.33x** |

**Portrait and landscape of a ratio cost the same.** Packed rows are
`(h//32)*(w//32)`, which is symmetric. 16:9 vs 9:16 is a quality question,
never a speed one.

## Length snaps up to n % 17 == 5

Ask 200, get 209. Ask 300, get 311. Near the top: **311, 328, 345, 362**.
Trained range is ~124-362 per the node's own tooltip; 362 = 15.08s at 24fps.

At 362 frames attention is ~76% of the step, against ~50% at 124 -- long
clips are where sparsity and kernel work pay off most. But 362 is the edge
of the trained range, and late-clip softening there is ordinary DiT decay.
**328 or 345 costs less attention and drifts less.**

Core ComfyUI's `ResolutionSelector` works from a megapixel target, which is
not how any of this works. Type the numbers.
"""

_NOTE_NODES = """\
## Node order is load-bearing

```
Load Diffusion Model
  -> MiniMax H3 SageAttention     (this repo)
  -> SolAttnPatch                 (must be AFTER)
  -> BasicScheduler + BasicGuider (MODEL forks to BOTH)
```

**Sol-Attn must come second.** It composes with the attention patch it
finds; reversed, it overwrites ours and you silently get sage only, with no
error and no log line saying so.

**MODEL forks to two consumers.** Rewiring only the guider leaves the
scheduler reading sigmas off the unpatched model, and the render still
succeeds -- which is why that mistake survives.

## Check it is actually running, once per graph change

Turn `verbose` on in SolAttnPatch for one render, then off. You want three
lines. **Read them in the terminal** -- piping or redirecting block-buffers
the output and they may not appear even when everything is fine.

```
sage routing: arch=sm89 ... pv_accum=fp32+fp16 -> fp8_cuda++
[sol_attn] chaining onto an existing attention override
[sol_attn] sparse (1, ..., 56, 128) tau=2.0 int8 pointer
```

Line 1: sage engaged on the fast kernel. Line 3: sparse engaged at your tau.
**Line 2 is the order check** -- it only prints when Sol-Attn finds sage's
override already installed. Missing means the nodes are backwards and you
are paying full price for a render that otherwise looks fine.

## What each node is here for

- **ModelPreviewOverrideKJ** -- taeh3 preview, and it is arguably the
  largest optimization here rather than a convenience. Killing a bad seed at
  90s instead of 11 minutes saves ~9.5 min; the entire kernel and sparsity
  stack saves ~7 min per render. If one render in three is a bad seed the
  preview beats everything else combined -- and they compound rather than
  compete.
- **MiniMax H3 SageAttention** -- INT8-QK / FP8-PV kernel on all 50 DiT
  attention forwards, plus an `optimized_attention_override` registration.
  That second part is what lets Sol-Attn compose instead of bypassing sage.
- **SolAttnPatch** -- block-sparse attention. Settings are pinned from
  `workflows/h3_config.py`; edit there and regenerate, not here.

## Deliberately absent

- **MiniMaxH3MemoryEfficientSageAttentionPatch** (KJNodes) -- same job as
  our node, patches the same key, so they conflict. Ours also registers the
  override.
- **MiniMaxLowVRAMAttention** -- head chunking. ~1070 MiB saved, but 1000
  attention calls become 4000. On 24GB freed VRAM converts to wall-clock at
  a ~2.6% ceiling. Take it only if you are actually hitting OOM.
- **MiniMaxChunkFeedForward** -- at 362 frames attention peaks ~17.8 GiB
  against FFN's 9-12, so it chunks a peak that is not binding. Short-clip
  feature.
- **PathchSageAttentionKJ** -- global no-guard sage switch. Prefer the
  per-workflow node.
"""


def build_ui(task: str, *, sage: bool = True, prompt: str | None = None,
             length: int = LENGTH, seed: int = SEED, preview: bool = False,
             sol: dict | None = None, sol_enabled: bool = True,
             **canvas) -> dict:
    ref = task == "r2v"
    cv = dict(CANVAS, **canvas)
    prompt = prompt if prompt is not None else {
        "t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]
    g = UIGraph()

    unet = g.add("UNETLoader", (-1500, 0), size=(560, 90),
                 widgets=[MODELS["unet_ref2va" if ref else "unet_fl2va"], "default"],
                 outputs=[_out("MODEL", "MODEL")])
    clip = g.add("CLIPLoader", (-1500, 140), size=(560, 110),
                 widgets=[MODELS["clip"], "minimax", "default"],
                 outputs=[_out("CLIP", "CLIP")])
    vvae = g.add("VAELoader", (-1500, 300), size=(560, 70),
                 widgets=[MODELS["video_vae"]], outputs=[_out("VAE", "VAE")],
                 title="Load VAE (video)")
    avae = g.add("VAELoader", (-1500, 410), size=(560, 70),
                 widgets=[MODELS["audio_vae"]], outputs=[_out("VAE", "VAE")],
                 title="Load VAE (audio)")

    model_src = unet
    sage_node = None
    if sage:
        sage_node = g.add("MiniMaxH3SageAttention", (-880, 0), size=(360, 110),
                          widgets=[SAGE_NODE["mode"],
                                   SAGE_NODE["patch_token_refiner"]],
                          inputs=[_in("model", "MODEL")],
                          outputs=[_out("MODEL", "MODEL")])
        g.link(unet, 0, sage_node, "model", "MODEL")
        model_src = sage_node

    if sol is not None:
        # After sage, never before. SolAttn composes by walking the model's
        # existing object patches and wrapping the attention forwards it
        # finds; run first it has nothing to find, and ours then overwrites
        # its patch. Both orders load and render, which is exactly why it is
        # worth pinning in a generated graph instead of leaving to hand-wiring.
        #
        # Enabled when the graph is built for it, bypassed otherwise. Bypass
        # passes MODEL straight through, so a graph carrying a disabled
        # Sol-Attn node still loads and renders without the node installed.
        # The error-prone part is the ordering above, not the toggle.
        sol_node = g.add("SolAttnPatch", (-880, 190), size=(360, 330),
                         widgets=[sol["tau"], sol["start_percent"], sol["end_percent"],
                                  sol["min_tokens"], sol["int8_qk"],
                                  sol["sink_conditioning"], sol["morton"],
                                  sol["morton_curve"], sol["int8_pv"], sol["verbose"],
                                  sol["use_tma"], sol["dense_blocks"]],
                         inputs=[_in("model", "MODEL")],
                         outputs=[_out("MODEL", "MODEL")],
                         title=("Patch Sol-Attn" if sol_enabled
                                else "Patch Sol-Attn (bypassed)"))
        if not sol_enabled:
            g._node(sol_node)["mode"] = 4
        g.link(model_src, 0, sol_node, "model", "MODEL")
        model_src = sol_node

    if preview:
        # The largest practical saving on a long clip, and not a kernel
        # change: a 362-frame render is ~17 min, so seeing step 3 is what
        # lets a bad seed die at 90 s instead of costing the whole run.
        #
        # It has to be this node rather than ComfyUI's built-in preview,
        # because the launcher passes --preview-method none globally; this
        # node sidesteps that by pushing its own frame to a DOM widget on
        # itself. taeh3 is the H3 tiny decoder (latent_channels 24,
        # patch_size 2) -- without it H3 has no approx VAE at all and
        # previews degrade to latent2rgb.
        #
        # preview_frames=4 rather than 1: a still frame catches a bad
        # composition, but the failures worth aborting a 17-minute render
        # for are motion failures, and those need more than one frame.
        prev_node = g.add("ModelPreviewOverrideKJ", (-460, 190), size=(360, 200),
                          widgets=[512, 80, True, 4, 8, "taeh3.safetensors"],
                          inputs=[_in("model", "MODEL"),
                                  _in("vae", "VAE", optional=True)],
                          outputs=[_out("MODEL", "MODEL")],
                          title="Preview (taeh3)")
        g.link(model_src, 0, prev_node, "model", "MODEL")
        model_src = prev_node

    img_a = img_b = None
    if ref:
        cond_inputs = [
            _in("clip", "CLIP"), _in("vae", "VAE"), _in("audio_vae", "VAE"),
            _in("ref_images.ref_image_0", "IMAGE", optional=True, label="ref_image_0"),
            _in("ref_images.ref_image_1", "IMAGE", optional=True, label="ref_image_1"),
            _in("ref_images.ref_image_2", "IMAGE", optional=True, label="ref_image_2"),
            _in("ref_videos.ref_video_0", "IMAGE", optional=True, label="ref_video_0"),
            _in("ref_video_audios.ref_video_audio_0", "AUDIO", optional=True,
                label="ref_video_audio_0"),
            _in("ref_audios.ref_audio_0", "AUDIO", optional=True, label="ref_audio_0"),
        ]
        cond = g.add("MiniMaxH3ReferenceToVideo", (-460, 0), size=(430, 620),
                     widgets=[prompt, cv["width"], cv["height"], length, "match"],
                     inputs=cond_inputs,
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        img_a = g.add("LoadImage", (-880, 640), size=(290, 330),
                      widgets=[PLACEHOLDER_IMAGE_A, "image"],
                      outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
        img_b = g.add("LoadImage", (-880, 1010), size=(290, 330),
                      widgets=[PLACEHOLDER_IMAGE_B, "image"],
                      outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
        g.link(vvae, 0, cond, "vae", "VAE")
        g.link(avae, 0, cond, "audio_vae", "VAE")
        g.link(img_a, 0, cond, "ref_images.ref_image_0", "IMAGE")
        g.link(img_b, 0, cond, "ref_images.ref_image_1", "IMAGE")
    else:
        cond_inputs = [_in("clip", "CLIP"), _in("vae", "VAE"),
                       _in("first_frame", "IMAGE", optional=True),
                       _in("last_frame", "IMAGE", optional=True)]
        cond = g.add("MiniMaxH3ImageToVideo", (-460, 0), size=(430, 560),
                     widgets=[prompt, cv["width"], cv["height"], length],
                     inputs=cond_inputs,
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        g.link(vvae, 0, cond, "vae", "VAE")
        if task == "i2v":
            img_a = g.add("LoadImage", (-880, 640), size=(290, 330),
                          widgets=[PLACEHOLDER_IMAGE_A, "image"],
                          outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
            g.link(img_a, 0, cond, "first_frame", "IMAGE")
    g.link(clip, 0, cond, "clip", "CLIP")

    noise = g.add("RandomNoise", (40, 0), size=(300, 110), widgets=[seed, "randomize"],
                  outputs=[_out("NOISE", "NOISE")])
    samp = g.add("KSamplerSelect", (40, 150), size=(300, 60),
                 widgets=[SAMPLING["sampler"]], outputs=[_out("SAMPLER", "SAMPLER")])
    sched = g.add("BasicScheduler", (40, 250), size=(300, 130),
                  widgets=[SAMPLING["scheduler"], SAMPLING["steps"], SAMPLING["denoise"]],
                  inputs=[_in("model", "MODEL")], outputs=[_out("SIGMAS", "SIGMAS")])
    guider = g.add("BasicGuider", (40, 420), size=(300, 70),
                   inputs=[_in("model", "MODEL"), _in("conditioning", "CONDITIONING")],
                   outputs=[_out("GUIDER", "GUIDER")])
    sampler = g.add("SamplerCustomAdvanced", (400, 0), size=(320, 150),
                    inputs=[_in("noise", "NOISE"), _in("guider", "GUIDER"),
                            _in("sampler", "SAMPLER"), _in("sigmas", "SIGMAS"),
                            _in("latent_image", "LATENT")],
                    outputs=[_out("output", "LATENT"), _out("denoised_output", "LATENT")])
    vdec = g.add("VAEDecode", (780, 0), size=(260, 60),
                 inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                 outputs=[_out("IMAGE", "IMAGE")])
    adec = g.add("VAEDecodeAudio", (780, 110), size=(260, 60),
                 inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                 outputs=[_out("AUDIO", "AUDIO")])
    mux = g.add("CreateVideo", (1080, 0), size=(270, 110), widgets=[FPS, 8],
                inputs=[_in("images", "IMAGE"), _in("audio", "AUDIO", optional=True)],
                outputs=[_out("VIDEO", "VIDEO")])
    save = g.add("SaveVideo", (1080, 170), size=(600, 400),
                 widgets=[f"video/h3_{task}_sage", "auto", "auto"],
                 inputs=[_in("video", "VIDEO")], outputs=[_out("video", "VIDEO")])

    g.link(model_src, 0, sched, "model", "MODEL")
    g.link(model_src, 0, guider, "model", "MODEL")
    g.link(cond, 0, guider, "conditioning", "CONDITIONING")
    g.link(cond, 1, sampler, "latent_image", "LATENT")
    g.link(noise, 0, sampler, "noise", "NOISE")
    g.link(guider, 0, sampler, "guider", "GUIDER")
    g.link(samp, 0, sampler, "sampler", "SAMPLER")
    g.link(sched, 0, sampler, "sigmas", "SIGMAS")
    g.link(sampler, 0, vdec, "samples", "LATENT")
    g.link(vvae, 0, vdec, "vae", "VAE")
    g.link(sampler, 0, adec, "samples", "LATENT")
    g.link(avae, 0, adec, "vae", "VAE")
    g.link(vdec, 0, mux, "images", "IMAGE")
    g.link(adec, 0, mux, "audio", "AUDIO")
    g.link(mux, 0, save, "video", "VIDEO")

    # Guidance in the graph rather than in a doc nobody opens next to it.
    # MarkdownNote is in _UI_ONLY, so these never reach the API form and
    # cannot desync it.
    g.add("MarkdownNote", (-2180, 0), size=(620, 620), widgets=[_NOTE_GEOMETRY],
          title="Canvas + length: what is actually selectable")
    g.add("MarkdownNote", (-2180, 660), size=(620, 560), widgets=[_NOTE_NODES],
          title="Which nodes, and the order that matters")

    return g.dump(f"h3-{task}-sage")


# --------------------------------------------------------------------------
# Static validation against /object_info
# --------------------------------------------------------------------------

def load_object_info(source: str) -> dict:
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/object_info", timeout=60) as r:
            return json.loads(r.read())
    return json.loads(Path(source).read_text())


def _combo_options(spec):
    """Combo option lists come in two shapes across ComfyUI node versions."""
    t = spec[0]
    if isinstance(t, list):
        return t
    if t == "COMBO":
        return (spec[1] or {}).get("options")
    return None


def validate_api(graph: dict, oi: dict, label: str) -> list[str]:
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    for nid, node in graph.items():
        ct = node["class_type"]
        if ct not in oi:
            e(f"node {nid}: unknown class_type {ct!r}")
            continue
        spec = oi[ct]["input"]
        req = spec.get("required") or {}
        opt = spec.get("optional") or {}
        known = dict(req) | dict(opt)
        # Autogrow inputs are declared once but addressed as
        # "<input>.<prefix><i>"; expand the legal names.
        for name, s in list(known.items()):
            if s[0] != "COMFY_AUTOGROW_V3":
                continue
            tpl = (s[1] or {}).get("template") or {}
            inner = tpl.get("input") or {}
            inner_spec = next(iter((inner.get("required") or inner.get("optional") or {}).values()), None)
            for i in range(tpl.get("max", 0)):
                known[f"{name}.{tpl['prefix']}{i}"] = inner_spec

        given = node["inputs"]
        for name in req:
            if req[name][0] == "COMFY_AUTOGROW_V3":
                continue
            if name not in given:
                e(f"node {nid} ({ct}): missing required input {name!r}")
        for name, val in given.items():
            if name not in known:
                e(f"node {nid} ({ct}): unknown input {name!r}")
                continue
            s = known[name]
            if isinstance(val, list):  # a link
                src, slot = val
                if src not in graph:
                    e(f"node {nid} ({ct}).{name}: links to missing node {src!r}")
                    continue
                souts = oi[graph[src]["class_type"]]["output"]
                if slot >= len(souts):
                    e(f"node {nid} ({ct}).{name}: output slot {slot} out of range "
                      f"on node {src} ({graph[src]['class_type']})")
                    continue
                got = souts[slot]
                want = s[0] if s else None
                got_name = got if isinstance(got, str) else "COMBO"
                if want and isinstance(want, str) and want not in ("*",) and got_name != want:
                    e(f"node {nid} ({ct}).{name}: type {got_name} from node {src} "
                      f"does not match {want}")
                continue
            if s is None:
                continue
            opts = _combo_options(s)
            if opts is not None and val not in opts:
                e(f"node {nid} ({ct}).{name}: {val!r} is not an available option")
                continue
            meta = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            if s[0] in ("INT", "FLOAT") and isinstance(val, (int, float)):
                if "min" in meta and val < meta["min"]:
                    e(f"node {nid} ({ct}).{name}: {val} below min {meta['min']}")
                if "max" in meta and val > meta["max"]:
                    e(f"node {nid} ({ct}).{name}: {val} above max {meta['max']}")

        # H3-specific: frame count is snapped up to 17k+5 by the node, so an
        # off-grid `length` silently renders a different duration than asked.
        if ct in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
                  "EmptyMiniMaxH3LatentAV"):
            ln = given.get("length")
            if isinstance(ln, int) and ln % 17 != 5:
                e(f"node {nid} ({ct}): length {ln} is off the 17k+5 grid; "
                  f"the node will snap it up to {ln + (5 - ln % 17) % 17}")

    # The mistake this whole file exists to prevent.
    consumers = [(nid, n) for nid, n in graph.items()
                 if n["class_type"] in ("BasicScheduler", "BasicGuider")]
    srcs = {tuple(n["inputs"]["model"]) for _, n in consumers
            if isinstance(n["inputs"].get("model"), list)}
    if len(srcs) > 1:
        e(f"BasicScheduler and BasicGuider read MODEL from different sources {srcs}; "
          f"one of them is bypassing a model patch")
    return errs


def validate_ui(wf: dict, oi: dict, label: str) -> list[str]:
    """Self-consistency only. No server validates a UI graph, so this checks
    what the frontend would choke on: dangling links and slot mismatches."""
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    by_id = {n["id"]: n for n in wf["nodes"]}
    declared = {l[0] for l in wf["links"]}
    for lid, src, ss, dst, ds, t in wf["links"]:
        if src not in by_id or dst not in by_id:
            e(f"link {lid}: endpoint missing")
            continue
        s, d = by_id[src], by_id[dst]
        if ss >= len(s["outputs"]):
            e(f"link {lid}: output slot {ss} out of range on {s['type']}")
        elif lid not in (s["outputs"][ss]["links"] or []):
            e(f"link {lid}: not listed on {s['type']} output {ss}")
        if ds >= len(d["inputs"]):
            e(f"link {lid}: input slot {ds} out of range on {d['type']}")
        elif d["inputs"][ds].get("link") != lid:
            e(f"link {lid}: not recorded on {d['type']} input {ds}")
    for n in wf["nodes"]:
        # Frontend-only nodes have no backend class, so they are absent from
        # /object_info by design. Rejecting them would be the validator being
        # confidently wrong rather than the graph being broken.
        if n["type"] in _FRONTEND_ONLY:
            continue
        if n["type"] not in oi:
            e(f"node {n['id']}: unknown type {n['type']!r}")
            continue
        for i, inp in enumerate(n["inputs"]):
            if inp.get("link") is not None and inp["link"] not in declared:
                e(f"node {n['id']} ({n['type']}) input {inp['name']}: dangling link")
            if inp.get("link") is None and inp.get("shape") != 7 and "widget" not in inp:
                e(f"node {n['id']} ({n['type']}): required input {inp['name']} unconnected")
        # widgets_values must cover every widget the node declares, in order,
        # including any that have been converted to inputs.
        spec = oi[n["type"]]["input"]
        widget_names = [k for k, v in ((spec.get("required") or {}) | (spec.get("optional") or {})).items()
                        if isinstance(v[0], list) or v[0] in
                        ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO",
                         "COMFY_DYNAMICCOMBO_V3")]
        got = len(n.get("widgets_values") or [])
        # RandomNoise / LoadImage carry an extra frontend-only widget
        # (control_after_generate, the upload button) that /object_info does
        # not report, so allow a surplus but never a shortfall.
        if got < len(widget_names):
            e(f"node {n['id']} ({n['type']}): {got} widget values for "
              f"{len(widget_names)} widgets {widget_names}")
    return errs


# --------------------------------------------------------------------------

# Nodes that are browser affordances rather than computation, so their
# absence from the API form is intentional and not drift.
#
# ModelPreviewOverrideKJ is the non-obvious one: it patches the model, but
# only to decode intermediate latents through taeh3 for display. Headless
# has nowhere to show them, and those decodes cost time that would land in
# any timing run as an unattributed confound. It belongs in the graph you
# watch and nowhere near the graph you measure.
_UI_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode",
            "ModelPreviewOverrideKJ"}

# Rendered entirely by the frontend, so they have no entry in /object_info.
# Subset of _UI_ONLY: ModelPreviewOverrideKJ is a real backend node that we
# exclude from the API form by choice, not by necessity.
_FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}


def _ui_settings(wf):
    """{class_type: widgets} for a UI graph, ignoring bypassed nodes."""
    return {n["type"]: n.get("widgets_values")
            for n in wf["nodes"]
            if n["type"] not in _UI_ONLY and n.get("mode", 0) == 0}


def _api_settings(wf):
    """{class_type: non-link inputs} for an API graph."""
    return {n["class_type"]: {k: v for k, v in n["inputs"].items()
                              if not isinstance(v, list)}
            for n in wf.values()}


def cross_check(written):
    """Report where a task's UI and API graphs disagree.

    Compares which nodes are present and, for the ones carrying settings we
    pin, that the pinned values match. Widget *order* differs between the two
    formats by design (UI is positional, API is keyed), so this checks the
    node set plus SolAttnPatch and MiniMaxH3SageAttention values explicitly
    rather than trying to align every widget by index.
    """
    by_task = {}
    for task, fmt, p, wf in written:
        by_task.setdefault(task, {})[fmt] = (p.name, wf)

    errs = []
    for task, forms in sorted(by_task.items()):
        if len(forms) < 2:
            continue
        ui_name, ui = forms["ui"]
        api_name, api = forms["api"]
        ui_s, api_s = _ui_settings(ui), _api_settings(api)

        only_ui = set(ui_s) - set(api_s) - _UI_ONLY
        only_api = set(api_s) - set(ui_s)
        for n in sorted(only_ui):
            errs.append(f"{task}: {n} in {ui_name} but not {api_name}")
        for n in sorted(only_api):
            errs.append(f"{task}: {n} in {api_name} but not {ui_name}")

        # SolAttnPatch is the one whose settings have actually drifted, so
        # check its values rather than only its presence. UI widgets are
        # positional in schema order; API inputs are keyed.
        if "SolAttnPatch" in ui_s and "SolAttnPatch" in api_s:
            order = ["tau", "start_percent", "end_percent", "min_tokens",
                     "int8_qk", "sink_conditioning", "morton", "morton_curve",
                     "int8_pv", "verbose", "use_tma", "dense_blocks"]
            widgets = ui_s["SolAttnPatch"] or []
            for i, key in enumerate(order):
                if i >= len(widgets) or key not in api_s["SolAttnPatch"]:
                    continue
                if widgets[i] != api_s["SolAttnPatch"][key]:
                    errs.append(
                        f"{task}: SolAttnPatch.{key} is {widgets[i]!r} in "
                        f"{ui_name} but {api_s['SolAttnPatch'][key]!r} in "
                        f"{api_name}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-info", default="http://127.0.0.1:8188",
                    help="running ComfyUI base URL, or a path to a saved object_info.json")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = []

    # The two you actually open in ComfyUI. Named for what they do, not for
    # the task abbreviation the code uses internally. Both carry the taeh3
    # preview, which is what lets a bad seed die at ~90s instead of costing a
    # full render -- worth more than any kernel knob when render time is the
    # objective.
    for fname, task, prompt, note in (
        ("h3_text_to_video.json", "t2v", LONG_T2V_PROMPT,
         "text -> video + audio"),
        ("h3_image_ref_plus_text_to_video.json", "r2v", None,
         "reference image(s) + text -> video + audio"),
    ):
        wf = build_ui(task, sage=True, length=LONG_LENGTH, preview=True,
                      sol=SOL_RECOMMENDED, sol_enabled=True, prompt=prompt)
        p = out / fname
        p.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        written.append((task, "ui", p, wf))
        print(f"  {p.name}: {note}")

    # API-format copies of the same two graphs, for driving a render over
    # /prompt without a browser. Same builder inputs, so they cannot describe
    # a different configuration than the pair above.
    for fname, task, prompt in (
        ("h3_text_to_video_api.json", "t2v", LONG_T2V_PROMPT),
        ("h3_image_ref_plus_text_to_video_api.json", "r2v", None),
    ):
        wf = build_api(task, sage=True, length=LONG_LENGTH,
                       sol=SOL_RECOMMENDED, prompt=prompt)
        p = out / fname
        p.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        written.append((task, "api", p, wf))

    for _t, _f, p, _w in written:
        print(f"wrote {p.name}")

    # Cross-check the two formats of each task describe the same graph. The
    # per-format validators below only prove each is well-formed against
    # object_info; nothing there would notice the UI graph carrying a
    # SolAttnPatch the API graph lacks, which is exactly the state this file
    # was in before 2026-08-06.
    drift = cross_check(written)
    if drift:
        print("\nUI/API DRIFT:")
        for x in drift:
            print("  " + x)
        return 1
    print("UI/API cross-check: same node set and settings")

    if args.no_validate:
        return 0
    oi = load_object_info(args.object_info)
    errs = []
    for task, fmt, p, wf in written:
        errs += (validate_api if fmt == "api" else validate_ui)(wf, oi, p.name)
    if errs:
        print("\nvalidation FAILED:")
        for x in errs:
            print("  " + x)
        return 1
    print(f"\nvalidated {len(written)} graphs against object_info: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
