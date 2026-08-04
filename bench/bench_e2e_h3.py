#!/usr/bin/env python3
"""End-to-end A/B: a real MiniMax H3 render with the sage node in and out.

The per-module bench (`bench_minimax_attn.py`) measures one Attention in
isolation. This one submits an actual render to a running ComfyUI and
reports what the user experiences, which is the only number that settles
whether the kernel win survives contact with a 21 GB checkpoint on a
24 GB card.

Two things are measured per run:

  - **sampler time**, from ComfyUI's websocket node-transition events.
    This isolates `SamplerCustomAdvanced` from text encoding and VAE
    decode, so it is where an attention speedup has to show up.
  - **total wall-time**, from submit to history. This is what actually
    changes for the user, and it will always show a smaller ratio than
    sampler time because encode and decode are unaffected.

Method notes that matter for trusting the result:

  - The first run is a warmup and is discarded. It pays model load,
    Triton autotune for every new shape, and the Qwen3-VL-32B text
    encode. Including it would swamp everything else.
  - Arms alternate (A B A B ...) rather than running in blocks, so any
    drift in clocks, thermals or allocator state is shared rather than
    attributed to whichever arm ran second.
  - The graph is built here rather than converted from the bundled UI
    templates: two of those hide the sampler stack inside a subgraph,
    and hand-converting a subgraph to API format is a good way to
    measure something subtly different from what the template runs.
    Settings are copied from `video_minimax_h3_i2v.json`.

    ./bench/bench_e2e_h3.py --runs 3
    ./bench/bench_e2e_h3.py --runs 3 --width 768 --height 768 --steps 10

Needs a running ComfyUI with the MiniMax H3 models installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
import uuid

# Settings lifted from the bundled i2v template so this measures the
# configuration people actually run.
DEFAULTS = dict(
    unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    video_vae="minimax_h3_video_vae_fp16.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
    sampler="res_multistep",
    scheduler="simple",
    steps=20,
    width=1344,
    height=768,
    length=73,
    fps=24.0,
)

# A prompt shaped the way MiniMax's own writing guide recommends:
# style and composition first, then subject, scene, camera motion,
# action, then soundscape.
PROMPT = (
    "Live-action, cinematic, shallow depth of field. A medium-wide shot "
    "frames a lone lighthouse keeper on a wet stone balcony at dawn, "
    "wearing a heavy oilskin coat, the lamp housing glowing behind them. "
    "Grey-blue sea fog rolls past below, gulls crossing the frame. "
    "The camera pushes in slowly with small amplitude as the keeper "
    "raises a brass telescope, holds it steady, then lowers it and turns "
    "toward the light.\n\n"
    "Audio: low sea swell and wind against stone, a distant foghorn twice, "
    "gulls calling overhead.\n\n"
    "No dialogue, no text overlays, no cuts."
)


def build_prompt(cfg, *, sage, seed):
    """API-format graph. `sage` inserts the node between UNETLoader and
    the two MODEL consumers; otherwise they read the loader directly."""
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": cfg["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": cfg["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": PROMPT,
                         "width": cfg["width"], "height": cfg["height"],
                         "length": cfg["length"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": cfg["sampler"]}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None, "scheduler": cfg["scheduler"],
                         "steps": cfg["steps"], "denoise": 1.0}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "fps": cfg["fps"], "audio": ["12", 0]}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0], "filename_prefix": "video/h3_sage_ab",
                          "format": "auto", "codec": "auto"}},
    }
    if sage:
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": ["1", 0], "mode": "auto",
                              "patch_token_refiner": False}}
        model_src = ["20", 0]
    else:
        model_src = ["1", 0]
    g["8"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["model"] = model_src
    return g


def http_post(url, obj, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


async def run_once(host, prompt, client_id, timeout_s):
    """Submit and follow the websocket. Returns (total_s, per_node_s, error)."""
    import aiohttp

    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(
            f"ws://{host}/ws?clientId={client_id}", heartbeat=30
        ) as ws:
            t_submit = time.perf_counter()
            resp = http_post(f"http://{host}/prompt",
                             {"prompt": prompt, "client_id": client_id})
            prompt_id = resp["prompt_id"]

            per_node, current, t_node = {}, None, None
            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None, per_node, f"timed out after {timeout_s:.0f}s"
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None, per_node, f"timed out after {timeout_s:.0f}s"
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                mtype, d = data.get("type"), data.get("data", {})
                if d.get("prompt_id") not in (None, prompt_id):
                    continue

                if mtype == "executing":
                    now = time.perf_counter()
                    if current is not None and t_node is not None:
                        per_node[current] = per_node.get(current, 0.0) + (now - t_node)
                    node = d.get("node")
                    if node is None:                      # run finished
                        return now - t_submit, per_node, None
                    current, t_node = node, now
                elif mtype == "execution_error":
                    return None, per_node, d.get("exception_message", "execution error")
                elif mtype == "execution_interrupted":
                    return None, per_node, "interrupted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per arm")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--width", type=int, default=DEFAULTS["width"])
    ap.add_argument("--height", type=int, default=DEFAULTS["height"])
    ap.add_argument("--length", type=int, default=DEFAULTS["length"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Only if a comparable render already ran this session. "
                         "A cold first run pays model load and Triton autotune and "
                         "will read as a large fake win for whichever arm is second.")
    args = ap.parse_args()

    cfg = dict(DEFAULTS, steps=args.steps, width=args.width,
               height=args.height, length=args.length)
    client_id = str(uuid.uuid4())
    SAMPLER_NODE = "10"

    print(f"MiniMax H3 e2e A/B  {cfg['width']}x{cfg['height']} "
          f"length={cfg['length']} steps={cfg['steps']} seed={args.seed}")
    print(f"host={args.host}  runs={args.runs} per arm  (same seed both arms)\n")

    # ComfyUI caches node outputs, so re-submitting an identical graph
    # executes nothing and returns in milliseconds -- which reads as an
    # enormous fake speedup. Each iteration therefore gets its own seed,
    # shared by both arms so the A/B stays paired, and the warmup gets a
    # seed of its own so it cannot alias the first timed run.
    def seed_for(i):
        return args.seed + i

    if not args.skip_warmup:
        print("warmup (discarded: model load + Triton autotune + text encode) ...", flush=True)
        total, _, err = asyncio.run(run_once(
            args.host, build_prompt(cfg, sage=True, seed=seed_for(0)), client_id, args.timeout))
        if err:
            print(f"  warmup FAILED: {err}")
            return 1
        print(f"  {total:.1f}s\n")

    results = {"sage": [], "off": []}
    sampler = {"sage": [], "off": []}
    for i in range(args.runs):
        seed = seed_for(i + 1)
        for arm, use_sage in (("sage", True), ("off", False)):
            total, per_node, err = asyncio.run(run_once(
                args.host, build_prompt(cfg, sage=use_sage, seed=seed),
                client_id, args.timeout))
            if err:
                print(f"  run {i+1} {arm}: FAILED: {err}")
                return 1
            s = per_node.get(SAMPLER_NODE)
            if s is None:
                print(f"  run {i+1} {arm}: sampler node never executed -- ComfyUI "
                      f"served this graph from cache, so there is no timing to "
                      f"report. Vary the seed or restart ComfyUI.")
                return 1
            results[arm].append(total)
            sampler[arm].append(s)
            print(f"  run {i+1} {arm:4s}  seed={seed}  total {total:7.1f}s   "
                  f"sampler {s:7.1f}s", flush=True)

    print()
    def med(xs):
        return statistics.median(xs)
    s_on, s_off = med(sampler["sage"]), med(sampler["off"])
    t_on, t_off = med(results["sage"]), med(results["off"])
    print(f"{'':10s} {'sampler':>12s} {'total':>12s}")
    print(f"{'sage off':10s} {s_off:11.1f}s {t_off:11.1f}s")
    print(f"{'sage on':10s} {s_on:11.1f}s {t_on:11.1f}s")
    print(f"{'speedup':10s} {s_off/s_on:11.2f}x {t_off/t_on:11.2f}x")
    print(f"\nsampler share of total, sage off: {100*s_off/t_off:.0f}%  "
          f"-- the ceiling on what any attention work can move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
