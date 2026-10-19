# Openjutsu

**Open-source alternative to Higgsfield's Genjutsu Motion Transfer — local, on-device, open weights.**

Openjutsu takes a source video plus 1–9 reference images and produces a new
video that keeps the source's motion, camera, timing, and audio while
re-casting the character, location, and style from the references.

The reference implementation runs entirely on a single RTX 5080 (16 GB VRAM)
using **MiniMax H3 + H3 Fun ControlNet-Union** as the core model. The
submission is wrapped in the **Comfy Developer Platform** (`comfy-sdk` +
`comfy-api-proxy` + a local ComfyUI daemon) to satisfy the
[Open Source Higgsfield](https://comfy.org/open-source-higgsfield/) contest
rules.

---

## What it does

| Stage | What happens |
|---|---|
| **Input** | Source clip (4–30 s, 24 fps) + 1–9 reference images + optional prompt |
| **Preprocess** | Canny edge map of the source (downscaled to fit VRAM, upscaled by the controlnet) |
| **Generate** | H3 ref2va conditioned on the edge map, reference images, and prompt |
| **Audio** | Source audio is VAE-encoded and locked into the AV latent with a zero-denoise mask, then muxed into the output untouched |
| **Output** | 864×480 / 24 fps / 5–15 s MP4 (H.264 + AAC) |

The controlnet preserves motion. The reference images preserve identity.
The prompt gives you stylistic control. The `VRGDG_MiniMaxH3AudioDrive`
node preserves the source clip's audio.

> **Note on audio + `torchaudio`**: `VRGDG_MiniMaxH3AudioDrive` originally
> used `torchaudio.functional.resample`. No `torchaudio` cu130 wheel is
> compatible with torch 2.14.1, so the node as shipped can't load on
> the build target (RTX 5080 / 16 GB). This repo ships a one-function
> patch in `comfyui-vrgamedevgirl/VRGDG_MiniMaxH3AudioDrive.py` that
> replaces the `torchaudio` call with a `scipy.signal.resample_poly`
> equivalent. The patch is clearly commented, so the next VRGDG release
> shows the diff and you can either keep it or revert to upstream.

---

## Demo

The bundled example uses ComfyUI's official `dancer_field_pose.mp4` (5.17 s,
864×480, 24 fps) as the source, and the first frame of that clip as the
single reference image.

- **Source:** `examples/dancer_field_pose.mp4`
- **Reference:** `examples/dancer_ref.png` (first frame of source)
- **Output:** `examples/output/openjutsu_demo_v1.mp4`

The output keeps the meadow setting, golden-hour lighting, and rust dress
from the reference — H3 generates the in-between frames from the
Canny-conditioned latent.

---

## Architecture

```
                       comfy-sdk (Python client)
                                │
                                ▼
            comfy-api-proxy  (port 8189, Comfy API v2 surface)
                                │
                                ▼
           local ComfyUI daemon  (port 8188, server mode)
                                │
   ┌─────────────────┬──────────────────┬──────────────────┐
   │                 │                  │                  │
   ▼                 ▼                  ▼                  ▼
UNETLoader       CLIPLoader          VAELoader         ModelPatchLoader
(H3 ref2va       (Qwen3-VL-32B       (H3 video VAE     (H3 Fun ControlNet
int8, 21 GB)     NVFP4 AWQ)         int8, 2.6 GB)     Union, 2.3 GB)

                                │
                                ▼
                   MiniMaxH3ReferenceToVideo
       (prompt + 9 ref images + 3 ref videos + 3 ref audios)

                                │
                                ▼
              MiniMaxH3FunControlNetApply
   (H3 model + controlnet model_patch + Canny edges from source)

                                │
                                ▼
                  KSampler → VAEDecode → CreateVideo → SaveVideo
                      │
                      └─ audio: H3-generated from prompt (or source-preserved
                         if you wire in VRGDG_MiniMaxH3AudioDrive — see note above)
```

Memory note: H3 alone needs ~13.5 GB VRAM; the controlnet adds ~2 GB; the
int8 VAE is 2.6 GB; Qwen3 NVFP4 AWQ is ~6 GB. On a 16 GB card, the daemon
runs with `--lowvram --reserve-vram 0.5 --vram-headroom 1` and the Canny
preprocessor downsamples the source to 432×240 before re-injecting it (the
controlnet then upsamples to the target resolution internally).

---

## Quick start

### 1. Models

The build needs these on disk (all open-weights):

| File | Size | Source |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) → `diffusion_models/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | 2.6 GB | same repo → `vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | same repo → `vae/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB | same repo → `text_encoders/` |
| `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` | 2.3 GB | same repo → `model_patches/` |

### 2. ComfyUI

Install ComfyUI 0.39.0+ with the bundled `MiniMaxH3ReferenceToVideo` and
`MiniMaxH3FunControlNetApply` nodes (they ship in `comfy_extras/nodes_minimax_h3.py`).
Drop `comfyui-vrgamedevgirl` into `custom_nodes/` for `VRGDG_MiniMaxH3ReferenceMediaFromPaths`
and `VRGDG_MiniMaxH3AudioDrive`.

### 3. Comfy SDK + API Proxy

On the host that submits jobs:

```bash
pip install "comfy-sdk>=0.3.0"
pip install comfy-api-proxy
```

### 4. Run the daemon

```bash
cd /path/to/ComfyUI
PYTHONPATH="" .venv/bin/python main.py \
    --listen 0.0.0.0 --port 8188 \
    --fp8_e4m3fn-unet --fp16-vae --enable-cors-header '*' \
    --lowvram --reserve-vram 0.5 --vram-headroom 1 \
    --log-stdout
```

### 5. Run the proxy

```bash
PYTHONPATH="" comfy-api-proxy run \
    --comfyui http://127.0.0.1:8188 \
    --host 127.0.0.1 --port 8189 \
    --comfyui-base-dir /path/to/ComfyUI \
    --allow-insecure-bind
```

### 6. Submit

Edit `scripts/render.py` to point at your workflow file and asset, then:

```bash
PYTHONPATH="" python scripts/render.py
```

`COMFY_BASE_URL` defaults to `http://127.0.0.1:8189`. The script will
upload the video, submit the workflow, wait for completion, and download
the output to `examples/output/`.

---

## Files in this repo

```
openjutsu/
├── LICENSE                  # MIT
├── README.md                # this file
├── requirements.txt         # host-side pip extras
├── .gitignore
├── workflows/
│   └── openjutsu_api.json   # the graph, in Comfy API format
├── scripts/
│   └── render.py            # comfy-sdk submission entry point
├── examples/
│   ├── dancer_field_pose.mp4   # source video (from ComfyUI's official examples)
│   ├── dancer_ref.png          # reference image (first frame of source)
│   │   ├── openjucer_v1.mp4            # H3-generated audio (kept for comparison)
│   │   └── openjucer_v2.mp4            # source-audio preserved
├── docs/
│   └── architecture.md       # deeper notes on node choices + memory budget
├── web/
│   ├── server.py             # FastAPI web UI (Genjutsu-style form)
│   └── static/
│       ├── index.html        # source video, refs, prompt, quality
│       ├── style.css         # dark theme, lime accent
│       └── app.js            # upload + poll + render
```

---

## Local web UI

The repo ships a FastAPI app that exposes the same workflow as a web
form (Genjutsu-style: drop a source video, drop reference images, write
a prompt, hit Generate). The demo video and audio preservation flow
are both reachable from the UI.

```bash
# Terminal 1: ComfyUI daemon
PYTHONPATH="" /home/marc/nvme-data/.comfyui/.venv/bin/python \
    /home/marc/nvme-data/.comfyui/main.py --listen 0.0.0.0 --port 8188 \
    --fp8_e4m3fn-unet --fp16-vae --enable-cors-header '*' \
    --lowvram --reserve-vram 0.5 --vram-headroom 1 --log-stdout

# Terminal 2: Comfy API proxy
PYTHONPATH="" /home/marc/nvme-data/.comfyui/.venv/bin/comfy-api-proxy run \
    --comfyui http://127.0.0.1:8188 --host 127.0.0.1 --port 8189 \
    --comfyui-base-dir /home/marc/nvme-data/.comfyui --allow-insecure-bind

# Terminal 3: Openjutsu web UI
PYTHONPATH="" /home/marc/nvme-data/.comfyui/.venv/bin/python \
    /home/marc/repos/openjutsu/web/server.py --host 0.0.0.0 --port 8765

# Browser
open http://localhost:8765
```

Uploads land in ComfyUI's input directory; the workflow is submitted
through the Comfy SDK against `comfy-api-proxy`; outputs are streamed
back to the browser via the `/outputs/{filename}` mount.

---

## Contest compliance

| Rule | How Openjutsu satisfies it |
|---|---|
| Built on Comfy Developer Platform | `comfy-sdk` calls `comfy-api-proxy`, which fronts a self-hosted ComfyUI daemon |
| Open-source license + setup detail | MIT, this README walks through models, deps, flags, and submission |
| Other tools/models must be explained in demo video | See `docs/architecture.md` |
| Lawful, SFW, no unlicensed IP | Source clip is ComfyUI's public example asset; no third-party likeness or trademarked IP used |
| Promotional rights to sponsors | Standard contest submission |
| One submission per person | Single repo + single entry |

---

## Credits

- **MiniMax** for the [H3 ref2va model](https://huggingface.co/Comfy-Org/MiniMax-H3) and the [H3 Fun ControlNet-Union](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors)
- **ComfyUI / Comfy-Org** for the Comfy Developer Platform, `comfy-sdk`, `comfy-api-proxy`, and the H3 native nodes
- **VRGDG (comfyui-vrgamedevgirl)** for the H3 audio-drive and reference-media nodes
- **Kornia** for the Canny edge detector used inside ComfyUI core
- **Higgsfield** for the original Genjutsu Motion Transfer — this is the open-source alternative