# Openjutsu

**Open-source alternative to Higgsfield's Genjutsu Motion Transfer — local, on-device, open weights.**

Openjutsu takes a source video plus reference images and produces a new video
that keeps the source's motion, camera, timing and audio while re-casting the
character, location and style from the references.

The reference implementation runs entirely on a single RTX 5080 (16 GB VRAM)
using **MiniMax H3 + H3 Fun ControlNet-Union** as the core model, submitted
through the **Comfy Developer Platform** (`comfy-sdk` + `comfy-api-proxy` +
a local ComfyUI daemon) to satisfy the
[Open Source Higgsfield](https://comfy.org/open-source-higgsfield/) contest rules.

---

## Try it

The app is running publicly for the contest:

**<https://ai5080.taila6e2e.ts.net:8443>**

`/api/*` is behind a shared-secret token (`OPENJUTSU_TOKEN`), because
`/api/generate` renders video on the GPU and the app has no accounts. If you
get a `401`, ask for the token, or run it yourself — it is four commands
below.

Watch the results without rendering anything:

| Clip | |
|---|---|
| [`/examples/openjutsu_demo_reel.mp4`](examples/openjutsu_demo_reel.mp4) | 90 s demo reel (both modes, how it is built) |
| [`/examples/openjutsu_demo.mp4`](examples/openjutsu_demo.mp4) | Motion Transfer result |
| [`/examples/openjutsu_swap_demo.mp4`](examples/openjutsu_swap_demo.mp4) | Object Swap result — the dancer is replaced by the reference character, the meadow is untouched |

### Object Swap, honestly

H3's controlnet treats its `mask` input as a soft hint, not a region lock, so
the whole frame drifts. `web/composite.py` composites the render over the
source afterwards: the measured difference outside the mask drops to **2.4
mean absolute** — which is H.264 re-encode loss, since re-encoding the source
through the same encoder gives 2.4. Selectivity (in-mask vs out-of-mask
divergence) goes from **1.3x to 14x**, measured on the render below.

Two things about the mask, both measured rather than assumed:

- **A still mask is the reliable path.** The one shipped in `examples/` is a
  blob covering the subject. It is what the demo clip uses and it holds for
  the whole clip.
- **A tracked mask video drifts.** `scripts/make_mask.py` can build a
  per-frame mask by subtracting a temporal-median background, and it looks
  right, but its alignment with the source wanders by up to ±15 frames
  (measured by mask/dancer IoU, which stays between 0.08 and 0.28). The
  swap then lands on the wrong part of the shot in the later frames. It is
  wired up and reachable over the API (`mask_video=`), but it is not the
  default, and the web UI only accepts a still image.

The reference binding matters as much as the mask: node 136's prompt is a
*link* to node 772, whose value ends with "The face is the face of the person
in `<Picture 1>`". Replacing that link with a plain string drops the binding
and the model regenerates the original subject — the swap appears to succeed
while nothing has been swapped. `web/server.py` merges your text into node
772 and leaves the link intact.

### Running it on your own machine

Two limits are worth knowing before you start:

- **One render at a time.** H3 needs ~13.5 GB of VRAM; a second concurrent
  render on a 16 GB card gets the daemon OOM-killed. The app refuses a
  second render with HTTP 429 rather than fall over.
- **Give the ComfyUI daemon `--async-offload`.** The H3 UNet (21 GB) and the
  Qwen3-VL encoder (15 GB) will otherwise be mmapped into RAM, which on a
  32 GB machine with a full swap gets the daemon killed mid-render. The
  systemd unit in `deploy/systemd/` sets it.

---

## What it does

| Stage | What happens |
|---|---|
| **Input** | Source clip (4–30 s, 24 fps) + up to 9 reference images + optional prompt |
| **Preprocess** | Canny edge map of the source (downscaled to fit VRAM, upscaled by the controlnet) |
| **Generate** | H3 ref2va conditioned on the edge map, reference images and prompt |
| **Audio** | The source clip's audio track is preserved in the output (see [Audio](#audio)) |
| **Output** | 864×480 / 24 fps / 5–15 s MP4 (H.264 + AAC) |

The controlnet preserves motion. The reference images preserve identity. The
prompt gives you stylistic control.

Reference images have **roles** — *person*, *scene*, *outfit* — so you can say
which image is the character and which is the location. Roles you leave empty
are dropped, so H3 only ever sees the references you actually supplied.

---

## Quick start

### 1. Models

All open weights, from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| File | Size | Target dir |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21 GB | `models/diffusion_models/` |
| `minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` | 2.3 GB | `models/model_patches/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB | `models/text_encoders/` |
| `minimax_h3_video_vae_int8_convrot.safetensors` | 2.6 GB | `models/vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | 605 MB | `models/vae/` |

Optional — turbo mode, 4 steps instead of 20 (roughly 5× faster, slightly lower
audio and motion quality):

| File | Size | Target dir |
|---|---|---|
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.9 GB | `models/loras/` |

### 2. Prepare the workspace

```bash
git clone https://github.com/Ex8-ca/openjutsu.git
cd openjutsu
./scripts/setup.sh
```

`setup.sh` is idempotent. It copies the bundled example assets into ComfyUI's
`input/` directory, installs the optional `comfyui-vrgamedevgirl` custom node
pack and applies the `torchaudio` patch described under [Audio](#audio), then
prints the model checklist.

### 3. ComfyUI

Install ComfyUI 0.39.0 or newer. The `MiniMaxH3ReferenceToVideo` and
`MiniMaxH3FunControlNetApply` nodes ship in `comfy_extras/nodes_minimax_h3.py`
(built in — no custom node pack required for the core workflow).

### 4. Start the daemon and the proxy

```bash
cd /path/to/ComfyUI
PYTHONPATH="" .venv/bin/python main.py \
    --listen 0.0.0.0 --port 8188 \
    --fp8_e4m3fn-unet --fp16-vae --enable-cors-header '*' \
    --lowvram --reserve-vram 0.5 --vram-headroom 1 \
    --log-stdout
```

In a second shell:

```bash
PYTHONPATH="" comfy-api-proxy run \
    --comfyui http://127.0.0.1:8188 \
    --host 127.0.0.1 --port 8189 \
    --comfyui-base-dir /path/to/ComfyUI \
    --allow-insecure-bind
```

### 5. Start the web UI

```bash
pip install -r requirements.txt
python web/server.py --host 0.0.0.0 --port 8765
```

Then open <http://localhost:8765>: drop a source video, fill in the *person* /
*scene* / *outfit* reference slots, optionally write a prompt, and hit
**Generate**. The result comes back with a triptych — source frame, reference
and output — so identity and motion preservation are easy to check at a glance.

### 6. Or run it headless

```bash
python scripts/render.py
```

`render.py` submits `workflows/openjutsu_api.json` through the Comfy SDK, waits
for completion and writes the result to `examples/output/`.

### Configuration

Every path is environment-driven, so the app runs on any machine:

| Variable | Default | Purpose |
|---|---|---|
| `COMFY_BASE_URL` | `http://127.0.0.1:8189` | Comfy API v2 base URL (the proxy) |
| `COMFYUI_ROOT` | `/home/marc/nvme-data/.comfyui` | ComfyUI install root |
| `COMFYUI_INPUT_DIR` | `$COMFYUI_ROOT/input` | Where uploads land |
| `COMFYUI_VENV_SITE` | `$COMFYUI_ROOT/.venv/lib/python*/site-packages` | Where `comfy_sdk` lives |
| `COMFYUI_BASE_URL` | `http://127.0.0.1:8188` | Used only to probe which node types exist |
| `OPENJUTSU_OUTPUT_DIR` | `<repo>/web/outputs` | Where rendered outputs are written |
| `OPENJUTSU_WORKFLOW` | `auto` | Force `core` or `audio` instead of auto-detecting |
| `OPENJUTSU_TOKEN` | *(unset)* | Shared secret for the `/api/*` endpoints. Unset = open, which is fine on localhost. Set it before exposing the app to the internet, because `/api/generate` renders on the GPU. |

---

## Audio

Genjutsu keeps the source clip's audio. So does Openjutsu.

The `VRGDG_MiniMaxH3AudioDrive` node from the `comfyui-vrgamedevgirl` pack
encodes the source audio into H3's audio VAE, drops it into the joint AV latent
and marks it with a zero-denoise mask, so the sampler only generates the video
half. The original track is muxed into the final MP4 untouched.

**Verified:** output RMS matches the source within 0.2 dB across channels.

That node calls `torchaudio.functional.resample`, and there is no `torchaudio`
cu130 wheel that is ABI-compatible with torch 2.14.1 — the node fails to import
on the build target. This repo ships
`patches/vrgdg-minimax-h3-audio-drive-torchaudio.patch`, which replaces that one
call with a numerically equivalent `scipy.signal.resample_poly` (scipy is
already a ComfyUI dependency). `setup.sh` applies it for you. The patch is
clearly commented, so the next upstream release shows the diff and you can
either keep it or revert it.

**If you skip the patch**, the app still runs — it falls back to H3's own
generated audio. The web UI picks the audio-preserving workflow automatically
when the node is present, and the core workflow otherwise. Force either with
`OPENJUTSU_WORKFLOW=core|audio`.

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
   ┌─────────────────┬──────────────────┬──────────────────┐
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
                      └─ audio: source track preserved
                         (or H3-generated, see "Audio" above)
```

Memory note: H3 alone needs ~13.5 GB VRAM; the controlnet adds ~2 GB; the
int8 VAE is 2.6 GB; Qwen3 NVFP4 AWQ is ~6 GB. On a 16 GB card, the daemon runs
with `--lowvram --reserve-vram 0.5 --vram-headroom 1` and the Canny preprocessor
downsamples the source to 432×240 before re-injecting it (the controlnet then
upsamples to the target resolution internally).

---

## Files in this repo

```
openjutsu/
├── LICENSE                  # MIT
├── README.md                # this file
├── requirements.txt         # host-side pip deps
├── examples/
│   ├── dancer_field_pose.mp4   # source clip (ComfyUI's official example asset)
│   ├── dancer_ref.png          # reference image (first frame of the source)
│   ├── source_frame_832.png    # scene anchor (first frame, 832 px wide)
│   └── openjutsu_demo.mp4      # a rendered result, with preserved source audio
├── docs/
│   └── architecture.md       # deeper notes on node choices + memory budget
├── patches/
│   └── vrgdg-minimax-h3-audio-drive-torchaudio.patch
├── scripts/
│   ├── setup.sh              # one-shot workspace prep
│   └── render.py             # headless submission via the Comfy SDK
├── web/
│   ├── server.py             # FastAPI web UI
│   └── static/
│       ├── index.html        # source video, role slots, prompt, quality
│       ├── style.css         # dark theme, lime accent
│       └── app.js            # upload + poll + render
├── workflows/
│   ├── openjutsu_api.json        # core graph
│   └── openjutsu_api_audio.json  # source-audio-preserving graph
└── deploy/
    └── systemd/
        ├── comfyui-daemon.service  # ComfyUI on :8188 (Restart=always)
        ├── comfyui-proxy.service    # comfy-api-proxy on :8189
        └── openjutsu-web.service    # FastAPI UI on :8765
```

---

## Running it as services

The `deploy/systemd/` directory has three user-level unit files that manage the
daemon, proxy and UI. They bind to `0.0.0.0` so any LAN device (including a
Tailscale MagicDNS hostname) can reach them.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now comfyui-daemon.service comfyui-proxy.service openjutsu-web.service

# Check
systemctl --user status comfyui-daemon.service comfyui-proxy.service openjutsu-web.service
curl -s http://127.0.0.1:8765/healthz   # → {"status":"ok",...}
```

All three services have `Restart=always` and `RestartSec=3`, so any crash
(OOM, manual `pkill`, host reboot) is recovered automatically.

The units use `%h` and set `COMFYUI_ROOT`, `COMFYUI_INPUT_DIR`,
`COMFYUI_VENV_SITE` and `OPENJUTSU_OUTPUT_DIR` in the unit's `Environment=`
block, so the same files work on any machine. Edit those paths if your ComfyUI
lives somewhere else.

> **Tailscale / browser bridge note**: the UI's `fetch()` calls the API on
> whatever origin the page was loaded from, so the page must be loaded from a
> host that can reach `:8765`. CORS is configured `allow_origins=["*"]` so
> cross-origin requests are accepted regardless. From a Tailscale node,
> `http://<machine-name>.<tailnet>:8765` works.

Uploads land in ComfyUI's input directory; the workflow is submitted through
the Comfy SDK against `comfy-api-proxy`; outputs are streamed back to the
browser via the `/outputs/{filename}` mount.

---

## Contest compliance

| Rule | How Openjutsu satisfies it |
|---|---|
| Built on the Comfy Developer Platform | `comfy-sdk` calls `comfy-api-proxy` (both Comfy Org packages), which fronts a self-hosted ComfyUI daemon over the Comfy API v2 surface |
| Open-source license + enough setup detail | MIT; this README plus `scripts/setup.sh` walk through models, deps, flags and launch |
| Other tools/models explained in the demo video | See `docs/architecture.md`; the VRGDG audio node and the `torchaudio` patch are called out under [Audio](#audio) |
| Lawful, SFW, no unlicensed IP or likenesses | The bundled example clip and reference frames are ComfyUI's own public example assets |
| Promotional rights to sponsors | Standard contest submission |
| One submission per person | Single repo + single entry |

**Open Weights Only category** — no partner nodes and no closed-source APIs are
used. Every model is open weights, the only third-party code is the optional
`comfyui-vrgamedevgirl` ComfyUI custom node pack (AGPL-3.0, used unmodified
apart from the `torchaudio` compatibility patch above), and the app runs
entirely on your own hardware.

**Built with MiniMax H3** — H3 is the core model of the workflow, as the
contest's MiniMax H3 bonus track requires.

---

## Credits

- **MiniMax** for the [H3 ref2va model](https://huggingface.co/Comfy-Org/MiniMax-H3) and the H3 Fun ControlNet-Union
- **ComfyUI / Comfy-Org** for the Comfy Developer Platform, `comfy-sdk`, `comfy-api-proxy`, and the H3 native nodes
- **VRGDG (comfyui-vrgamedevgirl)** for the H3 audio-drive node
- **Kornia** for the Canny edge detector used inside ComfyUI core
- **Higgsfield** for the original Genjutsu Motion Transfer — this is the open-source alternative
