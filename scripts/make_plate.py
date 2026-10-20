#!/usr/bin/env python3
"""Build a scene-consistent reference plate for Object Swap.

The problem
-----------
Feeding H3 a reference of a *different* character in a *different* scene makes
it regenerate the whole frame and relocate the subject: the controlnet pulls
the output toward the source, the reference pulls it toward the plate, and the
model resolves the conflict by rebuilding the scene. The swap then lands in
the wrong place, and a region-locked composite has nothing to grab.

The fix
-------
Build the reference plate **from the source frame**, so the plate already
agrees with the source on scene, lighting, framing and pose, and differs
only in the character.

Two ways to build it, both tried here:

1. Plain img2img (source frame as init image, character as the prompt). The
   scene survives, but the *pose* drifts - the model re-stages the subject.
2. **Masked inpaint** (source frame as init image, the subject's own mask
   zeroed in the latent, character as the prompt). The unmasked pixels are
   preserved exactly and the masked region is regenerated, so the new
   character inherits the subject's position AND pose. This is the one that
   works.

Then the swap can run at a normal controlnet strength, because the plate and
the source no longer contradict each other.

Usage
-----
    python scripts/make_plate.py <source_video> <out.png> [mask_video] [prompt]

    # mask_video: the tracked mask from scripts/make_mask.py; without it the
    # whole frame is re-staged and the pose drifts.

Runs on the local ComfyUI daemon over its HTTP API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = (
    "a samurai in ornate crimson lacquered armour with a horned kabuto helmet, "
    "holding a katana, photorealistic, sharp focus, golden hour"
)


def _base() -> str:
    return os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")


def _input_dir() -> Path:
    root = os.environ.get("COMFYUI_ROOT", "/home/marc/nvme-data/.comfyui")
    return Path(os.environ.get("COMFYUI_INPUT_DIR", str(Path(root) / "input")))


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{_base()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _extract_frame(video: Path, at_seconds: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(at_seconds), "-i", str(video),
         "-frames:v", "1", str(out)], check=True)


def _extract_mask_frame(mask_video: Path, frame_index: int, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mask_video),
         "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(out)], check=True)


def build_graph(base_image: str, mask_image: str, prefix: str, prompt: str,
                denoise: float = 0.95) -> dict:
    """Source frame -> plate, regenerating ONLY the masked subject."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-dev-Q4_K_S.gguf"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
            "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": base_image}},
        "5": {"class_type": "LoadImage", "inputs": {"image": mask_image}},
        "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
        "7": {"class_type": "VAEEncodeForInpaint", "inputs": {
            "pixels": ["4", 0], "vae": ["3", 0], "mask": ["6", 0], "grow_mask_by": 8}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": 23, "steps": 28, "cfg": 4.5, "sampler_name": "euler",
            "scheduler": "simple", "denoise": denoise, "model": ["1", 0],
            "positive": ["8", 0], "negative": ["8", 0], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    video = Path(sys.argv[1])
    out = Path(sys.argv[2])
    mask_video = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    prompt = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PROMPT

    inp = _input_dir()
    inp.mkdir(parents=True, exist_ok=True)
    _extract_frame(video, 1.0, inp / "plate_base.png")
    mask_name = "plate_mask.png"
    if mask_video is not None:
        _extract_mask_frame(mask_video, 24, inp / mask_name)

    prefix = f"plate/{out.stem}"
    graph = build_graph("plate_base.png", mask_name, prefix, prompt)
    submitted = _post("/prompt", {"prompt": graph})
    prompt_id = submitted["prompt_id"]
    print(f"submitted plate job {prompt_id}")

    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{_base()}/history/{prompt_id}", timeout=20) as resp:
                hist = json.loads(resp.read())
        except urllib.error.URLError:
            time.sleep(5)
            continue
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            if entry["status"].get("status_str") == "error":
                print("plate render failed", file=sys.stderr)
                return 1
            for node_out in entry.get("outputs", {}).values():
                for image in node_out.get("images", []):
                    produced = inp.parent / "output" / image.get("subfolder", "") / image["filename"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(produced.read_bytes())
                    print(f"wrote {out}")
                    return 0
            break
        time.sleep(5)
    print("timed out waiting for the plate", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
