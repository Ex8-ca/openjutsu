"""Build a scene-consistent reference plate for Object Swap.

The problem
-----------
Feeding H3 a reference image of a *different* character in a *different*
scene makes it regenerate the whole frame and relocate the subject: the
controlnet pulls the output toward the source, the reference pulls it toward
the plate, and the model resolves the conflict by rebuilding the scene. The
swap then lands in the wrong place or the background drifts.

The fix
-------
Build the reference plate **from the source frame** first, so the plate
already agrees with the source on scene, lighting, framing and pose, and
differs only in the character. A local img2img pass does this: the source
frame is the init image, the character is the prompt, and the result is a
plate of the new character standing exactly where the old one stood, in
exactly the same light.

H3 then has no reason to move anything - the plate and the source agree on
everything except identity - so the masked region is the only thing that
changes.

Usage
-----
    python scripts/make_plate.py <source_video> <out.png> [denoise] [prompt]

    # defaults: 0.55 denoise, a generic "the person in <Picture 1>" subject

Run it through ComfyUI's API so the model does the work; this script
submits the graph and polls for the result.
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
    "holding a katana, standing in a sunlit wildflower meadow at golden hour, "
    "photorealistic, sharp focus"
)


def _comfy_base() -> str:
    return os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")


def _input_dir() -> Path:
    root = os.environ.get("COMFYUI_ROOT", "/home/marc/nvme-data/.comfyui")
    return Path(os.environ.get("COMFYUI_INPUT_DIR", str(Path(root) / "input")))


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{_comfy_base()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _extract_frame(video: Path, at_seconds: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(at_seconds), "-i", str(video),
         "-frames:v", "1", str(out)],
        check=True,
    )


def build_graph(base_image: str, prefix: str, denoise: float, prompt: str) -> dict:
    """The img2img graph: source frame -> plate with the new character."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-dev-Q4_K_S.gguf"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
            "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": base_image}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": 11, "steps": 24, "cfg": 4.5, "sampler_name": "euler",
            "scheduler": "simple", "denoise": denoise, "model": ["1", 0],
            "positive": ["6", 0], "negative": ["6", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    video = Path(sys.argv[1])
    out = Path(sys.argv[2])
    denoise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
    prompt = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PROMPT

    inp = _input_dir()
    inp.mkdir(parents=True, exist_ok=True)
    base_name = "plate_base.png"
    _extract_frame(video, 1.0, inp / base_name)

    prefix = f"plate/{out.stem}"
    graph = build_graph(base_name, prefix, denoise, prompt)
    submitted = _post("/prompt", {"prompt": graph})
    prompt_id = submitted["prompt_id"]
    print(f"submitted plate job {prompt_id}")

    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{_comfy_base()}/history/{prompt_id}", timeout=20) as resp:
                hist = json.loads(resp.read())
        except urllib.error.URLError:
            time.sleep(5)
            continue
        entry = hist.get(prompt_id)
        if entry and entry.get("status", {}).get("completed"):
            status = entry["status"].get("status_str")
            if status == "error":
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
