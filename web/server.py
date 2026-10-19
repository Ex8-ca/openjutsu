"""Openjutsu web UI — FastAPI app exposing the motion-transfer workflow as a
Genjutsu-style form. Files uploaded here are written into ComfyUI's input/
directory, then the workflow is submitted via the Comfy SDK pointed at
comfy-api-proxy (port 8189 → ComfyUI daemon on 8188).

Run:
    python web/server.py --host 0.0.0.0 --port 8765

Environment (all optional, sensible defaults derived from the repo):
    COMFY_BASE_URL       Comfy API v2 base URL.        default http://127.0.0.1:8189
    COMFYUI_ROOT         ComfyUI install root.          default /home/marc/nvme-data/.comfyui
    COMFYUI_INPUT_DIR    ComfyUI input/ directory.      default $COMFYUI_ROOT/input
    COMFYUI_VENV_SITE    site-packages dir holding comfy_sdk.
                         default $COMFYUI_ROOT/.venv/lib/python*/site-packages
    OPENJUTSU_OUTPUT_DIR Where rendered outputs are written.
                         default <repo>/web/outputs

Listens on http://0.0.0.0:8765 by default. Override with --host / --port.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import mimetypes
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

# Repository root = parent of the web/ directory this file lives in.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ComfyUI install root. Everything else derives from it.
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", "/home/marc/nvme-data/.comfyui")).resolve()


def _default_venv_site() -> str | None:
    """Locate the site-packages dir that provides comfy_sdk, if any."""
    explicit = os.environ.get("COMFYUI_VENV_SITE")
    if explicit:
        return explicit
    pattern = str(COMFYUI_ROOT / ".venv" / "lib" / "python*" / "site-packages")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


# Make sure the ComfyUI venv site-packages are on sys.path so we can import comfy_sdk.
_VENV_SITE = _default_venv_site()
os.environ["PYTHONPATH"] = ""
if _VENV_SITE and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

# IMPORTANT: set COMFY_BASE_URL BEFORE constructing the SDK client.
os.environ["COMFY_BASE_URL"] = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402


WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
WORKFLOW_PATH = REPO_ROOT / "workflows" / "openjutsu_api.json"
AUDIO_WORKFLOW_PATH = REPO_ROOT / "workflows" / "openjutsu_api_audio.json"
SWAP_WORKFLOW_PATH = REPO_ROOT / "workflows" / "openjutsu_api_swap.json"
SWAP_AUDIO_WORKFLOW_PATH = REPO_ROOT / "workflows" / "openjutsu_api_swap_audio.json"

# "transfer" (Motion Transfer) or "swap" (Object Swap).
MODE_TRANSFER = "transfer"
MODE_SWAP = "swap"

# The source-audio-preserving variant needs the VRGDG custom node. When that
# pack is not installed the node type does not exist and ComfyUI rejects the
# whole prompt, so probe the daemon's object_info and pick accordingly.
# Override with OPENJUTSU_WORKFLOW=core|audio to force a variant.
_WORKFLOW_CHOICE = os.environ.get("OPENJUTSU_WORKFLOW", "auto").lower()


def _daemon_object_info(node_type: str) -> bool:
    """True if the ComfyUI daemon exposes `node_type`."""
    import urllib.error
    import urllib.request

    base = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
    url = f"{base}/object_info/{node_type}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "replace")
        return bool(body.strip()) and node_type in body
    except Exception:
        return False


def _resolve_workflow_path(mode: str = MODE_TRANSFER) -> Path:
    """Pick the workflow file to submit.

    mode "transfer" -> Motion Transfer (Canny control)
    mode "swap"     -> Object Swap (mask + source-video inpaint)
    """
    if mode == MODE_SWAP:
        swap_core = SWAP_WORKFLOW_PATH
        swap_audio = SWAP_AUDIO_WORKFLOW_PATH
    else:
        swap_core = WORKFLOW_PATH
        swap_audio = AUDIO_WORKFLOW_PATH

    if _WORKFLOW_CHOICE in ("core", "audio"):
        return swap_audio if _WORKFLOW_CHOICE == "audio" else swap_core
    if swap_audio.exists() and _daemon_object_info("VRGDG_MiniMaxH3AudioDrive"):
        return swap_audio
    return swap_core

COMFYUI_INPUT_DIR = Path(
    os.environ.get("COMFYUI_INPUT_DIR", str(COMFYUI_ROOT / "input"))
).resolve()
OUTPUT_DIR = Path(
    os.environ.get("OPENJUTSU_OUTPUT_DIR", str(REPO_ROOT / "web" / "outputs"))
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Where uploads are staged before being pushed to ComfyUI over the SDK
# asset API. Local to this machine by design.
STAGING_DIR = Path(
    os.environ.get("OPENJUTSU_STAGING_DIR", str(REPO_ROOT / "web" / "staging"))
).resolve()
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry. Single-process dev server; replace with Redis/DB for production.
JOBS: dict[str, dict] = {}


def _stage_upload(prefix: str, upload: UploadFile, dest_dir: Path) -> Path:
    """Stage an UploadFile locally with a unique prefix. Returns the path.

    The staged copy is a staging area only — it is uploaded to ComfyUI
    through the SDK asset API, not referenced by filename.
    """
    if not upload.filename:
        raise HTTPException(400, "Empty filename")
    # Preserve a sanitized suffix
    suffix = Path(upload.filename).suffix or ".bin"
    safe = "".join(c for c in Path(upload.filename).stem if c.isalnum() or c in "._-")[:64]
    dest = dest_dir / f"{prefix}_{uuid.uuid4().hex[:8]}_{safe}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


# UI quality label -> target megapixels for the ResolutionSelector.
_QUALITY_MEGAPIXELS = {
    "864x480": 0.4,    # 16:9 -> 864x480
    "480x272": 0.13,   # 16:9 -> 480x272
}

_DEFAULT_PROMPT = (
    "A graceful dancer performing expressive contemporary dance in a sunlit "
    "wildflower meadow at golden hour. The character wears a flowing terracotta "
    "slip dress, hair swept back, arms extended. The scene includes overgrown grass, "
    "white wildflowers, a dense tree line, and warm lens flare."
)


# Bundled example assets, used to fill a reference slot the user left
# empty. They are uploaded to ComfyUI through the SDK asset API like any
# other file, so no shared filesystem is assumed.
_EXAMPLE_REFS = {
    "person": "dancer_ref.png",
    "scene": "source_frame_832.png",
    "outfit": "dancer_ref.png",
}


def _example_ref_for(role: str) -> Path | None:
    """Path to a bundled example image for `role`, or None if unavailable."""
    name = _EXAMPLE_REFS.get(role)
    if not name:
        return None
    src = REPO_ROOT / "examples" / name
    return src if src.exists() else None


def create_app() -> FastAPI:
    from comfy_sdk import Comfy  # imported lazily so the COMFY_BASE_URL env is set

    app = FastAPI(title="Openjutsu", version="0.1.0")

    # CORS — local dev server, allow any origin so the page can be loaded
    # from a Hermes browser bridge / Tailscale proxy and still POST to /api/*.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # Serve /static/* from the local static/ folder (CSS, JS).
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

    # Rendered outputs: served from OUTPUT_DIR under /outputs/{filename}.
    # We mount OUTPUT_DIR directly so it doesn't conflict with /static.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "time": time.time(),
            "swap_ready": SWAP_WORKFLOW_PATH.exists(),
            "union2_present": _daemon_object_info("MiniMaxH3FunControlNetApply"),
        }

    @app.post("/api/enhance-prompt")
    async def enhance_prompt_endpoint(prompt: str = Form(...)) -> dict:
        from prompt_enhance import enhance_prompt
        return {"enhanced": enhance_prompt(prompt)}

    @app.post("/api/generate")
    async def generate(
        source_video: UploadFile = File(...),
        ref_images: list[UploadFile] = Form(default=[]),
        refs_role: list[str] = Form(default=[]),
        prompt: str = Form(default=""),
        quality: str = Form(default="864x480"),
        mode: str = Form(default=MODE_TRANSFER),
        mask_image: UploadFile | None = File(default=None),
    ) -> dict:
        if not source_video.filename:
            raise HTTPException(400, "source_video is required")
        if mode not in (MODE_TRANSFER, MODE_SWAP):
            raise HTTPException(400, f"unknown mode: {mode}")
        if mode == MODE_SWAP and (mask_image is None or not mask_image.filename):
            raise HTTPException(400, "Object Swap needs a mask image")
        # Persist uploads
        video_path = _stage_upload("source", source_video, STAGING_DIR)
        ref_paths: list[Path] = []
        for ref in ref_images[:9]:  # H3 caps ref_images at 9
            if ref.filename:
                ref_paths.append(_stage_upload("ref", ref, STAGING_DIR))
        mask_path: Path | None = None
        if mode == MODE_SWAP and mask_image is not None and mask_image.filename:
            mask_path = _stage_upload("mask", mask_image, STAGING_DIR)

        # refs_role comes through as a parallel list (e.g., ["person","scene","outfit"])
        # to label which ref slot each file belongs to. Pad to length of ref_paths.
        role_labels = list(refs_role) + [""] * (len(ref_paths) - len(refs_role))
        role_labels = role_labels[:len(ref_paths)]

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created": time.time(),
            "source": str(video_path),
            "refs": [str(p) for p in ref_paths],
            "refs_role": role_labels,
            "mode": mode,
            "mask": str(mask_path) if mask_path else None,
            "prompt": prompt.strip(),
            "quality": quality,
            "log": [],
            "error": None,
            "outputs": [],
        }

        # Fire-and-forget task (returns immediately, frontend polls /api/jobs/{id})
        asyncio.get_event_loop().create_task(_run_job(job_id))

        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        # Surface only what's safe to expose
        return {
            "id": job["id"],
            "status": job["status"],
            "created": job["created"],
            "log": job["log"][-20:],  # last 20 lines
            "error": job["error"],
            "outputs": job["outputs"],
        }

    async def _run_job(job_id: str) -> None:
        job = JOBS[job_id]
        try:
            job["status"] = "submitting"
            client = Comfy()
            mode = job.get("mode", MODE_TRANSFER)
            workflow_path = _resolve_workflow_path(mode)
            job["log"].append(f"workflow: {workflow_path.name} (mode: {mode})")
            wf = client.workflows.from_file(str(workflow_path))

            # Source video via LoadVideo (node 174) — accepts asset handle
            source_asset = client.assets.from_file(JOBS[job_id]["source"])
            wf.set_input("174", "file", source_asset)

            # Save a still frame of the source for triptych display
            import subprocess
            job_id_safe = job_id
            src_still = OUTPUT_DIR / f"job_{job_id_safe}_source.png"
            try:
                # ffmpeg grabs one frame at t=1s; works for mp4/mov/avi
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", "1", "-i", JOBS[job_id_safe]["source"],
                     "-frames:v", "1", str(src_still)],
                    check=True, capture_output=True, timeout=30
                )
            except Exception:
                # Fallback: copy the source directly
                shutil.copy(JOBS[job_id_safe]["source"], src_still)

            # Reference images — wire each into node 900/901/902 by role
            # (person/scene/outfit).
            #
            # Every slot must resolve to a file that actually exists in
            # ComfyUI's input/ directory, or ComfyUI rejects the entire
            # prompt with "Invalid image file". Slots the user did not fill
            # are therefore pointed at a bundled example asset, and their
            # `ref_images.ref_image_N` link is removed so H3 does not see a
            # reference the user never asked for.
            role_to_node = {"person": "900", "scene": "901", "outfit": "902"}
            node_to_role = {v: k for k, v in role_to_node.items()}
            roles = list(job.get("refs_role") or [])
            roles += [""] * (len(job["refs"]) - len(roles))

            # Map role -> staged upload path (first upload wins for a role).
            chosen: dict[str, Path] = {}
            for ref_path, role in zip(job["refs"], roles):
                role = role or "person"
                chosen.setdefault(role, Path(ref_path))

            for node_id, role in node_to_role.items():
                source_path = chosen.get(role)
                if source_path is None:
                    # Unused slot. Drop the ref link so H3 does not see a
                    # reference the user never asked for, and leave the
                    # LoadImage node pointing at a bundled example that the
                    # SDK uploads below.
                    slot = "ref_images.ref_image_%s" % node_id[-1]
                    wf.json["136"]["inputs"].pop(slot, None)
                    source_path = _example_ref_for(role)
                    if source_path is None:
                        continue
                # Upload through the SDK asset API. The proxy pushes the bytes
                # into the DAEMON's input directory and substitutes the right
                # filename at submit time — which is what ComfyUI's LoadImage
                # resolves against. Passing a bare filename here only works
                # when this app happens to share a filesystem with the daemon.
                wf.set_input(node_id, "image", client.assets.from_file(str(source_path)))

            # Object Swap: the mask is the region to regenerate. Upload it
            # through the SDK asset API like every other file, so the daemon
            # resolves it in its own input directory.
            if mode == MODE_SWAP and job.get("mask"):
                wf.set_input("821", "image", client.assets.from_file(job["mask"]))

            # Patch the prompt
            wf.set_input("136", "prompt", job["prompt"] or _DEFAULT_PROMPT)

            # Apply the quality selection to the ResolutionSelector.
            # 0.4 MP -> 864x480 (16:9), 0.13 MP -> 480x272.
            megapixels = _QUALITY_MEGAPIXELS.get(job.get("quality", ""), 0.4)
            wf.set_input("115", "megapixels", megapixels)

            job["status"] = "rendering"
            job["log"].append("submitted, waiting for completion")

            render_job = client.run(wf)
            job["log"].append(f"job_id={render_job.id}")

            t0 = time.monotonic()
            last = None
            # render_job.events() returns a sync generator (not async), despite
            # the comfy-sdk's async naming. Use a regular `for`.
            for status in render_job.events():
                s = str(getattr(status, "status", status))
                if s != last:
                    job["log"].append(f"[{time.monotonic()-t0:6.1f}s] {s}")
                    last = s

            job["status"] = "downloading"
            outputs = render_job.get_outputs("707")
            for output in outputs:
                dest = OUTPUT_DIR / output.name
                output.to_file(str(dest))
                label = "Output video" if output.name.endswith(".mp4") else output.name
                job["outputs"].append({"name": output.name, "url": f"/outputs/{output.name}", "label": label})
            # Append the source still + first reference for the triptych
            src_still_name = src_still.name
            if (OUTPUT_DIR / src_still_name).exists():
                job["outputs"].append({
                    "name": src_still_name,
                    "url": f"/outputs/{src_still_name}",
                    "label": "Source video (t=1s)",
                })
            if job["refs"]:
                ref_path = Path(job["refs"][0])
                ref_roles = list(job.get("refs_role") or [])
                ref_label = f"Reference ({ref_roles[0] if ref_roles else 'person'})"
                if ref_path.exists():
                    # Copy into the output dir so the browser can show it.
                    try:
                        shown = OUTPUT_DIR / ref_path.name
                        if not shown.exists():
                            shutil.copy2(ref_path, shown)
                        job["outputs"].append({
                            "name": shown.name,
                            "url": f"/outputs/{shown.name}",
                            "label": ref_label,
                        })
                    except OSError:
                        pass
            job["status"] = "complete"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            job["log"].append(f"ERROR: {job['error']}")

    return app


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    import uvicorn

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())