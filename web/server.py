"""Openjutsu web UI — FastAPI app exposing the motion-transfer workflow as a
Genjutsu-style form. Files uploaded here are written into ComfyUI's input/
directory, then the workflow is submitted via the Comfy SDK pointed at
comfy-api-proxy (port 8189 → ComfyUI daemon on 8188).

Run:
    PYTHONPATH="" /home/marc/nvme-data/.comfyui/.venv/bin/python \\
        /home/marc/nvme-data/cinematic/openjutsu/web/server.py

Listens on http://0.0.0.0:8765 by default. Override with --host / --port.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

# Make sure the ComfyUI venv site-packages are on sys.path so we can import comfy_sdk.
_REPO = Path("/home/marc/repos/openjutsu")
_COMFYUI_VENV_SITE = Path("/home/marc/nvme-data/.comfyui/.venv/lib/python3.13/site-packages")
os.environ["PYTHONPATH"] = ""
if str(_COMFYUI_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_COMFYUI_VENV_SITE))

# IMPORTANT: set COMFY_BASE_URL BEFORE constructing the SDK client.
os.environ["COMFY_BASE_URL"] = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402


REPO_ROOT = _REPO
WEB_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = WEB_ROOT / "static"
WORKFLOW_PATH = REPO_ROOT / "workflows" / "openjutsu_api.json"

COMFYUI_INPUT_DIR = Path("/home/marc/nvme-data/.comfyui/input")
OUTPUT_DIR = Path("/home/marc/nvme-data/cinematic/openjutsu/web/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry. Single-process dev server; replace with Redis/DB for production.
JOBS: dict[str, dict] = {}


def _save_upload(prefix: str, upload: UploadFile, dest_dir: Path) -> Path:
    """Persist an UploadFile to dest_dir with a unique prefix. Returns the dest path."""
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


_DEFAULT_PROMPT = (
    "A graceful dancer performing expressive contemporary dance in a sunlit "
    "wildflower meadow at golden hour. The character wears a flowing terracotta "
    "slip dress, hair swept back, arms extended. The scene includes overgrown grass, "
    "white wildflowers, a dense tree line, and warm lens flare."
)


def create_app() -> FastAPI:
    from comfy_sdk import Comfy  # imported lazily so the COMFY_BASE_URL env is set

    app = FastAPI(title="Openjutsu", version="0.1.0")

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
        return {"status": "ok", "time": time.time()}

    @app.post("/api/generate")
    async def generate(
        source_video: UploadFile = File(...),
        ref_images: list[UploadFile] = File(default=[]),
        prompt: str = Form(default=""),
        quality: str = Form(default="864x480"),
    ) -> dict:
        if not source_video.filename:
            raise HTTPException(400, "source_video is required")
        # Persist uploads
        video_path = _save_upload("source", source_video, COMFYUI_INPUT_DIR)
        ref_paths: list[Path] = []
        for ref in ref_images[:9]:  # H3 caps ref_images at 9
            if ref.filename:
                ref_paths.append(_save_upload("ref", ref, COMFYUI_INPUT_DIR))

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created": time.time(),
            "source": str(video_path),
            "refs": [str(p) for p in ref_paths],
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
            wf = client.workflows.from_file(str(WORKFLOW_PATH))

            # Source video via LoadVideo (node 174) — accepts asset handle
            source_asset = client.assets.from_file(JOBS[job_id]["source"])
            wf.set_input("174", "file", source_asset)

            # Reference images via LoadImage (node 900) — accepts a filename
            # string relative to ComfyUI's input dir (which is where
            # _save_upload already wrote them).
            if job["refs"]:
                wf.set_input("900", "image", Path(job["refs"][0]).name)

            # Patch the prompt
            wf.set_input("136", "prompt", job["prompt"] or _DEFAULT_PROMPT)

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
                job["outputs"].append({"name": output.name, "url": f"/outputs/{output.name}"})
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