#!/usr/bin/env python3
"""Openjutsu — submit the motion-transfer workflow via Comfy SDK.

The Comfy SDK (`comfy-sdk` >= 0.3.0) is the official Python client for the
Comfy Developer Platform. Pointed at a local `comfy-api-proxy` (port 8189)
that fronts a self-hosted ComfyUI daemon (port 8188). This satisfies the
"Comfy Router and/or Comfy SDK" rule from the Open Source Higgsfield
contest.

Required:
    pip install comfy-sdk>=0.3.0 comfy-api-proxy httpx

Environment:
    COMFY_BASE_URL      defaults to http://127.0.0.1:8189
    COMFY_PROXY_TOKEN   optional; bearer token if your proxy requires one
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Clear PYTHONPATH so we don't pick up unrelated site-packages from a parent
# shell environment (e.g. a Hermes venv with a different Python).
os.environ.setdefault("PYTHONPATH", "")

# Add the ComfyUI venv's site-packages if it isn't on sys.path already. This
# lets you `pip install comfy-sdk` into the venv that ships with ComfyUI and
# then invoke this script with `/path/to/.venv/bin/python scripts/render.py`
# from any cwd.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _candidate in (
    _REPO / ".venv" / "lib" / "python3.13" / "site-packages",
    Path("/home/marc/nvme-data/.comfyui/.venv/lib/python3.13/site-packages"),
):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from comfy_sdk import Comfy  # noqa: E402


WORKFLOW = _REPO / "workflows" / "openjutsu_api.json"
VIDEO = _REPO / "examples" / "dancer_field_pose.mp4"
OUT = _REPO / "examples" / "output"
OUTPUT_NODE_ID = "707"  # SaveVideo node in the workflow
INPUT_NODE_ID = "174"   # LoadVideo node in the workflow


def main() -> int:
    base = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8189")
    print(f"Using COMFY_BASE_URL={base}")
    client = Comfy()

    if not WORKFLOW.exists():
        print(f"Workflow JSON not found: {WORKFLOW}", file=sys.stderr)
        return 1
    if not VIDEO.exists():
        print(f"Source video not found: {VIDEO}", file=sys.stderr)
        return 1

    print("Loading workflow ...")
    wf = client.workflows.from_file(str(WORKFLOW))

    print("Uploading source video ...")
    asset = client.assets.from_file(str(VIDEO))

    print(f"Setting input on node {INPUT_NODE_ID} ...")
    wf.set_input(INPUT_NODE_ID, "file", asset)

    print("Submitting job ...")
    job = client.run(wf)
    print(f"  Job id: {getattr(job, 'id', '?')}")

    print("\nWaiting for completion (3-5 minutes on 16 GB VRAM) ...")
    t0 = time.monotonic()
    last = None
    for status in job.events():
        s = str(getattr(status, "status", status))
        if s != last:
            print(f"  [{time.monotonic() - t0:6.1f}s] {s}")
            last = s

    print(f"\nDone in {time.monotonic() - t0:.1f}s. Fetching outputs from node {OUTPUT_NODE_ID} ...")
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = job.get_outputs(OUTPUT_NODE_ID)
    print(f"  Found {len(outputs)} outputs")
    for output in outputs:
        out_path = OUT / output.name
        output.to_file(str(out_path))
        print(f"  Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())