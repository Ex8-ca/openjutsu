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
    import argparse

    parser = argparse.ArgumentParser(
        prog="render.py",
        description="Submit the Openjutsu motion-transfer workflow via the Comfy SDK.",
    )
    parser.add_argument(
        "--workflow", default=str(WORKFLOW),
        help="path to the API-format workflow JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--video", default=str(VIDEO),
        help="source video to transfer motion from (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=str(OUT),
        help="directory to write the rendered video into (default: %(default)s)",
    )
    args = parser.parse_args()

    # CRITICAL: set COMFY_BASE_URL BEFORE constructing the client so the SDK
    # picks up our proxy URL instead of defaulting to Comfy Cloud (which
    # requires an API key).
    os.environ["COMFY_BASE_URL"] = os.environ.get(
        "COMFY_BASE_URL", "http://127.0.0.1:8189"
    )
    base = os.environ["COMFY_BASE_URL"]
    print(f"Using COMFY_BASE_URL={base}")
    client = Comfy()

    workflow_path = Path(args.workflow)
    video_path = Path(args.video)
    out_dir = Path(args.out)

    if not workflow_path.exists():
        print(f"Workflow JSON not found: {workflow_path}", file=sys.stderr)
        return 1
    if not video_path.exists():
        print(f"Source video not found: {video_path}", file=sys.stderr)
        return 1

    print("Loading workflow ...")
    wf = client.workflows.from_file(str(workflow_path))

    print("Uploading source video ...")
    asset = client.assets.from_file(str(video_path))

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
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = job.get_outputs(OUTPUT_NODE_ID)
    print(f"  Found {len(outputs)} outputs")
    for output in outputs:
        out_path = out_dir / output.name
        output.to_file(str(out_path))
        print(f"  Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())