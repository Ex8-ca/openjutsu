#!/usr/bin/env python3
"""Build a per-frame tracked mask for a static-camera clip.

The camera does not move, so the temporal median of the clip is a decent
background estimate. Foreground = the dancer, per frame. The result is a
grayscale video (white = regenerate) that the Object Swap workflow reads
through LoadVideo -> Video Slice -> GetVideoComponents -> ImageToMask(red).

    python scripts/make_mask.py examples/dancer_field_pose.mp4 out_mask.mp4

Only ffmpeg and numpy are needed.

How the subject is found
------------------------
Background subtraction alone does not work on this material, and the failure
is silent - the mask just misses the subject:

* The median keeps a *ghost* of the dancer wherever she lingers, so the
  difference against it goes to zero there and the mask loses her exactly
  where she is largest.
* The largest difference blob is often a sun flare or a patch of rustling
  foliage, not the dancer, so "keep the biggest blob" anchors the mask on
  the background.
* A person is not one connected component. An outstretched arm detaches
  from the torso, and a bent pose puts the head far from the body's centre
  of mass. Both get dropped as separate blobs, and the swap then leaves the
  old arm or head behind.

So the mask is built the other way round. The dancer's costume has a colour
nothing else in the scene has - saturated red, where the sunlit meadow is
yellow - so that is the anchor. Everything else (head, arms, hair) is added
by taking the difference blobs that *touch* the anchor, using proximity
rather than centroid distance, so a leaning head is still recognised as
part of the subject.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, FPS = 672, 384, 24
THRESHOLD = 60       # summed abs difference against the background
DILATE = 14          # pixels of dilation, to cover the soft edge of the
                    # subject and the motion blur around it.

# The subject's costume colour. Red must clearly exceed GREEN: golden
# sunlit grass is warm too, but it is yellow, so green stays high there.
# Measured on the meadow clip: this fires on 0.8% of the frame and on
# essentially none of the median background.
COSTUME_MIN_RED = 120
COSTUME_MIN_GREEN_GAP = 55

# Proximity, not centroid distance. When the subject leans or bends, her
# head is a separate blob far from the body's centre of mass but still
# physically attached; a centroid test drops it.
TOUCH_DISTANCE = 70  # px a blob may sit from the anchor and still be kept
MIN_BLOB_FRACTION = 0.10   # of the anchor, for costume sub-blobs
MIN_DIFF_BLOB = 200        # px, for difference blobs


def decode(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"scale={WIDTH}:{HEIGHT}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (WIDTH * HEIGHT * 3)
    return np.frombuffer(out, np.uint8)[: n * WIDTH * HEIGHT * 3].reshape(n, HEIGHT, WIDTH, 3)


def label_blobs(mask: np.ndarray):
    """Label 4-connected blobs and return (labels, sizes), without scipy."""
    labels = np.zeros(mask.shape, np.int32)
    current = 0
    sizes: dict[int, int] = {}
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1
                stack = [(sy, sx)]
                labels[sy, sx] = current
                size = 0
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                sizes[current] = size
    return labels, sizes


def dilate(mask: np.ndarray, rounds: int = DILATE) -> np.ndarray:
    out = mask.copy()
    for _ in range(rounds):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def costume_mask(frame: np.ndarray) -> np.ndarray:
    """The subject's costume: saturated red, nothing else in the scene."""
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    return (red > COSTUME_MIN_RED) & ((red - green) > COSTUME_MIN_GREEN_GAP)


def subject_mask(frame: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """The subject in one frame, anchored on the costume colour."""
    costume = costume_mask(frame)
    if not costume.any():
        return np.zeros(costume.shape, bool)

    # 1. Anchor: the largest costume blob, plus costume blobs touching it.
    labels, sizes = label_blobs(costume)
    order = sorted(sizes, key=sizes.get, reverse=True)
    anchor = order[0]
    keep = {anchor}
    grown = dilate(labels == anchor, TOUCH_DISTANCE)
    for blob in order[1:]:
        if sizes[blob] >= sizes[anchor] * MIN_BLOB_FRACTION and ((labels == blob) & grown).any():
            keep.add(blob)
    region = np.isin(labels, list(keep))

    # 2. Add the difference blobs that touch the anchor - the head, the
    #    arms, the hair. Proximity only, so a bent pose still connects.
    fg = diff > THRESHOLD
    if fg.any():
        diff_labels, diff_sizes = label_blobs(fg)
        near = dilate(region, TOUCH_DISTANCE)
        for blob, size in diff_sizes.items():
            if size >= MIN_DIFF_BLOB and ((diff_labels == blob) & near).any():
                region |= diff_labels == blob
    return region


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("swap_mask_track.mp4")
    frames = decode(src)
    if frames.shape[0] == 0:
        print(f"no frames decoded from {src}", file=sys.stderr)
        return 1
    background = np.median(frames, axis=0).astype(np.int16)
    diff = np.abs(frames.astype(np.int16) - background).sum(axis=3)

    tmp = dst.parent / (dst.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    for t in range(frames.shape[0]):
        mask = dilate(subject_mask(frames[t], diff[t]))
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmp / f"m{t:04d}.png")

    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
           "-i", str(tmp / "m%04d.png"), "-c:v", "libx264", "-pix_fmt", "gray",
           "-crf", "0", "-preset", "veryslow", str(dst)]
    subprocess.run(cmd, check=True)
    for leftover in tmp.glob("*.png"):
        leftover.unlink()
    tmp.rmdir()
    print(f"wrote {dst} ({frames.shape[0]} frames)")
    return 0


if __name__ == "__main__":
    from PIL import Image
    raise SystemExit(main())
