#!/usr/bin/env python3
"""Build a per-frame tracked mask for a static-camera clip.

The camera does not move, so the temporal median of the clip is the clean
background. Foreground = the dancer, per frame. The result is a grayscale
video (white = regenerate) that the Object Swap workflow reads through
LoadVideo -> Video Slice -> GetVideoComponents -> ImageToMask(red).

    python scripts/make_mask.py examples/dancer_field_pose.mp4 out_mask.mp4

Only ffmpeg and numpy are needed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, FPS = 672, 384, 24
THRESHOLD = 60       # summed abs difference against the background
DILATE = 12          # pixels of dilation, to cover the soft edge of the
                    # subject and any motion blur around it.
KEEP_BLOBS = 1
# A person is not one blob: an outstretched arm, a trailing sleeve or a
# strand of hair detaches from the torso and shows up as its own blob.
# Keeping only the largest blob drops those parts, and the swap then leaves
# them behind - a composite showing the new subject wearing the old one's
# arm. So keep every blob that is (a) a meaningful fraction of the main one
# and (b) spatially close to it.
MIN_BLOB_FRACTION = 0.05
# Proximity, not centroid distance. When the subject leans or bends, her head
# is a separate blob far from the body's centre of mass but still physically
# attached; a centroid test drops it and the swap leaves her head behind.
# Growing the main blob and keeping whatever touches it keeps the parts.
MAX_BLOB_DISTANCE = 50
# The subject's own colour. The temporal-median background contains a ghost
# of a performer who lingers in one spot, so difference alone loses her
# there; colour finds her anyway.
DRESS_MIN_RED = 95
DRESS_MIN_BLUE_GAP = 35
DRESS_MIN_GREEN_GAP = 28


def decode(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"scale={WIDTH}:{HEIGHT}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (WIDTH * HEIGHT * 3)
    return np.frombuffer(out, np.uint8)[: n * WIDTH * HEIGHT * 3].reshape(n, HEIGHT, WIDTH, 3)


def label_blobs(mask: np.ndarray):
    """Label 4-connected blobs and return (labels, sizes), no scipy needed."""
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


def keep_subject(mask: np.ndarray, k: int = KEEP_BLOBS) -> np.ndarray:
    """Keep the main blob plus any nearby blob that is part of the subject.

    A limb or a sleeve that has detached from the torso is a separate
    connected component. Dropping it leaves the old subject's arm in the
    final composite, so those are kept too when they are big enough and
    close enough to the main body.
    """
    labels, sizes = label_blobs(mask)
    if not sizes:
        return mask
    order = sorted(sizes, key=sizes.get, reverse=True)
    keep = {order[0]}
    # Proximity = touching the main blob grown by MAX_BLOB_DISTANCE pixels.
    grown = dilate(labels == order[0], MAX_BLOB_DISTANCE)
    for blob in order[1:]:
        if sizes[blob] < sizes[order[0]] * MIN_BLOB_FRACTION:
            break
        if ((labels == blob) & grown).any():
            keep.add(blob)
    if k > 1:
        keep.update(order[:k])
    return np.isin(labels, list(keep))


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
        fg = diff[t] > THRESHOLD
        # The temporal median is the dancer's ghost where she lingers: she
        # spends a good part of the clip in one area, so the median keeps her
        # there and the difference against it goes to zero - the mask then
        # loses her exactly when she is biggest. Add the rust colour of her
        # dress, which nothing else in the scene has, so the subject is
        # found by colour as well as by motion.
        frame = frames[t].astype(np.int16)
        red, green, blue = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        dress = ((red > DRESS_MIN_RED)
                 & (red - blue > DRESS_MIN_BLUE_GAP)
                 & (red - green > DRESS_MIN_GREEN_GAP))
        fg = keep_subject(fg | dress)
        fg = dilate(fg)
        Image.fromarray((fg * 255).astype(np.uint8)).save(tmp / f"m{t:04d}.png")

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
