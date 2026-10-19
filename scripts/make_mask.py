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
THRESHOLD = 90       # summed abs difference against the background
DILATE = 70          # pixels of dilation. The replacement subject is often
                    # LARGER than the one it replaces (armour, a hat, a
                    # prop), so the hole has to be generous or the new
                    # subject is clipped and the composite shows a hybrid.
KEEP_BLOBS = 1


def decode(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"scale={WIDTH}:{HEIGHT}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(out) // (WIDTH * HEIGHT * 3)
    return np.frombuffer(out, np.uint8)[: n * WIDTH * HEIGHT * 3].reshape(n, HEIGHT, WIDTH, 3)


def keep_largest(mask: np.ndarray, k: int = KEEP_BLOBS) -> np.ndarray:
    """Keep the k biggest 4-connected blobs, without needing scipy."""
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
    if not sizes:
        return mask
    keep = sorted(sizes, key=sizes.get, reverse=True)[:k]
    return np.isin(labels, keep)


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
        fg = dilate(keep_largest(fg))
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
