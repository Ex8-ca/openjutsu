"""Track a user-drawn region across a clip, producing a per-frame mask.

Why
---
The Object Swap workflow reads a mask and regenerates that region, keeping
the rest of the source. A hand-made mask image is a poor way to express
"this bit": you have to paint it offline, in the source's pixel
coordinates, and you get it subtly wrong.

A rectangle drawn on the frame in the UI is much better. But the subject
moves, so a static rectangle would drift off it over the clip. Measured on
the example clip, an automatically built per-frame mask drifted by up to 15
frames (mask/dancer IoU between 0.08 and 0.28), which made the swap land
on the wrong part of the shot in the later frames.

So: seed the tracker with the region the user drew, and let it follow the
subject from there.

Method
------
For a static camera the temporal median of the clip is the clean
background, so foreground = |frame - background| over a threshold. That
gives a per-frame foreground blob. The user-drawn rectangle tells us WHICH
blob they meant (the one under the rectangle), so we keep the connected
component that overlaps the rectangle most and discard the rest.

The result is a grayscale video (white = regenerate) that the workflow
reads through LoadVideo -> Video Slice -> GetVideoComponents ->
ImageToMask(red).

If the clip is not static, or the region cannot be tracked, this returns
None and the caller falls back to the mask the user drew.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["track_region", "TrackingUnavailable"]

# Analysis resolution. Small enough to be fast, big enough to find the
# subject. The mask is written at this size and the workflow scales it.
_W, _H = 672, 384
_FPS = 24

# Summed absolute difference against the background that counts as
# foreground. Tuned for the example clip.
_THRESHOLD = 90

# Morphological closing applied to the foreground before labelling.
#
# Without it the dancer fragments into 2000+ pieces: her dress against the
# sunlit meadow barely clears the threshold in places, so the silhouette
# breaks up and "the blob under the rectangle" becomes a scrap of grass
# instead of the dancer. Closing merges the pieces back into one subject.
_CLOSE = 5

# The tracked region is capped at this fraction of the frame. The foreground
# test also fires on the bright sky and on grass moving in the wind, and
# without a cap the region grows until it swallows the background. The
# composite restores the background from the source afterwards, but a mask
# that covers the whole frame is not a region any more.
_MAX_COVERAGE = 0.35

# Dilation applied to the tracked region, in analysis pixels. The
# replacement subject is often LARGER than the one it replaces (armour, a
# hat, a prop), so the hole has to be generous or the new subject gets
# clipped and the composite shows a hybrid.
_DILATE = 22


class TrackingUnavailable(Exception):
    """The region could not be tracked; the caller should fall back."""


def _decode(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={_W}:{_H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(out) // (_W * _H * 3)
    if n == 0:
        raise TrackingUnavailable("no frames decoded")
    return np.frombuffer(out, np.uint8)[: n * _W * _H * 3].reshape(n, _H, _W, 3)


def _dilate(mask: np.ndarray, rounds: int = _DILATE) -> np.ndarray:
    out = mask.copy()
    for _ in range(rounds):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Label 4-connected components without needing scipy."""
    labels = np.zeros(mask.shape, np.int32)
    h, w = mask.shape
    current = 0
    members: list[list[int]] = []
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and labels[sy, sx] == 0:
                current += 1
                stack = [(sy, sx)]
                labels[sy, sx] = current
                cells: list[int] = []
                while stack:
                    y, x = stack.pop()
                    cells.append(y * w + x)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                members.append(cells)
    return labels, members


def _close(mask: np.ndarray, rounds: int = _CLOSE) -> np.ndarray:
    """Binary closing: merge the fragments of a broken-up silhouette.

    Closing = dilate then erode. Dilating first bridges the gaps between
    the pieces of the subject; eroding afterwards puts the boundary back
    where it was. (The other order is an opening, which erodes the subject
    away instead of joining it up.)
    """
    if rounds <= 0:
        return mask
    out = _dilate(mask, rounds)
    for _ in range(rounds):
        n = out.copy()
        n[1:, :] &= out[:-1, :]
        n[:-1, :] &= out[1:, :]
        n[:, 1:] &= out[:, :-1]
        n[:, :-1] &= out[:, 1:]
        out = n
    return out


def _rect_mask(rect: tuple, h: int, w: int) -> np.ndarray:
    """A boolean mask of the rectangle the user drew."""
    ry0, ry1, rx0, rx1 = rect
    out = np.zeros((h, w), bool)
    out[ry0:ry1, rx0:rx1] = True
    return out


def _pick_blob(labels: np.ndarray, members: list[list[int]], rect: tuple,
               h: int, w: int) -> np.ndarray:
    """The component that overlaps the user's rectangle most.

    "Most" is weighted by size, not just raw overlap: a small blob of
    blowing grass that happens to sit under the rectangle can easily have
    more pixels inside it than a dancer whose arm is mostly outside. So
    score each component by the area it covers inside the rectangle, and
    break ties toward the larger component.
    """
    ry0, ry1, rx0, rx1 = rect
    rect_mask = np.zeros((h, w), bool)
    rect_mask[ry0:ry1, rx0:rx1] = True

    best_cells: list[int] | None = None
    best_score = 0.0
    for cells in members:
        inside = 0
        for idx in cells:
            y, x = divmod(idx, w)
            if rect_mask[y, x]:
                inside += 1
        if inside == 0:
            continue
        # Coverage of the rectangle this component accounts for, with a
        # small bonus for being a big component in its own right.
        score = inside / float(max(1, (ry1 - ry0) * (rx1 - rx0)))
        score += 0.15 * (len(cells) / float(h * w))
        if score > best_score:
            best_score = score
            best_cells = cells
    if not best_cells:
        raise TrackingUnavailable("nothing under the drawn region")
    out = np.zeros((h, w), bool)
    for idx in best_cells:
        y, x = divmod(idx, w)
        out[y, x] = True
    return out


def track_region(video: Path, region: dict, out_path: Path) -> Path | None:
    """Build a per-frame mask video that follows the drawn region.

    ``region`` is the rectangle the user drew, in SOURCE pixel coordinates:
    ``{"x", "y", "w", "h", "frameW", "frameH"}``.

    Returns the path of the written mask video, or None if the region could
    not be tracked.
    """
    video = Path(video)
    out_path = Path(out_path)

    try:
        frames = _decode(video)
    except (subprocess.CalledProcessError, TrackingUnavailable):
        return None
    if frames.shape[0] < 2:
        return None

    # The rectangle in analysis coordinates.
    fw = float(region.get("frameW") or 0)
    fh = float(region.get("frameH") or 0)
    if fw <= 0 or fh <= 0:
        return None
    sx = _W / fw
    sy = _H / fh
    rx0 = int(max(0, min(_W - 1, region.get("x", 0) * sx)))
    ry0 = int(max(0, min(_H - 1, region.get("y", 0) * sy)))
    rx1 = int(max(rx0 + 1, min(_W, (region.get("x", 0) + region.get("w", 0)) * sx)))
    ry1 = int(max(ry0 + 1, min(_H, (region.get("y", 0) + region.get("h", 0)) * sy)))
    rect = (ry0, ry1, rx0, rx1)

    # Background = temporal median (the camera is static).
    background = np.median(frames, axis=0).astype(np.int16)
    diff = np.abs(frames.astype(np.int16) - background).sum(axis=3)

    prev = np.zeros((_H, _W), bool)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tracked_count = 0
        for t in range(frames.shape[0]):
            fg = _close(diff[t] > _THRESHOLD)
            if not fg.any():
                # Nothing moving in this frame: keep the previous region so
                # the mask does not blink off the subject.
                Image.fromarray((prev * 255).astype(np.uint8)).save(
                    tmp / f"m{t:04d}.png")
                continue
            labels, members = _components(fg)
            if members:
                try:
                    blob = _pick_blob(labels, members, rect, _H, _W)
                except TrackingUnavailable:
                    blob = None
            else:
                blob = None
            if blob is None:
                # Could not identify the subject this frame: keep the
                # previous region rather than punching a hole somewhere else.
                if t:
                    Image.fromarray((prev * 255).astype(np.uint8)).save(tmp / f"m{t:04d}.png")
                    tracked_count += 1
                    continue
                blob = np.zeros((_H, _W), bool)
            blob = _dilate(blob)
            # Keep the region to a sane size: if the blob has swallowed the
            # sky or the grass, keep only the part of it that overlaps the
            # rectangle the user drew.
            if blob.mean() > _MAX_COVERAGE:
                clipped = blob & _rect_mask(rect, _H, _W)
                if clipped.any():
                    blob = clipped
            prev = blob
            Image.fromarray((prev * 255).astype(np.uint8)).save(tmp / f"m{t:04d}.png")
            tracked_count += 1

        if tracked_count == 0:
            return None

        cmd = [
            "ffmpeg", "-v", "error", "-y", "-framerate", str(_FPS),
            "-i", str(tmp / "m%04d.png"), "-c:v", "libx264", "-pix_fmt", "gray",
            "-crf", "0", "-preset", "veryslow", str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return None

    if not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return out_path
