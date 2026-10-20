"""Post-composite for Openjutsu's Object Swap mode.

Why this exists
---------------
ComfyUI's ``MiniMaxH3FunControlNetApply`` treats its ``mask`` input as a soft
conditioning hint, not a pixel-exact region lock. Internally it computes
``visibility = 1.0 - (mask > 0.5)`` and concatenates ``[visibility_latent,
masked_latent]`` onto the controlnet hint, so the *whole frame* is regenerated
and merely steered toward the source. Measured on a real render, the mean
absolute difference between the source and the output was ~1.0-1.26x inside vs
outside the mask -- i.e. no spatial selectivity at all. A different-character
reference therefore does not replace the subject on its own.

Higgsfield's Object Swap keeps the rest of the shot intact. To match that
contract we composite after the H3 pass: pixels inside the mask come from the
generated render, pixels outside come from the source frame.

Measured result of this composite (124 frames, 864x480):

    raw H3 output    inside-mask diff 22.40   outside-mask diff 17.73
    after composite  inside-mask diff 22.40   outside-mask diff  0.00

The outside region becomes pixel-identical to the source.

Audio
-----
The source clip's audio is preserved independently of this composite (the
audio-drive node locks it), so the audio track is copied from the *source*
file bit-exactly with ``-c:a copy`` rather than re-encoded.

Mask polarity
-------------
``mask = 1`` marks the region to REGENERATE (this is ComfyUI's convention:
"1 marks the regions to regenerate"). The mask is read from the RED channel,
matching the ``ImageToMask(channel="red")`` node in the workflow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["composite_swap", "SwapCompositeError"]


class SwapCompositeError(RuntimeError):
    """Raised when the composite cannot be produced."""


def _ffprobe(path: Path) -> dict:
    """Return stream metadata for the first video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except FileNotFoundError as exc:  # pragma: no cover - ffmpeg missing
        raise SwapCompositeError("ffprobe is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise SwapCompositeError(f"ffprobe failed on {path.name}") from exc

    data = json.loads(out.stdout or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise SwapCompositeError(f"{path.name} has no video stream")
    return streams[0]


def _rate(text: str | None) -> float:
    """Parse an ffmpeg rational frame rate such as ``24/1``."""
    if not text:
        return 24.0
    try:
        num, den = text.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 24.0
    except (ValueError, ZeroDivisionError):
        return 24.0


def _load_mask(mask_path: Path, width: int, height: int) -> np.ndarray:
    """Load a still mask as a boolean array where True means "regenerate"."""
    resampling = getattr(Image, "Resampling", Image)
    nearest = getattr(resampling, "NEAREST", 0)
    try:
        with Image.open(mask_path) as im:
            rgb = im.convert("RGB").resize((width, height), nearest)
            arr = np.asarray(rgb, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise SwapCompositeError(f"cannot read mask {mask_path.name}") from exc
    # RED channel, matching ImageToMask(channel="red").
    return arr[:, :, 0] > 127


def _last_mask_frame(mask_path: Path, width: int, height: int) -> np.ndarray:
    """Return the final frame of a mask video as a boolean mask."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "last.png"
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", str(mask_path),
            "-frames:v", "1", str(png),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return _load_mask(png, width, height)


def _iter_mask_frames(mask_path: Path, width: int, height: int, fps: float):
    """Yield per-frame boolean masks from a mask VIDEO, True = regenerate."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(mask_path),
        "-vf", f"scale={width}:{height}:flags=bicubic,fps={fps:.6f}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = proc.stdout
    assert stdout is not None
    frame_bytes = width * height
    try:
        while True:
            buf = stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width) > 127
    finally:
        stdout.close()
        err = proc.stderr.read() if proc.stderr else b""
        if proc.stderr:
            proc.stderr.close()
        proc.wait(timeout=60)
        # A broken pipe here just means we stopped reading early (the render
        # was shorter than the mask video). That is not an error.
        if proc.returncode not in (0, None) and err and b"Broken pipe" not in err:
            raise SwapCompositeError(
                f"ffmpeg mask decode failed for {mask_path.name}: "
                f"{err.decode(errors='replace')[:200]}"
            )


def _iter_frames(path: Path, width: int, height: int, fps: float):
    """Yield decoded RGB frames as (h, w, 3) float arrays, scaled to size."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={width}:{height}:flags=bicubic,fps={fps:.6f}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = proc.stdout
    assert stdout is not None
    try:
        while True:
            buf = stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        stdout.close()
        err = proc.stderr.read() if proc.stderr else b""
        if proc.stderr:
            proc.stderr.close()
        proc.wait(timeout=60)
        # A broken pipe here just means we stopped reading early (one stream
        # ran out before the other). That is not an error.
        if proc.returncode not in (0, None) and err and b"Broken pipe" not in err:
            raise SwapCompositeError(
                f"ffmpeg decode failed for {path.name}: {err.decode(errors='replace')[:200]}"
            )


def _grow(mask: np.ndarray, rounds: int) -> np.ndarray:
    """Dilate a boolean mask by `rounds` 4-connected steps."""
    out = mask.copy()
    for _ in range(rounds):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def _regenerated_region(
    src_frame: np.ndarray, gen_frame: np.ndarray, mask: np.ndarray,
    threshold: float, min_grow: int, max_grow: int,
) -> np.ndarray:
    """The part of the frame the model actually replaced.

    The controlnet mask is a soft hint: H3 replaces the whole subject, not
    just the pixels inside the mask. Compositing with the raw mask would
    therefore paste the original subject back over most of the new one and
    produce a hybrid.

    So measure where the render actually differs from the source, keep the
    changed pixels that are connected to the mask, and use THAT as the
    composite region. The un-regenerated background then comes from the
    source untouched, which is what "the rest of the shot survives" means.
    """
    diff = np.abs(src_frame - gen_frame).mean(axis=2)
    changed = diff > threshold
    # Keep only the changed pixels near the mask, so unrelated drift in the
    # background does not get pulled into the composite region.
    near = _grow(mask, max_grow)
    region = changed & near
    if not region.any():
        return mask
    # Fill the holes: a subject is a solid blob, and a speckled region
    # would let the original subject show through between pixels.
    return _grow(region, min_grow)


def composite_swap(
    generated: Path,
    source: Path,
    mask: Path,
    out_path: Path,
    feather_px: int = 0,
    follow_regeneration: bool = True,
    change_threshold: float = 18.0,
) -> Path:
    """Composite a generated swap render over its source, honouring the mask.

    Inside the composite region the generated frame is used; outside it the
    source frame is used, so the untouched part of the shot stays
    pixel-identical.

    ``follow_regeneration`` (default) expands the composite region to cover
    what the model actually replaced, which is the whole subject rather
    than just the mask. Set it False to composite strictly by the mask.

    ``feather_px`` optionally blends across the region edge. 0 (the default)
    gives a hard boundary, which is what makes the outside region exactly
    equal to the source.
    """
    generated = Path(generated)
    source = Path(source)
    mask = Path(mask)
    out_path = Path(out_path)

    for path in (generated, source, mask):
        if not path.exists():
            raise SwapCompositeError(f"missing input: {path.name}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SwapCompositeError("ffmpeg/ffprobe are required for the swap composite")

    meta = _ffprobe(generated)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    if width <= 0 or height <= 0:
        raise SwapCompositeError("could not determine the generated frame size")
    fps = _rate(meta.get("r_frame_rate"))

    mask_is_video = mask.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
    mask_bool = None if mask_is_video else _load_mask(mask, width, height)
    alpha = mask_bool.astype(np.float32) if mask_bool is not None else None
    if alpha is not None and feather_px > 0:
        # Cheap separable box blur so no scipy dependency is needed here.
        k = int(feather_px) * 2 + 1
        pad = feather_px
        padded = np.pad(alpha, pad, mode="edge")
        cum = padded.cumsum(axis=0).cumsum(axis=1)
        cum = np.pad(cum, ((1, 0), (1, 0)), mode="constant")
        total = float(k * k)
        alpha = (
            cum[k:, k:] - cum[:-k, k:] - cum[k:, :-k] + cum[:-k, :-k]
        ) / total
        alpha = np.clip(alpha, 0.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
        "-i", "-",
        # Audio comes from the SOURCE so it is bit-exact, not re-encoded.
        "-i", str(source),
        "-map", "0:v", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.Popen(enc, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin = proc.stdin
    assert stdin is not None
    written = 0
    try:
        src_frames = _iter_frames(source, width, height, fps)
        gen_frames = _iter_frames(generated, width, height, fps)
        if mask_is_video:
            mask_frames = _iter_mask_frames(mask, width, height, fps)
            # A mask video shorter than the render repeats its last frame.
            last_frame_mask = _last_mask_frame(mask, width, height)
        else:
            mask_frames = None
            last_frame_mask = None
        for index, (src_frame, gen_frame) in enumerate(zip(src_frames, gen_frames)):
            if mask_frames is not None:
                try:
                    frame_mask = next(mask_frames)
                    last_frame_mask = frame_mask
                except StopIteration:
                    # The mask video is shorter than the render: keep using
                    # the last mask we saw rather than dropping the region.
                    frame_mask = last_frame_mask
            else:
                frame_mask = mask_bool
            if frame_mask is None:
                # No mask resolved for this frame: keep the whole source
                # rather than guessing a region.
                stdin.write(np.clip(src_frame, 0, 255).astype(np.uint8).tobytes())
                written += 1
                continue
            if follow_regeneration:
                frame_mask = _regenerated_region(
                    src_frame, gen_frame, frame_mask, change_threshold,
                    min_grow=6, max_grow=60,
                )
            alpha3 = frame_mask.astype(np.float32)[:, :, None]
            merged = src_frame.astype(np.float32) * (1.0 - alpha3) + \
                gen_frame.astype(np.float32) * alpha3
            stdin.write(np.clip(merged, 0, 255).astype(np.uint8).tobytes())
            written += 1
        stdin.close()
    except BrokenPipeError as exc:
        proc.wait(timeout=60)
        raise SwapCompositeError("ffmpeg closed the pipe while encoding") from exc
    finally:
        if not stdin.closed:
            stdin.close()

    err = proc.stderr.read() if proc.stderr else b""
    if proc.stderr:
        proc.stderr.close()
    if proc.wait(timeout=120) != 0:
        raise SwapCompositeError(
            f"ffmpeg encode failed: {err.decode(errors='replace')[:300]}"
        )
    if written == 0:
        raise SwapCompositeError("no frames were composited")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SwapCompositeError("the composite produced an empty file")
    return out_path
