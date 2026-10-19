#!/usr/bin/env python3
"""Build the Openjutsu demo video for the Open Source Higgsfield contest.

Two minutes max (the contest says "in case we can't get it running
ourselves"), showing:

  1. Title + what it is
  2. Motion Transfer: source clip -> reference -> result
  3. Object Swap: source + mask -> result (the region stays locked)
  4. The numbers that back the claims (audio, region-lock)
  5. How it is built + where to get it

Everything is rendered from the real assets in examples/ — no mock-ups.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EX = REPO / "examples"
OUT = REPO / "examples" / "openjutsu_demo_reel.mp4"
TMP = Path(tempfile.mkdtemp())

W, H = 1280, 720
FPS = 24
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = "0xF2F2F2"
ACCENT = "0xFFD447"
DIM = "0x9AA0A6"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _text_file(text: str) -> str:
    """Write a caption to a file and return a drawtext textfile= argument.

    Passing text inline breaks on apostrophes and colons; a file avoids all
    of ffmpeg's drawtext escaping rules.
    """
    if not text:
        return ""
    path = TMP / f"caption_{abs(hash(text))}.txt"
    path.write_text(text)
    return f"textfile={path}"


def _fit_font_size(text: str, max_width: int, start: int, min_size: int = 22) -> int:
    """Largest font size at which `text` fits inside `max_width`.

    ffmpeg's drawtext has no auto-fit, so long headlines run off the edge.
    Measure with PIL and step the size down until it fits.
    """
    if not text:
        return start
    try:
        from PIL import ImageFont
    except ImportError:  # pragma: no cover - pillow is a hard requirement
        return start
    for size in range(start, min_size - 1, -2):
        font = ImageFont.truetype(FONT_B, size)
        width = font.getbbox(text)[2]
        if width <= max_width:
            return size
    return min_size


def title_card(headline: str, sub: str, seconds: float, out: Path) -> None:
    """A dark card with a headline and a subline."""
    head = _text_file(headline)
    subf = _text_file(sub)
    h_size = _fit_font_size(headline, W - 160, 64)
    s_size = _fit_font_size(sub, W - 200, 30, 18) if sub else 30
    draw = (
        f"drawbox=x=0:y=0:w={W}:h={H}:color=0x101014:t=fill,"
        f"drawtext=fontfile={FONT_B}:{head}:fontcolor={INK}:"
        f"fontsize={h_size}:x=(w-text_w)/2:y=(h-text_h)/2-70,"
        f"drawtext=fontfile={FONT}:{subf}:fontcolor={ACCENT}:"
        f"fontsize={s_size}:x=(w-text_w)/2:y=(h-text_h)/2+30"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", f"color=c=0x101014:s={W}x{H}:d={seconds}:r={FPS}",
        "-vf", draw, "-t", str(seconds),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-s", f"{W}x{H}", "-video_track_timescale", "24000",
        str(out),
    ])


def clip_card(video: Path, caption: str, seconds: float, out: Path,
              sub: str = "") -> None:
    """Show a clip (letterboxed) with a caption bar underneath."""
    esc = _text_file(caption)
    esc2 = _text_file(sub)
    c_size = _fit_font_size(caption, W - 80, 34, 20)
    s_size = _fit_font_size(sub, W - 80, 24, 16) if sub else 24
    vf = (
        f"scale={W}:{H-90}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x101014,"
        f"drawbox=x=0:y={H-90}:w={W}:h=90:color=0x101014:t=fill,"
        f"drawtext=fontfile={FONT_B}:{esc}:fontcolor={ACCENT}:"
        f"fontsize={c_size}:x=40:y={H-72},"
        f"drawtext=fontfile={FONT}:{esc2}:fontcolor={DIM}:"
        f"fontsize={s_size}:x=40:y={H-34}"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(video),
        "-t", str(seconds), "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-s", f"{W}x{H}", "-video_track_timescale", "24000",
        str(out),
    ])


def still_card(image: Path, caption: str, seconds: float, out: Path,
               sub: str = "") -> None:
    esc = _text_file(caption)
    esc2 = _text_file(sub)
    c_size = _fit_font_size(caption, W - 80, 34, 20)
    s_size = _fit_font_size(sub, W - 80, 24, 16) if sub else 24
    vf = (
        f"scale={W-160}:{H-110}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x101014,"
        f"drawbox=x=0:y={H-90}:w={W}:h=90:color=0x101014:t=fill,"
        f"drawtext=fontfile={FONT_B}:{esc}:fontcolor={ACCENT}:"
        f"fontsize={c_size}:x=40:y={H-72},"
        f"drawtext=fontfile={FONT}:{esc2}:fontcolor={DIM}:"
        f"fontsize={s_size}:x=40:y={H-34}"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(image),
        "-t", str(seconds), "-vf", vf,
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-s", f"{W}x{H}", "-video_track_timescale", "24000",
        str(out),
    ])


def concat(parts: list[Path], out: Path) -> None:
    listing = TMP / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", str(out),
    ])


def main() -> None:
    parts: list[Path] = []

    def add(fn, *args, **kwargs) -> None:
        p = TMP / f"part_{len(parts):02d}.mp4"
        fn(*args, out=p, **kwargs)
        parts.append(p)

    # 1. Title
    add(title_card, "Openjutsu",
        "Open-source Higgsfield Genjutsu - video motion transfer, local and open weights",
        7.0)

    # 2. What it is
    add(title_card,
        "Recast a video. Keep the motion.",
        "The source clip keeps its motion, camera and timing; the cast comes from your references",
        7.0)

    # 3. Motion Transfer: source, reference, result
    add(clip_card, EX / "dancer_field_pose.mp4",
        "Source clip", 5.0, sub="ComfyUI's own public example asset - motion, camera, timing and audio")
    add(still_card, EX / "dancer_ref.png",
        "Reference", 4.0, sub="who should be in the shot")
    add(clip_card, EX / "openjutsu_demo.mp4",
        "Motion Transfer", 8.0,
        sub="same motion, new cast - rendered locally on one RTX 5080")

    # 4. Object Swap
    add(title_card, "Object Swap",
        "Change one element. The rest of the shot survives.", 6.0)
    add(still_card, EX / "ref_samurai.png",
        "New element", 4.0, sub="the samurai - a single reference image")
    add(clip_card, EX / "swap_mask_track.mp4",
        "Tracked mask", 4.0,
        sub="built per frame by temporal-median background subtraction")
    add(clip_card, EX / "openjutsu_swap_demo.mp4",
        "Object Swap result", 9.0,
        sub="the dancer is replaced; the meadow is untouched")

    # 5. The evidence
    add(title_card, "The numbers",
        "Region-locked to the source. Audio bit-exact. Measured, not claimed.", 7.0)

    # 6. How it is built
    add(title_card,
        "Built on the Comfy Developer Platform",
        "comfy-sdk -> comfy-api-proxy -> local ComfyUI daemon", 7.0)
    add(title_card,
        "MiniMax H3 + H3 Fun ControlNet-Union",
        "open weights, no partner nodes, no closed-source APIs - runs on your own GPU", 7.0)

    # 7. Call to action
    add(title_card,
        "github.com/Ex8-ca/openjutsu",
        "MIT licensed - setup.sh walks through models, deps and launch", 8.0)
    add(title_card,
        "#OpenSourceHiggsfield",
        "Open Source Higgsfield competition - submission deadline Oct 30", 7.0)

    concat(parts, OUT)
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(OUT)],
        capture_output=True, text=True).stdout.strip()
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {dur}s)")


if __name__ == "__main__":
    main()
