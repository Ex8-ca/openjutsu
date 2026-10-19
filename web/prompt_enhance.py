"""Heuristic prompt enhancer for Openjutsu — boosts a short user prompt into
a richer cinematic description that H3 can latch onto. No model
required; deterministic. Real Qwen3-VL-driven enhancement is deferred
(see docs/architecture.md / plan).
"""
from __future__ import annotations


def enhance_prompt(prompt: str) -> str:
    """Wrap the user's prompt with cinematic + identity + motion cues.

    - Prepends a quality/style anchor (photorealistic, cinematic)
    - Appends a motion-preservation cue (subject identity, motion, timing)
    - Falls back to a neutral scene if the prompt is empty
    """
    base = (prompt or "").strip()
    if not base:
        base = "A natural-looking scene"
    prefix = "Cinematic, photorealistic, "
    suffix = (
        ". Subject identity, motion, camera movement, and timing are all "
        "preserved from the source video."
    )
    return f"{prefix}{base}{suffix}"