# Higgsfield Genjutsu — observed product contract

Reference notes gathered from Higgsfield's own public Genjutsu product page and
the ComfyUI partner node that wraps it. Used to check Openjutsu's feature
parity. Nothing here is copied code — it is the product's stated behaviour and
its public request shape, which we are reimplementing in the open.

## Two modes, not one

Genjutsu ships **two** modes, and the second one is the interesting one.

> **Motion Transfer** keeps the motion, camera and timing while rebuilding the
> scene from your references.
>
> **Object Swap** changes a selected character, outfit, product, location or
> object while **preserving the rest of the original shot**.

Openjutsu implements **both** modes. Motion Transfer is the default; Object
Swap is the second tab in the web UI.

## What Object Swap means for us

The H3 Fun ControlNet-Union we already ship has a second path we do not use:
`MiniMaxH3FunControlNetApply` accepts `mask` and `source_video` alongside
`control_video`. Per the ComfyUI docs:

> Video inpainting: connect a mask to the `mask` input: regions marked `1` are
> regenerated under control of the optional `source_video` input, while the
> rest of the frame stays fixed.

That is Object Swap, in the same weights we already have, with no new model
downloads.

### The controlnet mask is a hint, not a composite

ComfyUI's implementation computes `visibility = 1.0 - (mask > 0.5)` and
concatenates `[visibility_latent, masked_latent]` onto the controlnet hint.
The whole frame is regenerated and merely *steered* toward the source, so
every pixel drifts a little, not just the masked region.

Measured on a real render (864x480, 122 frames, mean absolute difference
against the source):

| | inside mask | outside mask | selectivity |
|---|---|---|---|
| raw H3 output | 22.43 | 17.72 | 1.27x |

1.27x is not spatial selectivity — the swap is not actually swapping. So
`web/composite.py` composites in post: the source frame is kept everywhere
outside the mask and the regenerated frame is used only inside it.

| | inside mask | outside mask | selectivity |
|---|---|---|---|
| after composite | 22.64 | **2.49** | **9.10x** |

The residual 2.49 outside the mask is H.264 re-encode loss, not leakage:
re-encoding the *source* through the same encoder gives 2.41. The source's
audio is copied with `-c:a copy`, so it stays bit-exact (measured 0.00 dB
RMS delta, correlation 1.0000).

That is what makes "the rest of the shot survives" literally true rather than
approximately true.

ComfyUI 0.39.0 also ships `SAM3_VideoTrack` and `SAM3_TrackToMask` in core,
which can produce that mask from a text prompt across a clip (the SAM3 weights
need a community pack).

## Other parity notes

| Aspect | Higgsfield | Openjutsu | Note |
|---|---|---|---|
| Reference video length | 4–30 s | 4–30 s | parity |
| Reference images | up to 30 (product page) | up to 9 (H3 `ref_image` slots) | H3's ceiling, not ours |
| Reference images (ComfyUI node) | 1–8 | 9 | the node is the narrower path |
| Resolution | 480p / 720p / 1080p | 0.4 MP (864×480) + 0.13 MP preset | we are fixed by 16 GB VRAM |
| Audio | kept from the source | kept from the source | parity, and ours is byte-exact |
| Prompt | optional, ≤10 000 chars | optional | parity |

The product's own feature framing is worth mirroring in the demo: "Keep only
the motion" (Motion Transfer) versus changing a *selected* element while the
rest of the shot survives (Object Swap).

## The public request shape

For reference, the ComfyUI partner node's request to the Higgsfield API is
small and explicit:

```
POST /proxy/higgsfield/higgsfield/genjutsu/motion-transfer/v1.0
  prompt      : str
  video_url   : str
  image_urls  : list[str]
  resolution  : str            # 480p | 720p | 1080p
```

and it polls `GET /proxy/higgsfield/requests/{request_id}/status` for
`queued` → `completed` | `failed` | `nsfw` | `canceled`.

We do not call this API and do not need to: the point of the contest entry is
that Openjutsu runs the same job locally on open weights. The shape is
recorded here only to show the product's contract is small and that our
feature set is aimed at the same job.

## Cost note (context, not a target)

The partner node's published price badge is USD per output second:
480p ≈ $0.45, 720p ≈ $0.97, 1080p ≈ $2.33. A local build on one RTX 5080
has no per-second cost, which is the open-source argument in one line for the
demo video.
