#!/usr/bin/env bash
# Openjutsu setup — one-shot prep for a fresh machine.
#
#   ./scripts/setup.sh
#
# Idempotent. Does three things:
#   1. copies the bundled example assets into ComfyUI's input/ directory
#      (the workflow references them by bare filename)
#   2. installs the optional comfyui-vrgamedevgirl pack (source-audio
#      preservation) and applies the scipy/torchaudio patch
#   3. prints the model files that still need downloading
#
# Environment:
#   COMFYUI_ROOT   ComfyUI install root. default /home/marc/nvme-data/.comfyui
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-/home/marc/nvme-data/.comfyui}"
INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_ROOT/input}"

echo "== Openjutsu setup =="
echo "repo:        $REPO_ROOT"
echo "comfyui:     $COMFYUI_ROOT"
echo

# --- 1. example assets ---
echo "-- copying example assets into $INPUT_DIR"
mkdir -p "$INPUT_DIR"
for f in "$REPO_ROOT"/examples/*.png "$REPO_ROOT"/examples/*.mp4; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  if [ -e "$INPUT_DIR/$base" ]; then
    echo "   = $base (already present)"
  else
    cp "$f" "$INPUT_DIR/$base"
    echo "   + $base"
  fi
done
echo

# --- 2. optional custom node pack ---
PACK="$COMFYUI_ROOT/custom_nodes/comfyui-vrgamedevgirl"
if [ -d "$PACK/.git" ]; then
  echo "-- comfyui-vrgamedevgirl already installed"
else
  echo "-- installing comfyui-vrgamedevgirl (optional: source-audio preservation)"
  mkdir -p "$COMFYUI_ROOT/custom_nodes"
  git clone --depth=1 https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git "$PACK" || {
    echo "   ! clone failed — the app still works, it will use H3's own generated audio"
  }
fi
# The node lives in optional_nodes/ltx_minimax_tools/ in current upstream;
# older checkouts kept it at the pack root. Check both.
AUDIO_NODE=""
for cand in "$PACK/optional_nodes/ltx_minimax_tools/VRGDG_MiniMaxH3AudioDrive.py" \
            "$PACK/VRGDG_MiniMaxH3AudioDrive.py"; do
  [ -f "$cand" ] && { AUDIO_NODE="$cand"; break; }
done
if [ -n "$AUDIO_NODE" ] && [ -f "$REPO_ROOT/patches/vrgdg-minimax-h3-audio-drive-torchaudio.patch" ]; then
  if grep -q "Openjutsu patch" "$AUDIO_NODE"; then
    echo "   = torchaudio patch already applied"
  else
    echo "   + applying torchaudio -> scipy resample patch"
    (cd "$PACK" && git apply "$REPO_ROOT/patches/vrgdg-minimax-h3-audio-drive-torchaudio.patch") \
      && echo "     applied" \
      || echo "     ! patch did not apply (upstream changed?). The app falls back to H3's own audio."
  fi
fi
echo

# --- 3. model checklist ---
cat <<'EOF'
-- models still needed (download into ComfyUI's models/ dir)

diffusion_models/
  minimax_h3_ref2va_pruned_int8_convrot.safetensors      21 GB   (required)
model_patches/
  minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors  2.3 GB (required)
text_encoders/
  qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors             14.6 GB  (required)
vae/
  minimax_h3_video_vae_int8_convrot.safetensors            2.6 GB   (required)
  minimax_h3_audio_vae_fp32.safetensors                    605 MB   (required)

optional (turbo mode, 4 steps instead of 20 — 5x faster):
loras/
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors   1.9 GB

All from: https://huggingface.co/Comfy-Org/MiniMax-H3
EOF
echo
echo "== done. Next: start the daemon + proxy, then run web/server.py =="
