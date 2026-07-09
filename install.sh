#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is required. Install it: https://docs.astral.sh/uv/" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "warning: 'ffmpeg' not found (needed for audio/video transcription)." >&2
fi

echo "==> Creating isolated Python 3.12 environment (.venv)"
uv venv --python 3.12

echo "==> Installing CPU-only PyTorch (no CUDA/ROCm pulled in)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing Sift with all features"
uv pip install -e ".[semantic,image,asr,ocr,gui,kde]"

echo "==> Writing default config (if missing)"
.venv/bin/sift init

echo "==> Pre-downloading models for enabled features (semantic, image, audio/video)"
echo "    This can take a while and several GB. Ctrl-C to skip; they would"
echo "    otherwise download on first use."
.venv/bin/sift download-models || echo "warning: model pre-download incomplete; will fetch on first use."

cat <<'EOF'

Done. Sift is installed in ./.venv and a default config was written.

Next steps:
  edit ~/.config/sift/config.toml to add folders, then:
  .venv/bin/sift reindex          build the index
  .venv/bin/sift install-kde      KRunner global bar + Dolphin right-click
  .venv/bin/sift install-service  daily auto-reindex (systemd user timer)
  .venv/bin/sift bench            check throughput and desktop impact

Default profile is "heavy" (text + semantic + image + audio/video + OCR),
CPU-only. Change the profile in the config or from the GUI settings.
EOF
