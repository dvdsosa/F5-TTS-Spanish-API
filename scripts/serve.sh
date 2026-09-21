#!/usr/bin/env bash
# Start the Spanish F5-TTS model as a local voice-cloning API.
#
# Usage:
#   ./scripts/serve.sh [--port 8000] [--device mps]
#
# The model checkpoint is downloaded on first run (if not already present)
# into ./models/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${1:-8000}"
DEVICE="${2:-}"

MODEL_DIR="$ROOT/models"
CKPT="$MODEL_DIR/model_1200000.safetensors"
VOCAB="$MODEL_DIR/vocab.txt"

# --- Download the Spanish checkpoint + vocab on first run -------------------
if [[ ! -f "$CKPT" || ! -f "$VOCAB" ]]; then
  echo ">> Downloading the Spanish F5-TTS model into $MODEL_DIR ..."
  mkdir -p "$MODEL_DIR"
  uv run python - <<'PY'
from huggingface_hub import hf_hub_download
import os
out = os.path.join("models", "model_1200000.safetensors")
if not os.path.exists(out):
    p = hf_hub_download("jpgallegoar/F5-Spanish", "model_1200000.safetensors")
    os.replace(p, out)
    print(">> checkpoint saved to", out)
vocab = os.path.join("models", "vocab.txt")
if not os.path.exists(vocab):
    p = hf_hub_download("jpgallegoar/F5-Spanish", "vocab.txt")
    os.replace(p, vocab)
    print(">> vocab saved to", vocab)
PY
fi

echo ">> Starting API on port $PORT (device: ${DEVICE:-auto}) ..."
exec uv run f5-tts-spanish-api \
  --ckpt "$CKPT" \
  --vocab "$VOCAB" \
  --port "$PORT" \
  ${DEVICE:+--device "$DEVICE"}
