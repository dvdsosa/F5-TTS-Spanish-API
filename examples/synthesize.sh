#!/usr/bin/env bash
# Example: synthesize Spanish speech from a reference voice via the local API.
#
# Prereqs: the API must be running (./scripts/serve.sh).
# Usage:   ./examples/synthesize.sh "Hola, este es un ejemplo de voz en castellano." ref.wav out.wav
set -euo pipefail

TEXT="${1:-Hola, este es un ejemplo de voz en castellano.}"
REF="${2:-ref.wav}"
OUT="${3:-out.wav}"
PORT="${PORT:-8000}"

curl -s -X POST "http://127.0.0.1:${PORT}/v1/audio/speech" \
  -F "ref_audio=@${REF}" \
  -F "text=${TEXT}" \
  -o "${OUT}"

echo ">> Saved to ${OUT}"
