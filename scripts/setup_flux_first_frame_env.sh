#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install -U \
  diffusers \
  transformers \
  accelerate \
  huggingface_hub \
  sentencepiece \
  safetensors \
  peft \
  pillow

echo "[DONE] FLUX first-frame dependencies installed for ${PYTHON_BIN}"
