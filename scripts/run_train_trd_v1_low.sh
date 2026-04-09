#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme02/lingbot-world/output/physical_consistency}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/configs/accelerate_trd_v1.yaml}"
mkdir -p "${OUTPUT_ROOT}/logs"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  -m physical_consistency.cli.train_trd_v1 \
  --config "${PROJECT_ROOT}/configs/train_trd_v1.yaml" \
  --model_type low \
  "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/train_trd_v1_low.log"
