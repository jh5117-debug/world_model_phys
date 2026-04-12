#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
CONFIG_PATH="${PROJECT_ROOT}/configs/train_trd_v1.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/configs/accelerate_trd_v1.yaml}"
GPU_LIST="${GPU_LIST:-}"
NUM_GPUS="${NUM_GPUS:-}"
ULYSSES_SIZE="${ULYSSES_SIZE:-}"
MODEL_TYPE="${MODEL_TYPE:-dual}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_file)
      ENV_FILE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --accelerate_config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    --gpu_list)
      GPU_LIST="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --ulysses_size)
      ULYSSES_SIZE="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${GPU_LIST}" ]]; then
  GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 || -z "${GPU_ARRAY[0]}" ]]; then
  echo "[ERROR] Invalid GPU list: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="${#GPU_ARRAY[@]}"
fi
if [[ -z "${ULYSSES_SIZE}" ]]; then
  ULYSSES_SIZE="${NUM_GPUS}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
mkdir -p "${OUTPUT_ROOT}/logs"

cd "${PROJECT_ROOT}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  -m physical_consistency.cli.train_trd_v1 \
  --config "${CONFIG_PATH}" \
  --env_file "${ENV_FILE}" \
  --model_type "${MODEL_TYPE}" \
  --num_gpus "${NUM_GPUS}" \
  --ulysses_size "${ULYSSES_SIZE}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/train_trd_v1_${MODEL_TYPE}.log"
