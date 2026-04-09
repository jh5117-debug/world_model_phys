#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
CONFIG_PATH=""
GPU_LIST="${GPU_LIST:-}"
NUM_GPUS="${NUM_GPUS:-}"
ULYSSES_SIZE="${ULYSSES_SIZE:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --env_file)
      ENV_FILE="$2"
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
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 --config <yaml> [extra run_csgo_metrics args...]" >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${GPU_LIST}" ]]; then
  GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
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

python -m physical_consistency.cli.run_csgo_metrics \
  --config "${CONFIG_PATH}" \
  --env_file "${ENV_FILE}" \
  --num_gpus "${NUM_GPUS}" \
  --ulysses_size "${ULYSSES_SIZE}" \
  "${EXTRA_ARGS[@]}"
