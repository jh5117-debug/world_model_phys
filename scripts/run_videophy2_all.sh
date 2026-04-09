#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
set -a
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
set +a

MANIFEST="${PROJECT_ROOT}/data/manifests/csgo_phys_val50.csv"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
CONFIG_PATH="${PROJECT_ROOT}/configs/videophy2_eval.yaml"
EXPERIMENT_NAME=""
GENERATED_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --generated_root)
      GENERATED_ROOT="$2"
      shift 2
      ;;
    --manifest_csv)
      MANIFEST="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_one() {
  local experiment_name="$1"
  local generated_root="$2"
  python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${experiment_name}" \
    --manifest_csv "${MANIFEST}" \
    --generated_root "${generated_root}"
}

if [[ -n "${EXPERIMENT_NAME}" ]]; then
  if [[ -z "${GENERATED_ROOT}" ]]; then
    GENERATED_ROOT="${OUTPUT_ROOT}/runs/eval/${EXPERIMENT_NAME}"
  fi
  run_one "${EXPERIMENT_NAME}" "${GENERATED_ROOT}"
  exit 0
fi

run_one "exp_base_zeroshot" "${OUTPUT_ROOT}/runs/eval/exp_base_zeroshot"

run_one "exp_stage1_epoch2" "${OUTPUT_ROOT}/runs/eval/exp_stage1_epoch2"

if [[ -d "${OUTPUT_ROOT}/runs/eval/exp_stage1_epoch2_trd_v1" ]]; then
  run_one "exp_stage1_epoch2_trd_v1" "${OUTPUT_ROOT}/runs/eval/exp_stage1_epoch2_trd_v1"
fi
