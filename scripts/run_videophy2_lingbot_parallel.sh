#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/videophy2_eval.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/manifests/csgo_phys_val50.csv}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/links/processed_csgo_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEOPHY_REPO_DIR="${VIDEOPHY_REPO_DIR:-${PROJECT_ROOT}/third_party/videophy}"
VIDEOPHY2_CKPT_DIR="${VIDEOPHY2_CKPT_DIR:-${PROJECT_ROOT}/links/videophy2_checkpoint}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
BASE_GENERATED_ROOT="${BASE_GENERATED_ROOT:-${OUTPUT_ROOT}/runs/eval/exp_base_zeroshot}"
STAGE1_GENERATED_ROOT="${STAGE1_GENERATED_ROOT:-${OUTPUT_ROOT}/runs/eval/exp_stage1_epoch2}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[INFO] ${MANIFEST} missing, building fixed validation manifests..."
  python "${PROJECT_ROOT}/scripts/build_fixed_val_sets.py" --output_dir "${PROJECT_ROOT}/data/manifests"
fi

if [[ ! -f "${VIDEOPHY_REPO_DIR}/VIDEOPHY2/inference.py" ]]; then
  echo "[ERROR] VideoPhy-2 repo not found at ${VIDEOPHY_REPO_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VIDEOPHY2_CKPT_DIR}" ]]; then
  echo "[ERROR] VideoPhy-2 checkpoint dir not found at ${VIDEOPHY2_CKPT_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

TASKS=(
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|42"
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|123"
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|3407"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|42"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|123"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|3407"
)

run_one_seed() {
  local gpu="$1"
  local experiment_name="$2"
  local generated_root="$3"
  local seed="$4"
  local generated_dir="${generated_root}/seed_${seed}/csgo_metrics/videos"

  if [[ ! -d "${generated_dir}" ]]; then
    echo "[ERROR][GPU ${gpu}] generated videos missing: ${generated_dir}" >&2
    return 1
  fi

  echo "[RUN][GPU ${gpu}] ${experiment_name} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${experiment_name}" \
    --manifest_csv "${MANIFEST}" \
    --generated_root "${generated_root}" \
    --seed "${seed}" \
    --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
    --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
    --output_root "${OUTPUT_ROOT}"
}

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local stride="$3"
  local task=""
  local experiment_name=""
  local generated_root=""
  local seed=""

  for ((idx=worker_idx; idx<${#TASKS[@]}; idx+=stride)); do
    task="${TASKS[$idx]}"
    IFS='|' read -r experiment_name generated_root seed <<< "${task}"
    run_one_seed "${gpu}" "${experiment_name}" "${generated_root}" "${seed}"
  done
}

for worker_idx in "${!GPUS[@]}"; do
  worker "${GPUS[$worker_idx]}" "${worker_idx}" "${#GPUS[@]}" &
done
wait

for experiment_name in exp_base_zeroshot exp_stage1_epoch2; do
  python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${experiment_name}" \
    --summary_only \
    --output_root "${OUTPUT_ROOT}"
done

echo "[DONE] VideoPhy-2 summaries written under ${OUTPUT_ROOT}/runs/eval/videophy2"
