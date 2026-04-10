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

mkdir -p "${PROJECT_ROOT}/links" "${PROJECT_ROOT}/logs"
LOG_FILE="${PROJECT_ROOT}/logs/setup_symlinks.log"

link_one() {
  local target="$1"
  local source_path="$2"
  local target_real source_real current_real
  if [[ -z "${source_path}" ]]; then
    echo "[SKIP] empty source for ${target}" | tee -a "${LOG_FILE}"
    return 0
  fi
  target_real="$(realpath -m "${target}")"
  source_real="$(realpath -m "${source_path}")"
  if [[ "${target_real}" == "${source_real}" ]]; then
    echo "[SKIP] ${target} already points to itself" | tee -a "${LOG_FILE}"
    return 0
  fi
  if [[ ! -e "${source_path}" ]]; then
    echo "[WARN] source missing: ${source_path}" | tee -a "${LOG_FILE}"
    return 0
  fi

  if [[ -L "${target}" ]]; then
    current_real="$(readlink -f "${target}" || true)"
    if [[ "${current_real}" == "${source_real}" ]]; then
      echo "[SKIP] ${target} already points to ${source_path}" | tee -a "${LOG_FILE}"
      return 0
    fi
    rm -f "${target}"
    ln -s "${source_path}" "${target}"
    echo "[RELINK] ${target} -> ${source_path}" | tee -a "${LOG_FILE}"
    return 0
  fi

  if [[ -e "${target}" ]]; then
    echo "[SKIP] ${target} already exists as a non-symlink path" | tee -a "${LOG_FILE}"
    return 0
  fi

  ln -s "${source_path}" "${target}"
  echo "[LINK] ${target} -> ${source_path}" | tee -a "${LOG_FILE}"
}

link_one "${PROJECT_ROOT}/links/base_model" "${BASE_MODEL_DIR}"
link_one "${PROJECT_ROOT}/links/stage1_epoch2" "${STAGE1_CKPT_DIR}"
link_one "${PROJECT_ROOT}/links/stage1_final" "${STAGE1_FINAL_DIR}"
link_one "${PROJECT_ROOT}/links/processed_csgo_v3" "${DATASET_DIR}"
link_one "${PROJECT_ROOT}/links/raw_csgo_v3_train" "${RAW_DATA_DIR}"
link_one "${PROJECT_ROOT}/links/lingbot_code" "${LINGBOT_CODE_DIR}"
link_one "${PROJECT_ROOT}/links/finetune_code" "${FINETUNE_CODE_DIR}"
link_one "${PROJECT_ROOT}/links/teacher_ckpt" "${TEACHER_CKPT_DIR:-}"
link_one "${PROJECT_ROOT}/links/videophy2_checkpoint" "${VIDEOPHY2_CKPT_DIR:-}"
