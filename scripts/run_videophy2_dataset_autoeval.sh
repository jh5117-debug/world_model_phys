#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib_videophy2_env.sh"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/videophy2_dataset_autoeval.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_dataset_val_autoeval}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/metadata_val.csv}"
VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${DATASET_DIR}}"
VIDEO_SOURCE_MODE="${VIDEO_SOURCE_MODE:-dataset_clip}"
MANIFEST_VIDEO_COLUMN="${MANIFEST_VIDEO_COLUMN:-videopath}"
MANIFEST_CAPTION_COLUMN="${MANIFEST_CAPTION_COLUMN:-prompt}"
export VIDEOPHY_TORCH_DTYPE="${VIDEOPHY_TORCH_DTYPE:-fp32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEOPHY_REPO_DIR="${VIDEOPHY_REPO_DIR:-${PROJECT_ROOT}/third_party/videophy}"
VIDEOPHY2_CKPT_DIR="${VIDEOPHY2_CKPT_DIR:-${PROJECT_ROOT}/links/videophy2_checkpoint}"
GPU_ID="${GPU_ID:-4}"
VIDEO_FILENAME="${VIDEO_FILENAME:-video.mp4}"
VIDEOPHY2_PYTHON="${VIDEOPHY2_PYTHON:-$(resolve_videophy2_python "${PROJECT_ROOT}")}"

if ! verify_videophy2_python "${VIDEOPHY2_PYTHON}" "${PROJECT_ROOT}"; then
  echo "[ERROR] VideoPhy-2 python preflight failed: ${VIDEOPHY2_PYTHON}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${VIDEOPHY2_PYTHON}" -m physical_consistency.cli.run_videophy2 \
  --config "${CONFIG_PATH}" \
  --env_file "${ENV_FILE}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --manifest_csv "${MANIFEST}" \
  --video_source_mode "${VIDEO_SOURCE_MODE}" \
  --video_source_root "${VIDEO_SOURCE_ROOT}" \
  --manifest_video_column "${MANIFEST_VIDEO_COLUMN}" \
  --manifest_caption_column "${MANIFEST_CAPTION_COLUMN}" \
  --video_filename "${VIDEO_FILENAME}" \
  --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
  --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  "$@"
