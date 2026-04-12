#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_SUBDIR="${MODEL_SUBDIR:-lingbotstage1}"
TEST_INF_ROOT="${TEST_INF_ROOT:-${DATASET_DIR}/test_inf_result/${MODEL_SUBDIR}}"

export MANIFEST="${MANIFEST:-${TEST_INF_ROOT}/generated_videos.csv}"
export VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${TEST_INF_ROOT}}"
export VIDEO_SOURCE_MODE="${VIDEO_SOURCE_MODE:-manifest_video_column}"
export MANIFEST_VIDEO_COLUMN="${MANIFEST_VIDEO_COLUMN:-candidate_videopath}"
export MANIFEST_CAPTION_COLUMN="${MANIFEST_CAPTION_COLUMN:-prompt}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_videophy2_test_inf_${MODEL_SUBDIR}}"
export GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export MAX_ROWS_PER_GPU="${MAX_ROWS_PER_GPU:-10}"
export SEED="${SEED:-0}"

"${PROJECT_ROOT}/scripts/run_videophy2_dataset_autoeval_parallel.sh" "$@"
