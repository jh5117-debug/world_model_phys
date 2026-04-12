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

export MANIFEST="${MANIFEST:-${DATASET_DIR}/metadata_test.csv}"
export VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${DATASET_DIR}}"
export VIDEO_SOURCE_MODE="${VIDEO_SOURCE_MODE:-dataset_clip}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_dataset_test_autoeval_parallel}"
export GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export MAX_ROWS_PER_GPU="${MAX_ROWS_PER_GPU:-10}"
export SEED="${SEED:-0}"

"${PROJECT_ROOT}/scripts/run_videophy2_dataset_autoeval_parallel.sh" "$@"
