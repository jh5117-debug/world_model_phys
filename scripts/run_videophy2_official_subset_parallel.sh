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

OFFICIAL_ASSET_ROOT="${OFFICIAL_ASSET_ROOT:-${PROJECT_ROOT}/data/videophy2_official}"
VIDEO_PROMPT_MODE="${VIDEO_PROMPT_MODE:-original}"
VIDEO_HARD_ONLY="${VIDEO_HARD_ONLY:-0}"

manifest_suffix="${VIDEO_PROMPT_MODE}"
if [[ "${VIDEO_HARD_ONLY}" == "1" ]]; then
  manifest_suffix="${manifest_suffix}_hard"
fi

export MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/manifests/videophy2_test_official_${manifest_suffix}_videos.csv}"
export VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${OFFICIAL_ASSET_ROOT}/videos/test_subset}"
export VIDEO_SOURCE_MODE="${VIDEO_SOURCE_MODE:-manifest_video_column}"
export MANIFEST_VIDEO_COLUMN="${MANIFEST_VIDEO_COLUMN:-videopath}"
export MANIFEST_CAPTION_COLUMN="${MANIFEST_CAPTION_COLUMN:-prompt}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_videophy2_official_subset_${manifest_suffix}}"
export GPU_LIST="${GPU_LIST:-0,1,2,3}"
export MAX_ROWS_PER_GPU="${MAX_ROWS_PER_GPU:-10}"
export SEED="${SEED:-0}"

"${PROJECT_ROOT}/scripts/run_videophy2_dataset_autoeval_parallel.sh" "$@"
