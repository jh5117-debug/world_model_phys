#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
bash "${PROJECT_ROOT}/scripts/run_train_trd_v1.sh" \
  --env_file "${ENV_FILE}" \
  --config "${PROJECT_ROOT}/configs/train_trd_v1.yaml" \
  --model_type high \
  "$@"
