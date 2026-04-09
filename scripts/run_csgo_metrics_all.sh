#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${PROJECT_ROOT}/scripts/run_eval_base.sh"
bash "${PROJECT_ROOT}/scripts/run_eval_stage1_epoch2.sh"

if [[ -f "${PROJECT_ROOT}/configs/eval_trd_v1.yaml" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_eval_trd_v1.sh"
fi
