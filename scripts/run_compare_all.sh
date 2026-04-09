#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
set -a
source "${PROJECT_ROOT}/configs/path_config_cluster.env"
set +a

python -m physical_consistency.cli.compare_reports \
  --eval_root "${OUTPUT_ROOT}/runs/eval" \
  --output_json "${OUTPUT_ROOT}/runs/reports/compare_all.json"
