#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python -m physical_consistency.cli.run_csgo_metrics --config "${PROJECT_ROOT}/configs/eval_stage1_epoch2.yaml" "$@"
