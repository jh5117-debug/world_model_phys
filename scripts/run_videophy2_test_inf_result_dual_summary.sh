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

for model_subdir in lingbotbase lingbotstage1; do
  VIDEOPHY2_QUIET=1 \
  VIDEOPHY2_SUMMARY_STDOUT=0 \
  MODEL_SUBDIR="${model_subdir}" \
  "${PROJECT_ROOT}/scripts/run_videophy2_test_inf_result_parallel.sh" "$@"
done

python - "${PROJECT_ROOT}" <<'PY'
import sys
from pathlib import Path

from physical_consistency.common.io import read_json
from physical_consistency.common.summary_tables import format_videophy2_summary

project_root = Path(sys.argv[1])
summary_root = project_root / "runs" / "eval" / "videophy2"

tables = []
for model_subdir in ("lingbotbase", "lingbotstage1"):
    summary_path = summary_root / f"exp_videophy2_test_inf_{model_subdir}" / "summary.json"
    summary = read_json(summary_path)
    tables.append(format_videophy2_summary(summary, title=model_subdir))

print("\n\n".join(tables))
PY
