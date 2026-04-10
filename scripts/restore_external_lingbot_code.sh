#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
SNAPSHOT_ROOT="${PROJECT_ROOT}/third_party/lingbot_restore/code"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

set -a
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
set +a

if [[ -z "${LINGBOT_CODE_DIR:-}" || -z "${FINETUNE_CODE_DIR:-}" ]]; then
  echo "[ERROR] LINGBOT_CODE_DIR / FINETUNE_CODE_DIR not set. Source ${ENV_FILE} first." >&2
  exit 1
fi

restore_one() {
  local src="$1"
  local dst="$2"
  local backup=""
  if [[ ! -f "${src}" ]]; then
    echo "[ERROR] snapshot missing: ${src}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${dst}")"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    backup="${dst}.bak_${TIMESTAMP}"
    cp -a "${dst}" "${backup}"
    echo "[BACKUP] ${backup}"
  fi

  cp "${src}" "${dst}"
  echo "[RESTORE] ${dst}"
  if [[ ! -s "${dst}" ]]; then
    echo "[ERROR] restored file is empty: ${dst}" >&2
    exit 1
  fi
}

restore_one \
  "${SNAPSHOT_ROOT}/finetune_v3/lingbot-csgo-finetune/eval_batch.py" \
  "${FINETUNE_CODE_DIR}/eval_batch.py"
restore_one \
  "${SNAPSHOT_ROOT}/finetune_v3/lingbot-csgo-finetune/inference_csgo.py" \
  "${FINETUNE_CODE_DIR}/inference_csgo.py"
restore_one \
  "${SNAPSHOT_ROOT}/lingbot-world/wan/image2video.py" \
  "${LINGBOT_CODE_DIR}/wan/image2video.py"
restore_one \
  "${SNAPSHOT_ROOT}/lingbot-world/wan/modules/attention.py" \
  "${LINGBOT_CODE_DIR}/wan/modules/attention.py"

echo
echo "[OK] Restored LingBot runtime files:"
ls -lh \
  "${FINETUNE_CODE_DIR}/eval_batch.py" \
  "${FINETUNE_CODE_DIR}/inference_csgo.py" \
  "${LINGBOT_CODE_DIR}/wan/image2video.py" \
  "${LINGBOT_CODE_DIR}/wan/modules/attention.py"
