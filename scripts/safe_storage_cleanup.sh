#!/usr/bin/env bash
set -euo pipefail

# Safe disk cleanup helper for H20/PAI copies of this repo.
#
# Modes:
#   MODE=audit              Print disk usage and candidate cleanup paths. No changes.
#   MODE=quarantine         Move candidates into QUARANTINE_ROOT. Requires CONFIRM=YES.
#   MODE=purge-quarantine   Delete QUARANTINE_ROOT. Requires CONFIRM=YES.
#
# The script is deliberately conservative. It never removes anything directly in
# quarantine mode, and it refuses protected/current experiment paths.

ROOT="${ROOT:-$(pwd)}"
MODE="${MODE:-audit}"
CONFIRM="${CONFIRM:-NO}"
QUARANTINE_ROOT="${QUARANTINE_ROOT:-}"
CANDIDATE_FILE="${CANDIDATE_FILE:-}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"
INCLUDE_EVAL_OUTPUTS="${INCLUDE_EVAL_OUTPUTS:-0}"
INCLUDE_RUNS="${INCLUDE_RUNS:-0}"
MIN_FILE_SIZE_GB="${MIN_FILE_SIZE_GB:-10}"

CURRENT_STAGE2_EXP="${CURRENT_STAGE2_EXP:-exp_stage2_phycsgo_act_low_single_8gpu_81f_240x416_lora_block20_chunk512_ffn4096_ep4}"
CURRENT_DATASET_GLOB="${CURRENT_DATASET_GLOB:-PhysInOne_act_mixed}"

log() { echo "[$(date '+%F %T')] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

canonical() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

ROOT="$(canonical "${ROOT}")"
[[ -d "${ROOT}" ]] || fail "ROOT does not exist or is not a directory: ${ROOT}"
case "${ROOT}" in
  "/"|"/home"|"/mnt"|"/mnt/workspace"|"/home/nvme04"|"/home/nvme03")
    fail "Refusing unsafe broad ROOT=${ROOT}"
    ;;
esac

if [[ -z "${QUARANTINE_ROOT}" ]]; then
  QUARANTINE_ROOT="${ROOT}/_quarantine_cleanup_$(date +%Y%m%d_%H%M%S)"
fi
QUARANTINE_ROOT="$(canonical "${QUARANTINE_ROOT}")"

relpath() {
  python3 - "$ROOT" "$1" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
path = Path(sys.argv[2]).resolve()
try:
    print(path.relative_to(root))
except ValueError:
    print(path)
PY
}

is_protected() {
  local path="$1"
  local rel
  rel="$(relpath "${path}")"
  case "${rel}" in
    "."|".git"|".git/"*|src|src/"*|scripts|scripts/"*|configs|configs/"*|tests|tests/"*|third_party|third_party/"*|links|links/"*|Makefile|pyproject.toml)
      return 0
      ;;
    "checkpoints/${CURRENT_STAGE2_EXP}/epoch_3"|\
    "checkpoints/${CURRENT_STAGE2_EXP}/epoch_3/"*|\
    "checkpoints/"*"stage1"*"act"*|\
    "checkpoints/"*"physinone_act"*|\
    "checkpoints/"*"moving_act"*)
      return 0
      ;;
    "Dataset/Phy_Dataset/PhysInOne/raw"|\
    "Dataset/Phy_Dataset/PhysInOne/raw/"*|\
    "Dataset/Phy_Dataset/"*"${CURRENT_DATASET_GLOB}"*|\
    "Dataset/Phy_Dataset/"*"${CURRENT_DATASET_GLOB}"*"/"*|\
    "data/Phy_Dataset/PhysInOne/raw"|\
    "data/Phy_Dataset/PhysInOne/raw/"*|\
    "weight/Lingbot-base-act"|\
    "weight/Lingbot-base-act/"*|\
    "weight/Lingbot-base"|\
    "weight/Lingbot-base/"*)
      return 0
      ;;
    "_quarantine_cleanup_"*|"_quarantine_cleanup_"*"/"*)
      return 0
      ;;
  esac
  return 1
}

print_size() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null | awk '{print $1}'
  else
    echo "-"
  fi
}

add_candidate() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  local full
  full="$(canonical "${path}")"
  if is_protected "${full}"; then
    log "SKIP protected candidate: $(relpath "${full}")"
    return 0
  fi
  printf '%s\n' "${full}" >> "${TMP_CANDIDATES}"
}

TMP_CANDIDATES="$(mktemp)"
trap 'rm -f "${TMP_CANDIDATES}"' EXIT

collect_candidates() {
  if [[ -n "${CANDIDATE_FILE}" ]]; then
    [[ -f "${CANDIDATE_FILE}" ]] || fail "CANDIDATE_FILE not found: ${CANDIDATE_FILE}"
    while IFS= read -r raw || [[ -n "${raw}" ]]; do
      raw="${raw%%#*}"
      raw="${raw#"${raw%%[![:space:]]*}"}"
      raw="${raw%"${raw##*[![:space:]]}"}"
      [[ -z "${raw}" ]] && continue
      if [[ "${raw}" = /* ]]; then
        add_candidate "${raw}"
      else
        add_candidate "${ROOT}/${raw}"
      fi
    done < "${CANDIDATE_FILE}"
    return
  fi

  # High-confidence old/debug artifacts.
  add_candidate "${ROOT}/archive_broken_runs_20260417_100620"
  add_candidate "${ROOT}/.venv_flux2_eval"
  add_candidate "${ROOT}/.venv_flux2_eval_cu128"
  add_candidate "${ROOT}/train_exp_stage1_epoch2_trd_v1_high_81f_480x832_4gpu_lora_fp32_autocast_real_high"

  # Optional broad artifact classes. Disabled by default.
  if [[ "${INCLUDE_CHECKPOINTS}" == "1" && -d "${ROOT}/checkpoints" ]]; then
    while IFS= read -r path; do
      add_candidate "${path}"
    done < <(find "${ROOT}/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  if [[ "${INCLUDE_EVAL_OUTPUTS}" == "1" && -d "${ROOT}/eval_outputs" ]]; then
    while IFS= read -r path; do
      add_candidate "${path}"
    done < <(find "${ROOT}/eval_outputs" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  if [[ "${INCLUDE_RUNS}" == "1" && -d "${ROOT}/runs" ]]; then
    while IFS= read -r path; do
      add_candidate "${path}"
    done < <(find "${ROOT}/runs" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
}

audit() {
  log "ROOT=${ROOT}"
  log "MODE=${MODE}"
  log "QUARANTINE_ROOT=${QUARANTINE_ROOT}"

  echo
  echo "========== Top-level disk usage =========="
  du -xhd1 "${ROOT}" 2>/dev/null | sort -h | tail -60 || true

  echo
  echo "========== Large files >= ${MIN_FILE_SIZE_GB}GiB =========="
  find "${ROOT}" -xdev -type f -size +"${MIN_FILE_SIZE_GB}"G \
    -printf '%s\t%p\n' 2>/dev/null \
    | sort -nr \
    | head -120 \
    | awk '{size=$1; $1=""; printf "%.1fGiB\t%s\n", size/1024/1024/1024, $0}' || true

  echo
  echo "========== Current protected targets =========="
  for path in \
    "${ROOT}/checkpoints/${CURRENT_STAGE2_EXP}/epoch_3" \
    "${ROOT}/Dataset/Phy_Dataset/PhysInOne/raw" \
    "${ROOT}/Dataset/Phy_Dataset" \
    "${ROOT}/weight/Lingbot-base-act" \
    "${ROOT}/src" \
    "${ROOT}/scripts"; do
    [[ -e "${path}" ]] && printf '%8s  %s\n' "$(print_size "${path}")" "$(relpath "${path}")"
  done
}

collect_candidates
sort -u "${TMP_CANDIDATES}" -o "${TMP_CANDIDATES}"

audit

echo
echo "========== Cleanup candidates =========="
if [[ ! -s "${TMP_CANDIDATES}" ]]; then
  echo "(none)"
else
  while IFS= read -r path; do
    printf '%8s  %s\n' "$(print_size "${path}")" "$(relpath "${path}")"
  done < "${TMP_CANDIDATES}"
fi

case "${MODE}" in
  audit)
    echo
    log "Audit only. No files were changed."
    ;;
  quarantine)
    [[ "${CONFIRM}" == "YES" ]] || fail "quarantine mode requires CONFIRM=YES"
    [[ -s "${TMP_CANDIDATES}" ]] || fail "No candidates to quarantine"
    mkdir -p "${QUARANTINE_ROOT}"
    while IFS= read -r path; do
      if is_protected "${path}"; then
        log "SKIP protected at move time: $(relpath "${path}")"
        continue
      fi
      rel="$(relpath "${path}")"
      dest="${QUARANTINE_ROOT}/${rel}"
      mkdir -p "$(dirname "${dest}")"
      log "MOVE $(print_size "${path}") ${rel} -> ${dest}"
      mv "${path}" "${dest}"
    done < "${TMP_CANDIDATES}"
    log "Quarantine complete: ${QUARANTINE_ROOT}"
    ;;
  purge-quarantine)
    [[ "${CONFIRM}" == "YES" ]] || fail "purge-quarantine mode requires CONFIRM=YES"
    case "${QUARANTINE_ROOT}" in
      *"_quarantine_cleanup_"*) ;;
      *) fail "Refusing to purge non-quarantine path: ${QUARANTINE_ROOT}" ;;
    esac
    [[ -d "${QUARANTINE_ROOT}" ]] || fail "QUARANTINE_ROOT not found: ${QUARANTINE_ROOT}"
    log "PURGE $(print_size "${QUARANTINE_ROOT}") ${QUARANTINE_ROOT}"
    rm -rf "${QUARANTINE_ROOT}"
    log "Purge complete"
    ;;
  *)
    fail "Unsupported MODE=${MODE}; expected audit, quarantine, purge-quarantine"
    ;;
esac
