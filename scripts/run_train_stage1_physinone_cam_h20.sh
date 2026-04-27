#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/train_stage1_physinone_cam_75f_384_bf16_trainonly.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/configs/accelerate_stage1_h20_bf16_mixed_safe.yaml}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-}"
BRANCH_MODE="${BRANCH_MODE:-sequence}"
RUN_WITH_NOHUP="${RUN_WITH_NOHUP:-1}"
TAIL_IMPORTANT_LOGS="${TAIL_IMPORTANT_LOGS:-1}"
TAIL_LINES="${TAIL_LINES:-0}"
FORCE_CLEAR_GPUS_BEFORE_LAUNCH="${FORCE_CLEAR_GPUS_BEFORE_LAUNCH:-0}"
IMPORTANT_LOG_REGEX="${IMPORTANT_LOG_REGEX:-(\\[Stage1\\]|SIGFPE|Traceback|OutOfMemoryError|RuntimeError|ERROR physical_consistency|WARNING physical_consistency)}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/train_stage1_bf16_mixed_safe.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/train_stage1_bf16_mixed_safe.pid}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_file)
      ENV_FILE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --accelerate_config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    --gpu_list)
      GPU_LIST="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --branch_mode)
      BRANCH_MODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -x "${PROJECT_ROOT}/scripts/setup_symlinks.sh" ]]; then
  "${PROJECT_ROOT}/scripts/setup_symlinks.sh" >/dev/null 2>&1 || true
fi
if [[ -e "${PROJECT_ROOT}/links/lingbot_code" ]]; then
  export LINGBOT_CODE_DIR="${PROJECT_ROOT}/links/lingbot_code"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 || -z "${GPU_ARRAY[0]}" ]]; then
  echo "[ERROR] Invalid GPU list: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="${#GPU_ARRAY[@]}"
fi

force_clear_target_gpus() {
  if [[ "${FORCE_CLEAR_GPUS_BEFORE_LAUNCH}" != "1" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found; skipping GPU cleanup" >&2
    return
  fi

  local -a pids=()
  local gpu
  local pid
  for gpu in "${GPU_ARRAY[@]}"; do
    while IFS= read -r pid; do
      pid="${pid//[[:space:]]/}"
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "[GPU RESET] No existing compute processes found on GPUs ${CUDA_VISIBLE_DEVICES}" >&2
    return
  fi

  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u)
  echo "[GPU RESET] Force killing existing compute PIDs on GPUs ${CUDA_VISIBLE_DEVICES}: ${pids[*]}" >&2
  kill -9 "${pids[@]}" 2>/dev/null || true
  sleep 2
}

mkdir -p "$(dirname "${LOG_FILE}")"

# Clear any stale precision overrides from older Stage-1 debugging sessions.
unset PC_STAGE1_FORCE_FP32 || true
unset PC_STAGE1_PRECISION_PROFILE || true
unset PC_STAGE1_LOWP_DTYPE || true
unset PC_FORCE_SDPA_FALLBACK || true
unset PC_FORCE_SDPA_MATH || true
unset PC_FORCE_ATTN_FP32 || true
unset PC_VAE_FORCE_FP32 || true
unset PC_FORCE_LORA_FP32 || true
unset PC_LORA_DISABLE_AUTOCAST || true

# Re-export the intended mixed-safe bf16 policy explicitly for robustness.
export PC_STAGE1_PRECISION_PROFILE="${PC_STAGE1_PRECISION_PROFILE:-mixed_safe}"
export PC_STAGE1_LOWP_DTYPE="${PC_STAGE1_LOWP_DTYPE:-bf16}"
export PC_FORCE_SDPA_FALLBACK="${PC_FORCE_SDPA_FALLBACK:-1}"
export PC_FORCE_SDPA_MATH="${PC_FORCE_SDPA_MATH:-1}"
export PC_FORCE_ATTN_FP32="${PC_FORCE_ATTN_FP32:-1}"
export PC_VAE_FORCE_FP32="${PC_VAE_FORCE_FP32:-1}"
export PC_FORCE_LORA_FP32="${PC_FORCE_LORA_FP32:-1}"
export PC_LORA_DISABLE_AUTOCAST="${PC_LORA_DISABLE_AUTOCAST:-1}"

echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "[INFO] ENV_FILE=${ENV_FILE}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}"
echo "[INFO] BRANCH_MODE=${BRANCH_MODE}"
echo "[INFO] PC_STAGE1_PRECISION_PROFILE=${PC_STAGE1_PRECISION_PROFILE}"
echo "[INFO] PC_STAGE1_LOWP_DTYPE=${PC_STAGE1_LOWP_DTYPE}"
echo "[INFO] PC_FORCE_SDPA_FALLBACK=${PC_FORCE_SDPA_FALLBACK}"
echo "[INFO] PC_FORCE_SDPA_MATH=${PC_FORCE_SDPA_MATH}"
echo "[INFO] PC_FORCE_ATTN_FP32=${PC_FORCE_ATTN_FP32}"
echo "[INFO] PC_VAE_FORCE_FP32=${PC_VAE_FORCE_FP32}"
echo "[INFO] PC_FORCE_LORA_FP32=${PC_FORCE_LORA_FP32}"
echo "[INFO] PC_LORA_DISABLE_AUTOCAST=${PC_LORA_DISABLE_AUTOCAST}"

cd "${PROJECT_ROOT}"
force_clear_target_gpus

TRAIN_CMD=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_GPUS}"
  -m physical_consistency.cli.train_stage1_physinone_cam
  --config "${CONFIG_PATH}"
  --env_file "${ENV_FILE}"
  --branch_mode "${BRANCH_MODE}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${RUN_WITH_NOHUP}" == "1" ]]; then
  : > "${LOG_FILE}"
  if command -v python3 >/dev/null 2>&1; then
    TRAIN_PID="$(
      PROJECT_ROOT="${PROJECT_ROOT}" \
      LOG_FILE="${LOG_FILE}" \
      PID_FILE="${PID_FILE}" \
      python3 - "${TRAIN_CMD[@]}" <<'PY'
import os
import subprocess
import sys

project_root = os.environ["PROJECT_ROOT"]
log_file = os.environ["LOG_FILE"]
pid_file = os.environ["PID_FILE"]
cmd = sys.argv[1:]

with open(log_file, "ab", buffering=0) as log_stream:
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

with open(pid_file, "w", encoding="utf-8") as handle:
    handle.write(f"{proc.pid}\n")

print(proc.pid)
PY
    )"
  else
    TRAIN_CMD_STRING="$(printf '%q ' "${TRAIN_CMD[@]}")"
    if command -v setsid >/dev/null 2>&1; then
      setsid bash -lc "cd $(printf '%q' "${PROJECT_ROOT}") && exec ${TRAIN_CMD_STRING}" </dev/null >>"${LOG_FILE}" 2>&1 &
    else
      nohup bash -lc "cd $(printf '%q' "${PROJECT_ROOT}") && exec ${TRAIN_CMD_STRING}" </dev/null >>"${LOG_FILE}" 2>&1 &
    fi
    TRAIN_PID=$!
    printf '%s\n' "${TRAIN_PID}" > "${PID_FILE}"
  fi

  echo "[NOHUP] Started background training with PID ${TRAIN_PID}"
  echo "[NOHUP] Full log: ${LOG_FILE}"
  echo "[NOHUP] PID file: ${PID_FILE}"

  if [[ "${TAIL_IMPORTANT_LOGS}" != "1" ]]; then
    exit 0
  fi

  echo "[TAIL] Monitoring important log lines from ${LOG_FILE}"
  if tail --help 2>&1 | grep -q -- '--pid'; then
    tail --pid="${TRAIN_PID}" -n "${TAIL_LINES}" -F "${LOG_FILE}" \
      | stdbuf -oL -eL grep --line-buffered -E "${IMPORTANT_LOG_REGEX}"
  else
    tail -n "${TAIL_LINES}" -F "${LOG_FILE}" \
      | stdbuf -oL -eL grep --line-buffered -E "${IMPORTANT_LOG_REGEX}"
  fi
  exit 0
fi

"${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
