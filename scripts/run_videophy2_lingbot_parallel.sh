#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/lib_videophy2_env.sh"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/videophy2_eval.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/manifests/csgo_phys_val50.csv}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/links/processed_csgo_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEOPHY_REPO_DIR="${VIDEOPHY_REPO_DIR:-${PROJECT_ROOT}/third_party/videophy}"
VIDEOPHY2_CKPT_DIR="${VIDEOPHY2_CKPT_DIR:-${PROJECT_ROOT}/links/videophy2_checkpoint}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
BASE_GENERATED_ROOT="${BASE_GENERATED_ROOT:-${OUTPUT_ROOT}/runs/eval/exp_base_zeroshot}"
STAGE1_GENERATED_ROOT="${STAGE1_GENERATED_ROOT:-${OUTPUT_ROOT}/runs/eval/exp_stage1_epoch2}"
STATUS_EVERY_SEC="${STATUS_EVERY_SEC:-30}"
KILL_EXISTING_GPU_PIDS="${KILL_EXISTING_GPU_PIDS:-0}"
KILL_GRACE_SEC="${KILL_GRACE_SEC:-5}"
VIDEOPHY2_PYTHON="${VIDEOPHY2_PYTHON:-$(resolve_videophy2_python "${PROJECT_ROOT}")}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[INFO] ${MANIFEST} missing, building fixed validation manifests..."
  python "${PROJECT_ROOT}/scripts/build_fixed_val_sets.py" --output_dir "${PROJECT_ROOT}/data/manifests"
fi

if [[ ! -f "${VIDEOPHY_REPO_DIR}/VIDEOPHY2/inference.py" ]]; then
  echo "[ERROR] VideoPhy-2 repo not found at ${VIDEOPHY_REPO_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VIDEOPHY2_CKPT_DIR}" ]]; then
  echo "[ERROR] VideoPhy-2 checkpoint dir not found at ${VIDEOPHY2_CKPT_DIR}" >&2
  exit 1
fi
if ! verify_videophy2_python "${VIDEOPHY2_PYTHON}" "${PROJECT_ROOT}"; then
  echo "[ERROR] VideoPhy-2 python preflight failed: ${VIDEOPHY2_PYTHON}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

print_target_gpu_processes() {
  local gpu_csv="$1"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', *' -v gpu_csv="${gpu_csv}" '
      BEGIN {
        split(gpu_csv, gpus, ",");
        while (("nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits" | getline line) > 0) {
          split(line, parts, /, */);
          gpu_uuid[parts[1]] = parts[2];
        }
        for (i in gpus) {
          wanted_uuid[gpu_uuid[gpus[i]]] = gpus[i];
        }
      }
      ($1 in wanted_uuid) {
        printf("gpu=%s pid=%s mem=%sMiB cmd=%s\n", wanted_uuid[$1], $2, $4, $3);
      }
    '
}

collect_target_gpu_pids() {
  local gpu_csv="$1"
  print_target_gpu_processes "${gpu_csv}" | awk '{for (i=1;i<=NF;i++) if ($i ~ /^pid=/) {sub(/^pid=/, "", $i); print $i}}' | sort -u
}

log "Project root: ${PROJECT_ROOT}"
log "Config: ${CONFIG_PATH}"
log "Manifest: ${MANIFEST}"
log "Checkpoint dir: ${VIDEOPHY2_CKPT_DIR}"
log "VideoPhy-2 python: ${VIDEOPHY2_PYTHON}"
log "GPU list: ${GPU_LIST}"
log "Target GPU processes before launch:"
print_target_gpu_processes "${GPU_LIST}" || true

if [[ "${KILL_EXISTING_GPU_PIDS}" == "1" ]]; then
  mapfile -t TARGET_PIDS < <(collect_target_gpu_pids "${GPU_LIST}")
  if [[ "${#TARGET_PIDS[@]}" -gt 0 ]]; then
    log "Killing existing processes on GPUs ${GPU_LIST}: ${TARGET_PIDS[*]}"
    kill "${TARGET_PIDS[@]}" || true
    sleep "${KILL_GRACE_SEC}"
    mapfile -t TARGET_PIDS < <(collect_target_gpu_pids "${GPU_LIST}")
    if [[ "${#TARGET_PIDS[@]}" -gt 0 ]]; then
      log "Force killing remaining processes on GPUs ${GPU_LIST}: ${TARGET_PIDS[*]}"
      kill -9 "${TARGET_PIDS[@]}" || true
    fi
  else
    log "No existing processes found on GPUs ${GPU_LIST}"
  fi
fi

TASKS=(
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|42"
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|123"
  "exp_base_zeroshot|${BASE_GENERATED_ROOT}|3407"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|42"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|123"
  "exp_stage1_epoch2|${STAGE1_GENERATED_ROOT}|3407"
)

run_one_seed() {
  local gpu="$1"
  local experiment_name="$2"
  local generated_root="$3"
  local seed="$4"
  local generated_dir="${generated_root}/seed_${seed}/csgo_metrics/videos"

  if [[ ! -d "${generated_dir}" ]]; then
    echo "[ERROR][GPU ${gpu}] generated videos missing: ${generated_dir}" >&2
    return 1
  fi

  log "[RUN][GPU ${gpu}] ${experiment_name} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${VIDEOPHY2_PYTHON}" -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${experiment_name}" \
    --manifest_csv "${MANIFEST}" \
    --generated_root "${generated_root}" \
    --seed "${seed}" \
    --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
    --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
    --output_root "${OUTPUT_ROOT}"
}

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local stride="$3"
  local task=""
  local experiment_name=""
  local generated_root=""
  local seed=""

  for ((idx=worker_idx; idx<${#TASKS[@]}; idx+=stride)); do
    task="${TASKS[$idx]}"
    IFS='|' read -r experiment_name generated_root seed <<< "${task}"
    run_one_seed "${gpu}" "${experiment_name}" "${generated_root}" "${seed}"
  done
}

declare -a JOB_PIDS=()
declare -a JOB_GPUS=()

for worker_idx in "${!GPUS[@]}"; do
  worker "${GPUS[$worker_idx]}" "${worker_idx}" "${#GPUS[@]}" &
  JOB_PIDS+=("$!")
  JOB_GPUS+=("${GPUS[$worker_idx]}")
  log "[PID][GPU ${GPUS[$worker_idx]}] worker pid=$!"
done

monitor_jobs() {
  while true; do
    alive=0
    status_parts=()
    for idx in "${!JOB_PIDS[@]}"; do
      pid="${JOB_PIDS[$idx]}"
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        status_parts+=("gpu=${JOB_GPUS[$idx]}:pid=${pid}")
      fi
    done
    if [[ "${alive}" -eq 0 ]]; then
      break
    fi
    log "[WAIT] ${status_parts[*]}"
    sleep "${STATUS_EVERY_SEC}"
  done
}

monitor_jobs &
MONITOR_PID=$!

failed=0
for idx in "${!JOB_PIDS[@]}"; do
  if wait "${JOB_PIDS[$idx]}"; then
    log "[OK][GPU ${JOB_GPUS[$idx]}] worker finished"
  else
    failed=1
    log "[FAIL][GPU ${JOB_GPUS[$idx]}] worker failed"
  fi
done

wait "${MONITOR_PID}" || true

if [[ "${failed}" -ne 0 ]]; then
  log "[ERROR] One or more VideoPhy-2 workers failed. Summary will not be generated."
  exit 1
fi

for experiment_name in exp_base_zeroshot exp_stage1_epoch2; do
  log "[SUMMARY] Writing aggregate summary for ${experiment_name}"
  "${VIDEOPHY2_PYTHON}" -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${experiment_name}" \
    --summary_only \
    --output_root "${OUTPUT_ROOT}"
done

log "[DONE] VideoPhy-2 summaries written under ${OUTPUT_ROOT}/runs/eval/videophy2"
