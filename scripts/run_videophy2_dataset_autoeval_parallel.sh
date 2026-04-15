#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

declare -A PRESERVED_ENV_VARS=()
PRESERVE_KEYS=(
  CONFIG_PATH
  EXPERIMENT_NAME
  MANIFEST
  VIDEO_SOURCE_ROOT
  VIDEO_SOURCE_MODE
  MANIFEST_VIDEO_COLUMN
  MANIFEST_CAPTION_COLUMN
  VIDEOPHY_TORCH_DTYPE
  OUTPUT_ROOT
  VIDEOPHY_REPO_DIR
  VIDEOPHY2_CKPT_DIR
  VIDEO_FILENAME
  GPU_LIST
  SEED
  STATUS_EVERY_SEC
  KILL_EXISTING_GPU_PIDS
  KILL_GRACE_SEC
  MAX_ROWS_PER_GPU
  VIDEOPHY2_QUIET
  VIDEOPHY2_SUMMARY_STDOUT
)
for key in "${PRESERVE_KEYS[@]}"; do
  if [[ -v "${key}" ]]; then
    PRESERVED_ENV_VARS["${key}"]="${!key}"
  fi
done

log() {
  local message="$*"
  if [[ "${VIDEOPHY2_QUIET:-0}" == "1" ]]; then
    case "${message}" in
      "[ERROR]"*|"[FAIL]"*)
        ;;
      *)
        return
        ;;
    esac
  fi
  printf '[%s] %s\n' "$(date '+%F %T')" "${message}"
}

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

for key in "${!PRESERVED_ENV_VARS[@]}"; do
  printf -v "${key}" '%s' "${PRESERVED_ENV_VARS[${key}]}"
  export "${key}"
done

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/videophy2_dataset_autoeval.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_dataset_val_autoeval_parallel}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/metadata_val.csv}"
VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${DATASET_DIR}}"
VIDEO_SOURCE_MODE="${VIDEO_SOURCE_MODE:-dataset_clip}"
MANIFEST_VIDEO_COLUMN="${MANIFEST_VIDEO_COLUMN:-videopath}"
MANIFEST_CAPTION_COLUMN="${MANIFEST_CAPTION_COLUMN:-prompt}"
export VIDEOPHY_TORCH_DTYPE="${VIDEOPHY_TORCH_DTYPE:-fp32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEOPHY_REPO_DIR="${VIDEOPHY_REPO_DIR:-${PROJECT_ROOT}/third_party/videophy}"
VIDEOPHY2_CKPT_DIR="${VIDEOPHY2_CKPT_DIR:-${PROJECT_ROOT}/links/videophy2_checkpoint}"
VIDEO_FILENAME="${VIDEO_FILENAME:-video.mp4}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
SEED="${SEED:-0}"
STATUS_EVERY_SEC="${STATUS_EVERY_SEC:-30}"
KILL_EXISTING_GPU_PIDS="${KILL_EXISTING_GPU_PIDS:-0}"
KILL_GRACE_SEC="${KILL_GRACE_SEC:-5}"
MAX_ROWS_PER_GPU="${MAX_ROWS_PER_GPU:-0}"
VIDEOPHY2_QUIET="${VIDEOPHY2_QUIET:-0}"
VIDEOPHY2_SUMMARY_STDOUT="${VIDEOPHY2_SUMMARY_STDOUT:-1}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ "${VIDEO_SOURCE_MODE}" != "manifest_videopath" && "${VIDEO_SOURCE_MODE}" != "manifest_video_column" && ! -d "${VIDEO_SOURCE_ROOT}" ]]; then
  echo "[ERROR] Video source root not found: ${VIDEO_SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${VIDEOPHY_REPO_DIR}/VIDEOPHY2/inference.py" ]]; then
  echo "[ERROR] VideoPhy-2 repo not found at ${VIDEOPHY_REPO_DIR}" >&2
  exit 1
fi
if [[ ! -d "${VIDEOPHY2_CKPT_DIR}" ]]; then
  echo "[ERROR] VideoPhy-2 checkpoint dir not found at ${VIDEOPHY2_CKPT_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
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
log "Env file: ${ENV_FILE}"
log "Manifest: ${MANIFEST}"
log "Video source root: ${VIDEO_SOURCE_ROOT}"
log "Video source mode: ${VIDEO_SOURCE_MODE}"
log "Manifest video column: ${MANIFEST_VIDEO_COLUMN}"
log "Manifest caption column: ${MANIFEST_CAPTION_COLUMN}"
log "VideoPhy torch dtype: ${VIDEOPHY_TORCH_DTYPE}"
log "Checkpoint dir: ${VIDEOPHY2_CKPT_DIR}"
log "GPU list: ${GPU_LIST}"
log "Experiment: ${EXPERIMENT_NAME}"
if [[ "${VIDEOPHY2_QUIET}" != "1" ]]; then
  log "Target GPU processes before launch:"
  print_target_gpu_processes "${GPU_LIST}" || true
fi

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

BASE_OUTPUT="${OUTPUT_ROOT}/runs/eval/videophy2/${EXPERIMENT_NAME}"
SHARD_DIR="${BASE_OUTPUT}/shards"
mkdir -p "${SHARD_DIR}"

MANIFEST_ROWS="$(python - "${MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    count = sum(1 for _ in reader)
print(count)
PY
)"

if [[ "${MANIFEST_ROWS}" -le 0 ]]; then
  echo "[ERROR] Manifest has no data rows: ${MANIFEST}" >&2
  exit 1
fi

SHARD_COUNT="${#GPUS[@]}"
if [[ "${MAX_ROWS_PER_GPU}" -gt 0 ]]; then
  SHARD_COUNT="$(( (MANIFEST_ROWS + MAX_ROWS_PER_GPU - 1) / MAX_ROWS_PER_GPU ))"
  if [[ "${SHARD_COUNT}" -gt "${#GPUS[@]}" ]]; then
    echo "[ERROR] Need ${SHARD_COUNT} GPUs to keep <= ${MAX_ROWS_PER_GPU} rows/GPU, but GPU_LIST only has ${#GPUS[@]} GPUs." >&2
    exit 1
  fi
fi

log "Manifest rows: ${MANIFEST_ROWS}"
log "Shard count: ${SHARD_COUNT}"
if [[ "${MAX_ROWS_PER_GPU}" -gt 0 ]]; then
  log "Max rows per GPU: ${MAX_ROWS_PER_GPU}"
fi

python - "${MANIFEST}" "${SHARD_DIR}" "${SHARD_COUNT}" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])

with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    shards = [[] for _ in range(shard_count)]
    for idx, row in enumerate(reader):
        shards[idx % shard_count].append(row)

for idx, rows in enumerate(shards):
    output_path = shard_dir / f"manifest_shard_{idx}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
PY

declare -a JOB_PIDS=()
declare -a JOB_GPUS=()
declare -a JOB_EXPERIMENTS=()

for shard_idx in "${!GPUS[@]}"; do
  if [[ "${shard_idx}" -ge "${SHARD_COUNT}" ]]; then
    break
  fi
  shard_experiment="${EXPERIMENT_NAME}_shard_${shard_idx}"
  shard_manifest="${SHARD_DIR}/manifest_shard_${shard_idx}.csv"
  if [[ "$(wc -l < "${shard_manifest}")" -le 1 ]]; then
    echo "[SKIP] ${shard_manifest} has no data rows"
    continue
  fi
  gpu="${GPUS[$shard_idx]}"
  sa_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_sa.log"
  pc_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_pc.log"
  row_count="$(($(wc -l < "${shard_manifest}") - 1))"
  log "[RUN][GPU ${gpu}] ${shard_experiment} rows=${row_count}"
  log "[LOG][GPU ${gpu}] SA=${sa_log}"
  log "[LOG][GPU ${gpu}] PC=${pc_log}"
  if [[ "${VIDEOPHY2_QUIET}" == "1" ]]; then
    runner_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_runner.log"
    mkdir -p "$(dirname "${runner_log}")"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m physical_consistency.cli.run_videophy2 \
      --config "${CONFIG_PATH}" \
      --env_file "${ENV_FILE}" \
      --experiment_name "${shard_experiment}" \
      --manifest_csv "${shard_manifest}" \
      --video_source_mode "${VIDEO_SOURCE_MODE}" \
      --video_source_root "${VIDEO_SOURCE_ROOT}" \
      --manifest_video_column "${MANIFEST_VIDEO_COLUMN}" \
      --manifest_caption_column "${MANIFEST_CAPTION_COLUMN}" \
      --video_filename "${VIDEO_FILENAME}" \
      --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
      --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
      --output_root "${OUTPUT_ROOT}" \
      --seed "${SEED}" >"${runner_log}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -m physical_consistency.cli.run_videophy2 \
      --config "${CONFIG_PATH}" \
      --env_file "${ENV_FILE}" \
      --experiment_name "${shard_experiment}" \
      --manifest_csv "${shard_manifest}" \
      --video_source_mode "${VIDEO_SOURCE_MODE}" \
      --video_source_root "${VIDEO_SOURCE_ROOT}" \
      --manifest_video_column "${MANIFEST_VIDEO_COLUMN}" \
      --manifest_caption_column "${MANIFEST_CAPTION_COLUMN}" \
      --video_filename "${VIDEO_FILENAME}" \
      --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
      --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
      --output_root "${OUTPUT_ROOT}" \
      --seed "${SEED}" &
  fi
  job_pid=$!
  JOB_PIDS+=("${job_pid}")
  JOB_GPUS+=("${gpu}")
  JOB_EXPERIMENTS+=("${shard_experiment}")
  log "[PID][GPU ${gpu}] ${shard_experiment} pid=${job_pid}"
done

monitor_jobs() {
  while true; do
    alive=0
    status_parts=()
    for idx in "${!JOB_PIDS[@]}"; do
      pid="${JOB_PIDS[$idx]}"
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        shard_experiment="${JOB_EXPERIMENTS[$idx]}"
        gpu="${JOB_GPUS[$idx]}"
        sa_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_sa.log"
        sa_lines=0
        if [[ -f "${sa_log}" ]]; then
          sa_lines="$(wc -l < "${sa_log}")"
        fi
        status_parts+=("gpu=${gpu}:pid=${pid}:sa_lines=${sa_lines}")
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
  pid="${JOB_PIDS[$idx]}"
  gpu="${JOB_GPUS[$idx]}"
  shard_experiment="${JOB_EXPERIMENTS[$idx]}"
  sa_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_sa.log"
  pc_log="${OUTPUT_ROOT}/runs/eval/videophy2/${shard_experiment}/seed_${SEED}/videophy2_pc.log"
  if wait "${pid}"; then
    log "[OK][GPU ${gpu}] ${shard_experiment}"
  else
    failed=1
    log "[FAIL][GPU ${gpu}] ${shard_experiment}"
    log "[FAIL][GPU ${gpu}] Inspect logs: ${sa_log} ${pc_log}"
  fi
done

wait "${MONITOR_PID}" || true

if [[ "${failed}" -ne 0 ]]; then
  log "[ERROR] One or more shard jobs failed. Summary will not be generated."
  exit 1
fi

python - "${OUTPUT_ROOT}" "${EXPERIMENT_NAME}" "${SEED}" "${SHARD_COUNT}" <<'PY'
import csv
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
experiment_name = sys.argv[2]
seed = sys.argv[3]
shard_count = int(sys.argv[4])

videophy_root = output_root / "runs" / "eval" / "videophy2"
target_dir = videophy_root / experiment_name / f"seed_{seed}"
target_dir.mkdir(parents=True, exist_ok=True)

for task in ("sa", "pc"):
    merged_rows = []
    fieldnames = None
    for shard_idx in range(shard_count):
        shard_csv = (
            videophy_root
            / f"{experiment_name}_shard_{shard_idx}"
            / f"seed_{seed}"
            / f"output_{task}.csv"
        )
        if not shard_csv.exists():
            continue
        with shard_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames or []
            merged_rows.extend(reader)
    if fieldnames is None:
        fieldnames = ["videopath", "caption", "score"] if task == "sa" else ["videopath", "score"]
    output_csv = target_dir / f"output_{task}.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
PY

if [[ "${VIDEOPHY2_SUMMARY_STDOUT}" == "1" ]]; then
  log "[SUMMARY] Writing aggregate summary for ${EXPERIMENT_NAME}"
  python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --summary_only \
    --output_root "${OUTPUT_ROOT}"

  log "[DONE] Parallel VideoPhy-2 AutoEval summary written to ${BASE_OUTPUT}/summary.json"
else
  python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --summary_only \
    --output_root "${OUTPUT_ROOT}" >/dev/null 2>&1
fi
