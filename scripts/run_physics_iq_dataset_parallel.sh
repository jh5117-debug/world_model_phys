#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

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

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/physics_iq_dataset_eval.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_dataset_val_physics_iq_real}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/metadata_val.csv}"
REFERENCE_SOURCE_ROOT="${REFERENCE_SOURCE_ROOT:-${DATASET_DIR}}"
CANDIDATE_SOURCE_ROOT="${CANDIDATE_SOURCE_ROOT:-${DATASET_DIR}}"
REFERENCE_SOURCE_MODE="${REFERENCE_SOURCE_MODE:-dataset_clip}"
CANDIDATE_SOURCE_MODE="${CANDIDATE_SOURCE_MODE:-dataset_clip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEO_FILENAME="${VIDEO_FILENAME:-video.mp4}"
VIDEO_SUFFIX="${VIDEO_SUFFIX:-_gen.mp4}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
SEED="${SEED:-0}"
STATUS_EVERY_SEC="${STATUS_EVERY_SEC:-30}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${REFERENCE_SOURCE_ROOT}" ]]; then
  echo "[ERROR] Reference source root not found: ${REFERENCE_SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CANDIDATE_SOURCE_ROOT}" ]]; then
  echo "[ERROR] Candidate source root not found: ${CANDIDATE_SOURCE_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

log "Project root: ${PROJECT_ROOT}"
log "Config: ${CONFIG_PATH}"
log "Manifest: ${MANIFEST}"
log "Reference root: ${REFERENCE_SOURCE_ROOT} (${REFERENCE_SOURCE_MODE})"
log "Candidate root: ${CANDIDATE_SOURCE_ROOT} (${CANDIDATE_SOURCE_MODE})"
log "Experiment: ${EXPERIMENT_NAME}"
log "GPU list: ${GPU_LIST}"
log "Note: Physics-IQ-style metric is CPU-bound; GPU labels are used to shard workers on H20."

BASE_OUTPUT="${OUTPUT_ROOT}/runs/eval/physics_iq/${EXPERIMENT_NAME}"
SHARD_DIR="${BASE_OUTPUT}/shards"
mkdir -p "${SHARD_DIR}"

python - "${MANIFEST}" "${SHARD_DIR}" "${#GPUS[@]}" <<'PY'
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
  shard_experiment="${EXPERIMENT_NAME}_shard_${shard_idx}"
  shard_manifest="${SHARD_DIR}/manifest_shard_${shard_idx}.csv"
  if [[ "$(wc -l < "${shard_manifest}")" -le 1 ]]; then
    echo "[SKIP] ${shard_manifest} has no data rows"
    continue
  fi
  gpu="${GPUS[$shard_idx]}"
  row_count="$(($(wc -l < "${shard_manifest}") - 1))"
  log "[RUN][GPU ${gpu}] ${shard_experiment} rows=${row_count}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m physical_consistency.cli.run_physics_iq \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${shard_experiment}" \
    --manifest_csv "${shard_manifest}" \
    --reference_source_mode "${REFERENCE_SOURCE_MODE}" \
    --candidate_source_mode "${CANDIDATE_SOURCE_MODE}" \
    --reference_source_root "${REFERENCE_SOURCE_ROOT}" \
    --candidate_source_root "${CANDIDATE_SOURCE_ROOT}" \
    --video_filename "${VIDEO_FILENAME}" \
    --video_suffix "${VIDEO_SUFFIX}" \
    --output_root "${OUTPUT_ROOT}" \
    --seed "${SEED}" &
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
    log "[OK][GPU ${JOB_GPUS[$idx]}] ${JOB_EXPERIMENTS[$idx]}"
  else
    failed=1
    log "[FAIL][GPU ${JOB_GPUS[$idx]}] ${JOB_EXPERIMENTS[$idx]}"
  fi
done

wait "${MONITOR_PID}" || true

if [[ "${failed}" -ne 0 ]]; then
  log "[ERROR] One or more Physics-IQ shard jobs failed. Summary will not be generated."
  exit 1
fi

python - "${OUTPUT_ROOT}" "${EXPERIMENT_NAME}" "${SEED}" "${#GPUS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
experiment_name = sys.argv[2]
seed = sys.argv[3]
shard_count = int(sys.argv[4])

physics_root = output_root / "runs" / "eval" / "physics_iq"
target_dir = physics_root / experiment_name / f"seed_{seed}"
target_dir.mkdir(parents=True, exist_ok=True)

merged_rows = []
fieldnames = None
for shard_idx in range(shard_count):
    shard_csv = (
        physics_root
        / f"{experiment_name}_shard_{shard_idx}"
        / f"seed_{seed}"
        / "output_pairs.csv"
    )
    if not shard_csv.exists():
        continue
    with shard_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames or []
        merged_rows.extend(reader)

if fieldnames is None:
    fieldnames = [
        "sample_id",
        "clip_path",
        "prompt",
        "reference_videopath",
        "candidate_videopath",
        "compare_frame_count",
        "mse_mean",
        "spatiotemporal_iou_mean",
        "spatial_iou",
        "weighted_spatial_iou",
        "physics_iq_style_score",
    ]

output_csv = target_dir / "output_pairs.csv"
with output_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)
PY

log "[SUMMARY] Writing aggregate summary for ${EXPERIMENT_NAME}"
python -m physical_consistency.cli.run_physics_iq \
  --config "${CONFIG_PATH}" \
  --env_file "${ENV_FILE}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --summary_only \
  --output_root "${OUTPUT_ROOT}"

log "[DONE] Physics-IQ-style summary written to ${OUTPUT_ROOT}/runs/eval/physics_iq/${EXPERIMENT_NAME}/summary.json"
