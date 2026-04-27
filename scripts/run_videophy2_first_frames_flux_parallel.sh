#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_cluster.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/manifests/videophy2_test_upsampled.csv}"
MODEL_TAG="${MODEL_TAG:-flux2_dev_turbo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/videophy2_conditioned/${MODEL_TAG}}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-dev}"
TURBO_LORA_ID="${TURBO_LORA_ID:-fal/FLUX.2-dev-Turbo}"
TURBO_WEIGHT_NAME="${TURBO_WEIGHT_NAME:-flux.2-turbo-lora.safetensors}"
DISABLE_TURBO="${DISABLE_TURBO:-0}"
BASE_SEED="${BASE_SEED:-42}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:--1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MAX_ROWS_PER_GPU="${MAX_ROWS_PER_GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FLUX_FIRST_FRAME_PYTHON="${FLUX_FIRST_FRAME_PYTHON:-python}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/shards"

ROW_COUNT="$("${FLUX_FIRST_FRAME_PYTHON}" - "${MANIFEST}" "${MAX_SAMPLES}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
max_samples = int(sys.argv[2])
with manifest.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    count = sum(1 for _ in reader)
if max_samples > 0:
    count = min(count, max_samples)
print(count)
PY
)"

if [[ "${ROW_COUNT}" -le 0 ]]; then
  echo "[ERROR] Manifest has no data rows: ${MANIFEST}" >&2
  exit 1
fi

SHARD_COUNT="${#GPUS[@]}"
if [[ "${MAX_ROWS_PER_GPU}" -gt 0 ]]; then
  SHARD_COUNT="$(( (ROW_COUNT + MAX_ROWS_PER_GPU - 1) / MAX_ROWS_PER_GPU ))"
  if [[ "${SHARD_COUNT}" -gt "${#GPUS[@]}" ]]; then
    echo "[ERROR] Need ${SHARD_COUNT} GPUs to keep <= ${MAX_ROWS_PER_GPU} rows/GPU, but only ${#GPUS[@]} were provided." >&2
    exit 1
  fi
fi

"${FLUX_FIRST_FRAME_PYTHON}" - "${MANIFEST}" "${OUTPUT_DIR}/shards" "${SHARD_COUNT}" "${MAX_SAMPLES}" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])
max_samples = int(sys.argv[4])

with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

if max_samples > 0:
    rows = rows[:max_samples]

fieldnames = list(rows[0].keys()) if rows else []
shards = [[] for _ in range(shard_count)]
for idx, row in enumerate(rows):
    shards[idx % shard_count].append(row)

for idx, shard_rows in enumerate(shards):
    output_path = shard_dir / f"manifest_shard_{idx}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard_rows)
PY

declare -a JOB_PIDS=()
declare -a JOB_GPUS=()
declare -a JOB_SHARDS=()

for shard_idx in "${!GPUS[@]}"; do
  if [[ "${shard_idx}" -ge "${SHARD_COUNT}" ]]; then
    break
  fi
  shard_manifest="${OUTPUT_DIR}/shards/manifest_shard_${shard_idx}.csv"
  if [[ "$(wc -l < "${shard_manifest}")" -le 1 ]]; then
    continue
  fi
  gpu="${GPUS[$shard_idx]}"
  shard_manifest_out="${OUTPUT_DIR}/shards/generated_first_frames_shard_${shard_idx}.csv"
  cmd=(
    "${FLUX_FIRST_FRAME_PYTHON}" "${PROJECT_ROOT}/scripts/generate_flux_first_frames.py"
    --manifest_csv "${shard_manifest}"
    --output_dir "${OUTPUT_DIR}"
    --output_manifest_csv "${shard_manifest_out}"
    --model_id "${MODEL_ID}"
    --turbo_lora_id "${TURBO_LORA_ID}"
    --turbo_weight_name "${TURBO_WEIGHT_NAME}"
    --base_seed "${BASE_SEED}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --torch_dtype "${TORCH_DTYPE}"
    --device "cuda:0"
    --image_filename "image.jpg"
    --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
  )
  if [[ "${DISABLE_TURBO}" == "1" ]]; then
    cmd+=(--disable_turbo)
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip_existing)
  fi
  echo "[RUN][GPU ${gpu}] shard=${shard_idx} rows=$(($(wc -l < "${shard_manifest}") - 1))"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" &
  JOB_PIDS+=("$!")
  JOB_GPUS+=("${gpu}")
  JOB_SHARDS+=("${shard_idx}")
done

failed=0
for idx in "${!JOB_PIDS[@]}"; do
  pid="${JOB_PIDS[$idx]}"
  gpu="${JOB_GPUS[$idx]}"
  shard="${JOB_SHARDS[$idx]}"
  if wait "${pid}"; then
    echo "[OK][GPU ${gpu}] shard=${shard}"
  else
    failed=1
    echo "[FAIL][GPU ${gpu}] shard=${shard}" >&2
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] One or more FLUX first-frame shard jobs failed." >&2
  exit 1
fi

"${FLUX_FIRST_FRAME_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
shard_paths = sorted((output_dir / "shards").glob("generated_first_frames_shard_*.csv"))
rows = []
fieldnames = None
for shard_path in shard_paths:
    with shard_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = reader.fieldnames or []
        rows.extend(reader)
if fieldnames is None:
    fieldnames = []
with (output_dir / "generated_first_frames.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(output_dir / "generated_first_frames.csv")
PY
