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

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/videophy2_dataset_autoeval.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_dataset_val_autoeval_parallel}"
MANIFEST="${MANIFEST:-${DATASET_DIR}/metadata_val.csv}"
VIDEO_SOURCE_ROOT="${VIDEO_SOURCE_ROOT:-${DATASET_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
VIDEOPHY_REPO_DIR="${VIDEOPHY_REPO_DIR:-${PROJECT_ROOT}/third_party/videophy}"
VIDEOPHY2_CKPT_DIR="${VIDEOPHY2_CKPT_DIR:-${PROJECT_ROOT}/links/videophy2_checkpoint}"
VIDEO_FILENAME="${VIDEO_FILENAME:-video.mp4}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
SEED="${SEED:-0}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${VIDEO_SOURCE_ROOT}" ]]; then
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

BASE_OUTPUT="${OUTPUT_ROOT}/runs/eval/videophy2/${EXPERIMENT_NAME}"
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

for shard_idx in "${!GPUS[@]}"; do
  shard_experiment="${EXPERIMENT_NAME}_shard_${shard_idx}"
  shard_manifest="${SHARD_DIR}/manifest_shard_${shard_idx}.csv"
  if [[ "$(wc -l < "${shard_manifest}")" -le 1 ]]; then
    echo "[SKIP] ${shard_manifest} has no data rows"
    continue
  fi
  gpu="${GPUS[$shard_idx]}"
  echo "[RUN][GPU ${gpu}] ${shard_experiment}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m physical_consistency.cli.run_videophy2 \
    --config "${CONFIG_PATH}" \
    --env_file "${ENV_FILE}" \
    --experiment_name "${shard_experiment}" \
    --manifest_csv "${shard_manifest}" \
    --video_source_mode dataset_clip \
    --video_source_root "${VIDEO_SOURCE_ROOT}" \
    --video_filename "${VIDEO_FILENAME}" \
    --videophy_repo_dir "${VIDEOPHY_REPO_DIR}" \
    --checkpoint_dir "${VIDEOPHY2_CKPT_DIR}" \
    --output_root "${OUTPUT_ROOT}" \
    --seed "${SEED}" &
done
wait

python - "${OUTPUT_ROOT}" "${EXPERIMENT_NAME}" "${SEED}" "${#GPUS[@]}" <<'PY'
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

python -m physical_consistency.cli.run_videophy2 \
  --config "${CONFIG_PATH}" \
  --env_file "${ENV_FILE}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --summary_only \
  --output_root "${OUTPUT_ROOT}"

echo "[DONE] Parallel VideoPhy-2 AutoEval summary written to ${BASE_OUTPUT}/summary.json"
