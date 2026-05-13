#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/world_model_phys/PHYS/world_model_phys}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/world_model_phys}"
ASSET_ROOT="${ASSET_ROOT:-${WORKSPACE_ROOT}/PHYS}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_pai.env}"
PY="${PY:-${WORKSPACE_ROOT}/.conda_envs/phys-main/bin/python}"
RAW="${RAW:-${ASSET_ROOT}/Dataset/Phy_Dataset/PhysInOne/raw}"
OUT="${OUT:-${ASSET_ROOT}/Dataset/Phy_Dataset/PhysInOne_act_mixed_81f_384_stride40}"
BASE_ACT="${BASE_ACT:-${ASSET_ROOT}/weight/Lingbot-base-act}"
LINGBOT_CODE_DIR="${LINGBOT_CODE_DIR:-${WORKSPACE_ROOT}/code/lingbot-world}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
WINDOW_STRIDE="${WINDOW_STRIDE:-40}"
STATIC_CAMERA_IDS="${STATIC_CAMERA_IDS:-all}"
MOVING_REPEAT="${MOVING_REPEAT:-4}"
VAL_COUNT="${VAL_COUNT:-30}"
VAL_SEED="${VAL_SEED:-20260513}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_SPLIT_VAL="${RUN_SPLIT_VAL:-1}"
RUN_STAGE1_SMOKE="${RUN_STAGE1_SMOKE:-1}"
RUN_STAGE1_FULL="${RUN_STAGE1_FULL:-0}"

SMOKE_MICRO_STEPS="${SMOKE_MICRO_STEPS:-2}"
SMOKE_EXP="${SMOKE_EXP:-smoke_stage1_physinone_act_mixed_low_8gpu_81f384_ms${SMOKE_MICRO_STEPS}}"
FULL_EXP="${FULL_EXP:-exp_stage1_physinone_act_mixed_low_8gpu_81f384_stride${WINDOW_STRIDE}}"
FULL_NUM_EPOCHS="${FULL_NUM_EPOCHS:-1}"
FULL_GRAD_ACCUM="${FULL_GRAD_ACCUM:-4}"

section() { echo; echo "========== $* =========="; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

cd "${PROJECT_ROOT}"
mkdir -p logs configs/generated

[[ -x "${PY}" ]] || fail "PY is not executable: ${PY}"
[[ -d "${RAW}" ]] || fail "RAW missing: ${RAW}"
[[ -d "${BASE_ACT}" ]] || fail "BASE_ACT missing: ${BASE_ACT}"
[[ -f "${LINGBOT_CODE_DIR}/wan/modules/model.py" ]] || fail "LINGBOT_CODE_DIR invalid: ${LINGBOT_CODE_DIR}"

if [[ "${RUN_PREPROCESS}" == "1" ]]; then
  section "Preprocess PhysInOne mixed act dataset"
  LOG="${PROJECT_ROOT}/logs/preprocess_physinone_act_mixed_stride${WINDOW_STRIDE}.log"
  PID="${PROJECT_ROOT}/logs/preprocess_physinone_act_mixed_stride${WINDOW_STRIDE}.pid"
  rm -rf "${OUT}"
  CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH="${PROJECT_ROOT}/src" \
  nohup "${PY}" -m physical_consistency.cli.preprocess_physinone_moving_act \
    --input_root "${RAW}" \
    --output_dir "${OUT}" \
    --clip_frames 81 \
    --sampling_mode contiguous_windows \
    --window_stride "${WINDOW_STRIDE}" \
    --output_height 384 \
    --output_width 384 \
    --target_fps 16 \
    --val_ratio 0.0 \
    --include_static_cameras \
    --static_camera_ids "${STATIC_CAMERA_IDS}" \
    --moving_repeat "${MOVING_REPEAT}" \
    > "${LOG}" 2>&1 & echo $! > "${PID}"

  echo "PID: $(cat "${PID}")"
  echo "LOG: ${LOG}"
  while ps -p "$(cat "${PID}" 2>/dev/null)" >/dev/null 2>&1; do
    date
    tail -n 25 "${LOG}" || true
    sleep 60
  done
  echo "=== preprocess finished ==="
  tail -n 120 "${LOG}" || true
fi

if [[ "${RUN_SPLIT_VAL}" == "1" ]]; then
  section "Split metadata_val.csv"
  PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - "${OUT}" "${VAL_COUNT}" "${VAL_SEED}" <<'PY'
from pathlib import Path
import csv
import random
import shutil
import sys

root = Path(sys.argv[1])
val_count = int(sys.argv[2])
seed = int(sys.argv[3])
src_all = root / "metadata_train_all_before_val_split.csv"
if not src_all.exists():
    shutil.copy2(root / "metadata_train.csv", src_all)

rows = list(csv.DictReader(open(src_all, newline="", encoding="utf-8")))
if not rows:
    raise SystemExit(f"No rows in {src_all}")
fields = list(rows[0].keys())

unique_by_clip = {}
for row in rows:
    unique_by_clip.setdefault(row["clip_path"], row)

by_traj = {}
for row in unique_by_clip.values():
    by_traj.setdefault(row["trajectory_name"], []).append(row)

one_per_traj = []
for traj, traj_rows in sorted(by_traj.items()):
    moving = [r for r in traj_rows if r["camera_id"] == "CineCamera_Moving"]
    one_per_traj.append((moving or traj_rows)[0])

random.Random(seed).shuffle(one_per_traj)
val_unique = one_per_traj[:val_count]
val_paths = {row["clip_path"] for row in val_unique}
train_rows = [row for row in rows if row["clip_path"] not in val_paths]
val_rows = val_unique

for name, out_rows in [("metadata_train.csv", train_rows), ("metadata_val.csv", val_rows)]:
    with open(root / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

print("source_rows", len(rows))
print("unique_clips", len(unique_by_clip))
print("train_rows", len(train_rows))
print("val_rows", len(val_rows))
print("val_unique_trajectories", len({row["trajectory_name"] for row in val_rows}))
PY
fi

make_config() {
  local out_config="$1"
  local exp_name="$2"
  local num_epochs="$3"
  local grad_accum="$4"
  local max_micro_steps="$5"
  PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - "${out_config}" "${exp_name}" "${num_epochs}" "${grad_accum}" "${max_micro_steps}" <<'PY'
from pathlib import Path
import sys

out_config, exp_name, num_epochs, grad_accum, max_micro_steps = sys.argv[1:6]
src = Path("configs/train_stage1_physinone_cam_75f_384_bf16_trainonly.yaml")
updates = {
    "experiment_name": exp_name,
    "run_group": "stage1_physinone_act_mixed_low",
    "control_type": "act",
    "num_epochs": num_epochs,
    "gradient_accumulation_steps": grad_accum,
    "num_frames": "81",
    "height": "384",
    "width": "384",
    "dataset_repeat": "1",
    "num_workers": "4",
    "save_every_n_epochs": "0",
    "max_train_micro_steps": max_micro_steps,
    "student_lora_block_start": "20",
    "student_lora_chunk_size": "512",
    "student_ffn_chunk_size": "4096",
    "videophy2_eval": "",
}
lines = []
seen = set()
skip_videophy_block = False
for raw in src.read_text(encoding="utf-8").splitlines():
    if raw.startswith("videophy2_eval:"):
        lines.extend([
            "videophy2_eval:",
            "  enabled: false",
            "  every_n_epochs: 1",
            "  generation_command: \"\"",
            "  score_command: \"\"",
            "  summary_json: \"\"",
            "  working_dir: .",
            "  fail_fast: false",
            "  env: {}",
        ])
        seen.add("videophy2_eval")
        skip_videophy_block = True
        continue
    if skip_videophy_block and (raw.startswith(" ") or not raw.strip()):
        continue
    skip_videophy_block = False
    key = raw.split(":", 1)[0].strip() if ":" in raw and not raw.startswith(" ") else ""
    if key in updates and key != "videophy2_eval":
        lines.append(f"{key}: {updates[key]}")
        seen.add(key)
    else:
        lines.append(raw)
for key, value in updates.items():
    if key not in seen and key != "videophy2_eval":
        lines.append(f"{key}: {value}")
Path(out_config).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_config)
PY
}

run_stage1_low() {
  local exp_name="$1"
  local config="$2"
  local log="${PROJECT_ROOT}/logs/train_${exp_name}.log"
  local pid="${PROJECT_ROOT}/logs/train_${exp_name}.pid"
  section "Stage1 low run: ${exp_name}"
  GPU_LIST="${GPU_LIST}" NUM_GPUS="${NUM_GPUS}" \
  RUN_WITH_NOHUP=1 TAIL_IMPORTANT_LOGS=0 \
  LOG_FILE="${log}" PID_FILE="${pid}" \
  LINGBOT_CODE_DIR="${LINGBOT_CODE_DIR}" \
  bash scripts/run_train_stage1_physinone_cam_h20.sh \
    --env_file "${ENV_FILE}" \
    --config "${config}" \
    --gpu_list "${GPU_LIST}" \
    --num_gpus "${NUM_GPUS}" \
    --branch_mode low \
    --experiment_name "${exp_name}" \
    --dataset_dir "${OUT}" \
    --base_model_dir "${BASE_ACT}" \
    --source_checkpoint_dir "${BASE_ACT}" \
    --companion_checkpoint_dir "${BASE_ACT}" \
    --control_type act \
    --num_frames 81 \
    --height 384 \
    --width 384

  echo "PID: $(cat "${pid}")"
  echo "LOG: ${log}"
  while ps -p "$(cat "${pid}" 2>/dev/null)" >/dev/null 2>&1; do
    date
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv || true
    grep -nE "\\[Stage1\\]|control_type|Dataset ready|step=|final_branch_dir|Traceback|OutOfMemory|RuntimeError|ERROR" "${log}" | tail -100 || true
    sleep 60
  done
  echo "=== stage1 ${exp_name} finished ==="
  tail -n 180 "${log}" || true
}

if [[ "${RUN_STAGE1_SMOKE}" == "1" ]]; then
  SMOKE_CONFIG="${PROJECT_ROOT}/configs/generated/pai_stage1_act_mixed_low_smoke.yaml"
  make_config "${SMOKE_CONFIG}" "${SMOKE_EXP}" 1 1 "${SMOKE_MICRO_STEPS}"
  run_stage1_low "${SMOKE_EXP}" "${SMOKE_CONFIG}"
fi

if [[ "${RUN_STAGE1_FULL}" == "1" ]]; then
  FULL_CONFIG="${PROJECT_ROOT}/configs/generated/pai_stage1_act_mixed_low_full.yaml"
  make_config "${FULL_CONFIG}" "${FULL_EXP}" "${FULL_NUM_EPOCHS}" "${FULL_GRAD_ACCUM}" 0
  run_stage1_low "${FULL_EXP}" "${FULL_CONFIG}"
fi

section "Done"
echo "OUT=${OUT}"
