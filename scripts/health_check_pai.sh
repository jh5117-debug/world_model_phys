#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/world_model_phys/PHYS/world_model_phys}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/world_model_phys}"
ASSET_ROOT="${ASSET_ROOT:-${WORKSPACE_ROOT}/PHYS}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/configs/path_config_pai.env}"
PY="${PY:-${WORKSPACE_ROOT}/.conda_envs/phys-main/bin/python}"
RAW="${RAW:-${ASSET_ROOT}/Dataset/Phy_Dataset/PhysInOne/raw}"
BASE_ACT="${BASE_ACT:-${ASSET_ROOT}/weight/Lingbot-base-act}"
LINGBOT_CODE_DIR="${LINGBOT_CODE_DIR:-${WORKSPACE_ROOT}/code/lingbot-world}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/runs/smoke/pai_health_preprocess_act_mixed}"
RUN_PREPROCESS_SMOKE="${RUN_PREPROCESS_SMOKE:-1}"
RUN_PYTEST="${RUN_PYTEST:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*" >&2; }
fail() { echo "[FAIL] $*" >&2; exit 1; }
section() { echo; echo "========== $* =========="; }
exists() {
  local path="$1"
  [[ -e "${path}" ]] && ok "exists: ${path}" || fail "missing: ${path}"
}

section "Paths"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "ASSET_ROOT=${ASSET_ROOT}"
echo "ENV_FILE=${ENV_FILE}"
echo "PY=${PY}"
echo "RAW=${RAW}"
echo "BASE_ACT=${BASE_ACT}"
echo "LINGBOT_CODE_DIR=${LINGBOT_CODE_DIR}"
exists "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
exists "${ENV_FILE}"
exists "${RAW}"
exists "${BASE_ACT}"
exists "${BASE_ACT}/Wan2.1_VAE.pth"
exists "${BASE_ACT}/low_noise_model"
exists "${BASE_ACT}/high_noise_model"
exists "${LINGBOT_CODE_DIR}/wan/modules/model.py"

section "Git"
if [[ -d .git ]]; then
  git remote -v || true
  git status --short || true
  echo "HEAD=$(git rev-parse --short HEAD)"
  echo "BRANCH=$(git branch --show-current || true)"
else
  fail "${PROJECT_ROOT} is not a git checkout. Clone or reattach git before training."
fi

section "Python"
if [[ ! -x "${PY}" ]]; then
  warn "Configured PY is not executable: ${PY}"
  PY="$(command -v python || true)"
  [[ -n "${PY}" ]] || fail "No usable python found"
  warn "falling back to ${PY}"
fi
"${PY}" --version
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - <<'PY'
import importlib
mods = [
    "torch",
    "cv2",
    "numpy",
    "yaml",
    "accelerate",
    "physical_consistency.cli.preprocess_physinone_moving_act",
    "physical_consistency.cli.train_stage1_physinone_cam",
]
for name in mods:
    mod = importlib.import_module(name)
    print("IMPORT_OK", name, getattr(mod, "__version__", ""))
PY

section "Compile"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" -m py_compile \
  src/physical_consistency/stages/stage1_physinone_cam/preprocess_moving_act.py \
  src/physical_consistency/stages/stage1_physinone_cam/config.py \
  src/physical_consistency/stages/stage1_physinone_cam/trainer.py \
  src/physical_consistency/stages/stage1_physinone_cam/runner.py
ok "py_compile passed"

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv
else
  warn "nvidia-smi not found"
fi

section "PhysInOne Raw Audit"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - "${RAW}" <<'PY'
from pathlib import Path
import zipfile
import sys

raw = Path(sys.argv[1])
zips = sorted(raw.rglob("*_trajectory.zip"))
print("zip_count", len(zips))
if not zips:
    raise SystemExit("no trajectory zips found")
with zipfile.ZipFile(zips[0]) as zf:
    names = zf.namelist()
    moving_json = [n for n in names if n.endswith("blender_CineCamera_Moving.json")]
    moving_rgb = [n for n in names if "/CineCamera_Moving/rgb/" in n and n.lower().endswith(".jpg")]
    static_json = [n for n in names if "blender_CineCamera_" in n and "Moving" not in n and n.endswith(".json")]
    static_rgb = [n for n in names if "/CineCamera_" in n and "/rgb/" in n and "Moving" not in n and n.lower().endswith(".jpg")]
    print("sample_zip", zips[0])
    print("moving_json_count", len(moving_json), "moving_rgb_count", len(moving_rgb))
    print("static_json_count", len(static_json), "static_rgb_count", len(static_rgb))
    if not moving_json or not moving_rgb:
        raise SystemExit("missing moving-camera members")
    if not static_json or not static_rgb:
        raise SystemExit("missing static-camera members")
PY

if [[ "${RUN_PYTEST}" == "1" ]]; then
  section "Unit Tests"
  if PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" -m pytest -q tests/test_stage1_physinone_moving_act_preprocess.py; then
    ok "moving-act preprocess tests passed"
  else
    warn "pytest failed; install pytest or inspect the output above"
  fi
fi

if [[ "${RUN_PREPROCESS_SMOKE}" == "1" ]]; then
  section "Preprocess Smoke"
  rm -rf "${SMOKE_ROOT}"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" \
    -m physical_consistency.cli.preprocess_physinone_moving_act \
    --input_root "${RAW}" \
    --output_dir "${SMOKE_ROOT}" \
    --clip_frames 8 \
    --sampling_mode uniform_single \
    --window_stride 8 \
    --output_height 64 \
    --output_width 64 \
    --target_fps 16 \
    --val_ratio 0.0 \
    --max_zips 1 \
    --include_static_cameras \
    --static_camera_ids 0 \
    --moving_repeat 2
  PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - "${SMOKE_ROOT}" <<'PY'
from pathlib import Path
import csv
import json
import numpy as np
import sys

root = Path(sys.argv[1])
rows = list(csv.DictReader(open(root / "metadata_train.csv", newline="", encoding="utf-8")))
summary = json.load(open(root / "preprocess_summary.json"))
print("summary", summary)
print("rows", len(rows))
moving = [r for r in rows if r["camera_id"] == "CineCamera_Moving"]
static = [r for r in rows if r["action_source"] == "static_camera_zero"]
if len(moving) != 2 or len(static) != 1:
    raise SystemExit(f"unexpected smoke row mix: moving={len(moving)} static={len(static)}")
static_actions = np.load(root / static[0]["clip_path"] / "action.npy")
moving_actions = np.load(root / moving[0]["clip_path"] / "action.npy")
if not np.allclose(static_actions, 0.0):
    raise SystemExit("static action.npy is not all-zero")
if float(np.abs(moving_actions).max()) <= 0.0:
    raise SystemExit("moving action.npy is all-zero")
print("PREPROCESS_SMOKE_OK")
PY
fi

section "Stage1 Config Parse Smoke"
PYTHONPATH="${PROJECT_ROOT}/src" "${PY}" - "${ENV_FILE}" "${BASE_ACT}" "${RAW}" <<'PY'
from argparse import Namespace
from physical_consistency.stages.stage1_physinone_cam.config import Stage1PhysInOneConfig
import sys

env_file, base_act, dataset_dir = sys.argv[1:4]
args = Namespace(
    control_type="act",
    experiment_name="pai_health_stage1_config_smoke",
    dataset_dir=dataset_dir,
    base_model_dir=base_act,
    physinone_raw_dir="",
    lingbot_code_dir="",
    output_root="",
    wandb_dir="",
    seed=None,
    num_epochs=1,
    num_frames=81,
    height=384,
    width=384,
    gradient_accumulation_steps=1,
    env_file=env_file,
)
cfg = Stage1PhysInOneConfig.from_yaml(
    "configs/train_stage1_physinone_cam_75f_384_bf16_trainonly.yaml",
    env_file=env_file,
    cli_args=args,
)
assert cfg.control_type == "act"
assert cfg.base_model_dir == base_act
print("STAGE1_CONFIG_SMOKE_OK", cfg.experiment_name, cfg.control_type)
PY

section "Done"
ok "PAI health check completed"
