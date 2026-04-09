"""Centralized cluster-path defaults and project constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "physical_consistency"
CONFIG_DIR = PROJECT_ROOT / "configs"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LINKS_DIR = PROJECT_ROOT / "links"
DATASET_ROOT = PROJECT_ROOT / "Dataset"

DEFAULT_BASE_MODEL_DIR = str(LINKS_DIR / "base_model")
DEFAULT_STAGE1_CKPT_DIR = str(LINKS_DIR / "stage1_epoch2")
DEFAULT_STAGE1_FINAL_DIR = str(LINKS_DIR / "stage1_final")
DEFAULT_DATASET_DIR = str(DATASET_ROOT / "processed_csgo_v3")
DEFAULT_RAW_DATA_DIR = str(LINKS_DIR / "raw_csgo_v3_train")
DEFAULT_LINGBOT_CODE_DIR = str(LINKS_DIR / "lingbot_code")
DEFAULT_FINETUNE_CODE_DIR = str(LINKS_DIR / "finetune_code")
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT)
DEFAULT_WANDB_DIR = str(PROJECT_ROOT / "runs" / "wandb")
DEFAULT_VIDEOPHY_REPO_DIR = str(PROJECT_ROOT / "third_party" / "videophy")
DEFAULT_VIDEOREPA_REPO_DIR = str(PROJECT_ROOT / "third_party" / "VideoREPA")
DEFAULT_TEACHER_CKPT_DIR = str(LINKS_DIR / "teacher_ckpt")
DEFAULT_VIDEOPHY2_CKPT_DIR = str(LINKS_DIR / "videophy2_checkpoint")

DEFAULT_SEED_LIST = [42, 123, 3407]
DEFAULT_FRAME_NUM = 81
DEFAULT_SAMPLE_STEPS = 70
DEFAULT_GUIDE_SCALE = 5.0
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
