"""Path configuration loading with cluster defaults."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template

from .defaults import (
    CONFIG_DIR,
    DEFAULT_BASE_MODEL_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_FINETUNE_CODE_DIR,
    DEFAULT_LINGBOT_CODE_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHYCSGO_WEIGHTED_DIR,
    DEFAULT_PHYSINONE_CAM_DIR,
    DEFAULT_PHYSINONE_RAW_DIR,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_STAGE1_CKPT_DIR,
    DEFAULT_STAGE1_FINAL_DIR,
    DEFAULT_TEACHER_CKPT_DIR,
    DEFAULT_VIDEOPHY2_CKPT_DIR,
    DEFAULT_VIDEOPHY_REPO_DIR,
    DEFAULT_VIDEOREPA_REPO_DIR,
    DEFAULT_WANDB_DIR,
)
from .io import resolve_project_path


@dataclass(slots=True)
class PathConfig:
    """Resolved path configuration for this subproject."""

    base_model_dir: str = DEFAULT_BASE_MODEL_DIR
    stage1_ckpt_dir: str = DEFAULT_STAGE1_CKPT_DIR
    stage1_final_dir: str = DEFAULT_STAGE1_FINAL_DIR
    dataset_dir: str = DEFAULT_DATASET_DIR
    raw_data_dir: str = DEFAULT_RAW_DATA_DIR
    physinone_raw_dir: str = DEFAULT_PHYSINONE_RAW_DIR
    physinone_cam_dir: str = DEFAULT_PHYSINONE_CAM_DIR
    phycsgo_weighted_dir: str = DEFAULT_PHYCSGO_WEIGHTED_DIR
    lingbot_code_dir: str = DEFAULT_LINGBOT_CODE_DIR
    finetune_code_dir: str = DEFAULT_FINETUNE_CODE_DIR
    output_root: str = DEFAULT_OUTPUT_ROOT
    wandb_dir: str = DEFAULT_WANDB_DIR
    videophy_repo_dir: str = DEFAULT_VIDEOPHY_REPO_DIR
    videorepa_repo_dir: str = DEFAULT_VIDEOREPA_REPO_DIR
    teacher_ckpt_dir: str = DEFAULT_TEACHER_CKPT_DIR
    videophy2_ckpt_dir: str = DEFAULT_VIDEOPHY2_CKPT_DIR

    def to_dict(self) -> dict[str, str]:
        """Serialize for logging / W&B config."""
        return asdict(self)


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE lines from an env file."""
    env_path = Path(path or CONFIG_DIR / "path_config_cluster.env")
    if not env_path.exists():
        return {}

    output: dict[str, str] = {}
    scope = os.environ.copy()
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        expanded = os.path.expanduser(
            Template(value.strip()).safe_substitute({**scope, **output})
        )
        output[key.strip()] = expanded
        scope[key.strip()] = expanded
    return output


def resolve_path_config(
    args: argparse.Namespace | None = None,
    *,
    env_file: str | Path | None = None,
) -> PathConfig:
    """Resolve path config from defaults, env file, environment, and args."""
    file_env = load_env_file(env_file)
    value_map = {
        "base_model_dir": _pick("BASE_MODEL_DIR", file_env, args, DEFAULT_BASE_MODEL_DIR),
        "stage1_ckpt_dir": _pick("STAGE1_CKPT_DIR", file_env, args, DEFAULT_STAGE1_CKPT_DIR),
        "stage1_final_dir": _pick("STAGE1_FINAL_DIR", file_env, args, DEFAULT_STAGE1_FINAL_DIR),
        "dataset_dir": _pick("DATASET_DIR", file_env, args, DEFAULT_DATASET_DIR),
        "raw_data_dir": _pick("RAW_DATA_DIR", file_env, args, DEFAULT_RAW_DATA_DIR),
        "physinone_raw_dir": _pick("PHYSINONE_RAW_DIR", file_env, args, DEFAULT_PHYSINONE_RAW_DIR),
        "physinone_cam_dir": _pick("PHYSINONE_CAM_DIR", file_env, args, DEFAULT_PHYSINONE_CAM_DIR),
        "phycsgo_weighted_dir": _pick(
            "PHYCSGO_WEIGHTED_DIR", file_env, args, DEFAULT_PHYCSGO_WEIGHTED_DIR
        ),
        "lingbot_code_dir": _pick(
            "LINGBOT_CODE_DIR", file_env, args, DEFAULT_LINGBOT_CODE_DIR
        ),
        "finetune_code_dir": _pick(
            "FINETUNE_CODE_DIR", file_env, args, DEFAULT_FINETUNE_CODE_DIR
        ),
        "output_root": _pick("OUTPUT_ROOT", file_env, args, DEFAULT_OUTPUT_ROOT),
        "wandb_dir": _pick("WANDB_DIR", file_env, args, DEFAULT_WANDB_DIR),
        "videophy_repo_dir": _pick(
            "VIDEOPHY_REPO_DIR", file_env, args, DEFAULT_VIDEOPHY_REPO_DIR
        ),
        "videorepa_repo_dir": _pick(
            "VIDEOREPA_REPO_DIR", file_env, args, DEFAULT_VIDEOREPA_REPO_DIR
        ),
        "teacher_ckpt_dir": _pick(
            "TEACHER_CKPT_DIR", file_env, args, DEFAULT_TEACHER_CKPT_DIR
        ),
        "videophy2_ckpt_dir": _pick(
            "VIDEOPHY2_CKPT_DIR", file_env, args, DEFAULT_VIDEOPHY2_CKPT_DIR
        ),
    }
    return PathConfig(**value_map)


def _pick(
    env_key: str,
    file_env: dict[str, str],
    args: argparse.Namespace | None,
    default: str,
) -> str:
    attr_name = env_key.lower()
    arg_value = getattr(args, attr_name, None) if args is not None else None
    if arg_value:
        return resolve_project_path(arg_value)
    if env_key in os.environ and os.environ[env_key]:
        return resolve_project_path(os.environ[env_key])
    if env_key in file_env and file_env[env_key]:
        return resolve_project_path(file_env[env_key])
    return resolve_project_path(default)
