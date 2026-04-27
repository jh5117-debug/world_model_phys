"""Configuration loading for pure Stage-1 PhysInOne camera training."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from physical_consistency.common.defaults import CONFIG_DIR
from physical_consistency.common.io import read_yaml, resolve_project_path
from physical_consistency.common.path_config import resolve_path_config


def _coerce_bool(value: bool | str | int | None, default: bool) -> bool:
    if value in {"", None}:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot coerce to bool: {value!r}")


@dataclass(slots=True)
class VideoPhy2EvalConfig:
    """Optional official VideoPhy-2 validation hook."""

    enabled: bool = False
    every_n_epochs: int = 1
    generation_command: str = ""
    score_command: str = ""
    summary_json: str = ""
    working_dir: str = ""
    fail_fast: bool = False
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict | None) -> "VideoPhy2EvalConfig":
        payload = dict(payload or {})
        env_payload = {str(key): str(value) for key, value in dict(payload.get("env") or {}).items()}
        return cls(
            enabled=_coerce_bool(payload.get("enabled"), False),
            every_n_epochs=max(int(payload.get("every_n_epochs", 1) or 1), 1),
            generation_command=str(payload.get("generation_command", "") or ""),
            score_command=str(payload.get("score_command", "") or ""),
            summary_json=resolve_project_path(payload.get("summary_json", "") or ""),
            working_dir=resolve_project_path(payload.get("working_dir", "") or ""),
            fail_fast=_coerce_bool(payload.get("fail_fast"), False),
            env=env_payload,
        )


@dataclass(slots=True)
class Stage1PhysInOneConfig:
    """Resolved config for Stage-1 PhysInOne camera-only fine-tuning."""

    experiment_name: str
    run_group: str
    seed: int
    base_model_dir: str
    dataset_dir: str
    physinone_raw_dir: str
    lingbot_code_dir: str
    output_root: str
    output_dir: str
    wandb_dir: str
    config_path: str
    config_hash: str
    env_file: str
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_frames: int = 81
    height: int = 480
    width: int = 480
    dataset_repeat: int = 1
    num_workers: int = 4
    save_every_n_epochs: int = 1
    max_train_micro_steps: int = 0
    student_tuning_mode: str = "lora"
    student_lora_rank: int = 16
    student_lora_alpha: int = 16
    student_lora_dropout: float = 0.0
    student_lora_block_start: int = 0
    student_lora_chunk_size: int = 0
    student_lora_merge_mode: str = "inplace"
    student_memory_efficient_modulation: bool = True
    student_memory_efficient_checkpoint_mode: str = "full"
    student_ffn_chunk_size: int = 4096
    student_norm_chunk_size: int = 0
    student_precision_profile: str = "mixed_safe"
    student_low_precision_dtype: str = "bf16"
    gradient_checkpointing: bool = True
    student_checkpoint_use_reentrant: bool | None = None
    student_ddp_find_unused_parameters: bool = True
    distributed_timeout_hours: int = 8
    videophy2_eval: VideoPhy2EvalConfig = field(default_factory=VideoPhy2EvalConfig)

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        env_file: str | Path | None = None,
        cli_args: argparse.Namespace | None = None,
    ) -> "Stage1PhysInOneConfig":
        config_path = Path(config_path).resolve()
        payload = read_yaml(config_path)
        path_cfg = resolve_path_config(cli_args, env_file=env_file)

        def _override_str(key: str, fallback: str) -> str:
            if cli_args is not None:
                value = getattr(cli_args, key, "")
                if value:
                    return resolve_project_path(value)
            raw = payload.get(key, "")
            if raw:
                return resolve_project_path(raw)
            return fallback

        def _override_int(key: str, fallback: int) -> int:
            if cli_args is not None:
                value = getattr(cli_args, key, None)
                if value not in {"", None}:
                    return int(value)
            if payload.get(key, "") not in {"", None}:
                return int(payload[key])
            return int(fallback)

        experiment_name = str(
            getattr(cli_args, "experiment_name", "") or payload.get("experiment_name", "exp_stage1_physinone_cam")
        )
        output_root = _override_str("output_root", path_cfg.output_root)
        dataset_dir = _override_str("dataset_dir", path_cfg.physinone_cam_dir)
        base_model_dir = _override_str("base_model_dir", path_cfg.base_model_dir)
        physinone_raw_dir = _override_str("physinone_raw_dir", path_cfg.physinone_raw_dir)
        lingbot_code_dir = _override_str("lingbot_code_dir", path_cfg.lingbot_code_dir)
        wandb_dir = _override_str("wandb_dir", path_cfg.wandb_dir)
        run_group = str(payload.get("run_group", "stage1_physinone_cam"))
        seed = _override_int("seed", int(payload.get("seed", 42) or 42))
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]

        student_tuning_mode = str(payload.get("student_tuning_mode", "lora")).strip().lower()
        if student_tuning_mode not in {"full", "lora"}:
            raise ValueError(f"Unsupported student_tuning_mode: {student_tuning_mode}")

        student_precision_profile = str(
            payload.get("student_precision_profile", "mixed_safe") or "mixed_safe"
        ).strip().lower()
        if student_precision_profile in {"", "auto", "native", "native_lowp", "lowp"}:
            student_precision_profile = "native_lowp" if student_precision_profile in {"native", "native_lowp", "lowp"} else "mixed_safe"
        elif student_precision_profile in {"mixed_safe", "safe", "safe_mixed"}:
            student_precision_profile = "mixed_safe"
        elif student_precision_profile in {"fp32", "float32", "full_fp32"}:
            student_precision_profile = "fp32"
        else:
            raise ValueError(f"Unsupported student_precision_profile: {student_precision_profile}")

        student_low_precision_dtype = str(
            payload.get("student_low_precision_dtype", "bf16") or "bf16"
        ).strip().lower()
        if student_low_precision_dtype in {"bfloat16"}:
            student_low_precision_dtype = "bf16"
        elif student_low_precision_dtype in {"float16", "half"}:
            student_low_precision_dtype = "fp16"
        elif student_low_precision_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"Unsupported student_low_precision_dtype: {student_low_precision_dtype}")

        student_checkpoint_use_reentrant = payload.get("student_checkpoint_use_reentrant", None)
        if student_checkpoint_use_reentrant not in {"", None}:
            student_checkpoint_use_reentrant = _coerce_bool(student_checkpoint_use_reentrant, False)
        else:
            student_checkpoint_use_reentrant = None

        output_dir = str(Path(output_root) / "checkpoints" / experiment_name)
        return cls(
            experiment_name=experiment_name,
            run_group=run_group,
            seed=seed,
            base_model_dir=base_model_dir,
            dataset_dir=dataset_dir,
            physinone_raw_dir=physinone_raw_dir,
            lingbot_code_dir=lingbot_code_dir,
            output_root=output_root,
            output_dir=output_dir,
            wandb_dir=wandb_dir,
            config_path=str(config_path),
            config_hash=config_hash,
            env_file=str(env_file or getattr(cli_args, "env_file", "") or (CONFIG_DIR / "path_config_cluster.env")),
            learning_rate=float(payload.get("learning_rate", 1.0e-5)),
            weight_decay=float(payload.get("weight_decay", 0.01)),
            num_epochs=_override_int("num_epochs", int(payload.get("num_epochs", 5) or 5)),
            gradient_accumulation_steps=_override_int(
                "gradient_accumulation_steps",
                int(payload.get("gradient_accumulation_steps", 4) or 4),
            ),
            max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
            num_frames=_override_int("num_frames", int(payload.get("num_frames", 81) or 81)),
            height=_override_int("height", int(payload.get("height", 480) or 480)),
            width=_override_int("width", int(payload.get("width", 480) or 480)),
            dataset_repeat=int(payload.get("dataset_repeat", 1) or 1),
            num_workers=int(payload.get("num_workers", 4) or 4),
            save_every_n_epochs=int(payload.get("save_every_n_epochs", 1) or 1),
            max_train_micro_steps=int(payload.get("max_train_micro_steps", 0) or 0),
            student_tuning_mode=student_tuning_mode,
            student_lora_rank=int(payload.get("student_lora_rank", 16) or 16),
            student_lora_alpha=int(payload.get("student_lora_alpha", 16) or 16),
            student_lora_dropout=float(payload.get("student_lora_dropout", 0.0) or 0.0),
            student_lora_block_start=int(payload.get("student_lora_block_start", 0) or 0),
            student_lora_chunk_size=int(payload.get("student_lora_chunk_size", 0) or 0),
            student_lora_merge_mode=str(payload.get("student_lora_merge_mode", "inplace")).strip().lower(),
            student_memory_efficient_modulation=_coerce_bool(
                payload.get("student_memory_efficient_modulation"),
                True,
            ),
            student_memory_efficient_checkpoint_mode=str(
                payload.get("student_memory_efficient_checkpoint_mode", "full")
            ).strip().lower(),
            student_ffn_chunk_size=int(payload.get("student_ffn_chunk_size", 4096) or 4096),
            student_norm_chunk_size=int(payload.get("student_norm_chunk_size", 0) or 0),
            student_precision_profile=student_precision_profile,
            student_low_precision_dtype=student_low_precision_dtype,
            gradient_checkpointing=_coerce_bool(payload.get("gradient_checkpointing"), True),
            student_checkpoint_use_reentrant=student_checkpoint_use_reentrant,
            student_ddp_find_unused_parameters=_coerce_bool(
                payload.get("student_ddp_find_unused_parameters"),
                True,
            ),
            distributed_timeout_hours=int(payload.get("distributed_timeout_hours", 8) or 8),
            videophy2_eval=VideoPhy2EvalConfig.from_payload(payload.get("videophy2_eval")),
        )
