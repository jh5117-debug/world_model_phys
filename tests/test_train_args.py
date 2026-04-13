from physical_consistency.common.io import write_yaml
from physical_consistency.trainers.trd_v1 import (
    build_args,
    format_eta,
    maybe_scalar_to_float,
    should_apply_student_gradient_checkpointing,
    student_gradient_checkpointing_use_reentrant,
    tensor_to_numpy_float32,
)
import torch
import numpy as np


class _CliArgs:
    config: str
    env_file: str
    experiment_name = ""
    project_name = ""
    wandb_entity = ""
    model_type = ""
    stage1_ckpt_dir = ""
    base_model_dir = ""
    dataset_dir = ""
    lingbot_code_dir = ""
    output_root = ""
    wandb_dir = ""
    teacher_repo_dir = ""
    teacher_checkpoint_dir = ""
    teacher_checkpoint_path = ""
    validation_runtime_mode = ""
    allow_deepspeed_feature_hook_experimental = ""

    def __init__(self, config: str, env_file: str) -> None:
        self.config = config
        self.env_file = env_file


def test_build_args_does_not_resolve_teacher_checkpoint_early(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_test",
            "model_type": "low",
            "teacher_checkpoint_dir": str(tmp_path / "missing_teacher_dir"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.experiment_name == "exp_test"
    assert args.teacher_checkpoint_path == ""


def test_build_args_supports_dual_training_defaults(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_dual",
            "model_type": "dual",
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.model_type == "dual"
    assert args.output_dir.endswith("checkpoints/exp_dual")
    assert args.teacher_dtype == "bfloat16"
    assert args.teacher_offload_after_encode is True
    assert args.student_tuning_mode == "lora"
    assert args.student_lora_rank == 16
    assert args.student_lora_alpha == 16
    assert args.student_lora_dropout == 0.0
    assert args.student_memory_efficient_modulation is True
    assert args.student_ffn_chunk_size == 512
    assert args.wandb_relation_image_every_steps == 25
    assert args.num_frames == 69


def test_build_args_accepts_wandb_entity_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_wandb",
            "model_type": "dual",
            "project_name": "intro-example",
            "wandb_entity": "WorldModel_11",
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.project_name == "intro-example"
    assert args.wandb_entity == "WorldModel_11"


def test_build_args_accepts_teacher_dtype_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_dtype",
            "model_type": "dual",
            "teacher_dtype": "float32",
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.teacher_dtype == "float32"


def test_build_args_accepts_teacher_offload_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_teacher_offload",
            "model_type": "dual",
            "teacher_offload_after_encode": "false",
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.teacher_offload_after_encode is False


def test_build_args_accepts_student_modulation_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_student_patch",
            "model_type": "dual",
            "student_memory_efficient_modulation": "false",
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.student_memory_efficient_modulation is False


def test_build_args_accepts_student_ffn_chunk_size_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_student_ffn_chunk",
            "model_type": "dual",
            "student_ffn_chunk_size": 2048,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.student_ffn_chunk_size == 2048


def test_build_args_accepts_num_frames_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_num_frames",
            "model_type": "dual",
            "num_frames": 81,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.num_frames == 81


def test_build_args_accepts_wandb_relation_image_every_steps_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_wandb_images",
            "model_type": "dual",
            "wandb_relation_image_every_steps": 10,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.wandb_relation_image_every_steps == 10


def test_build_args_accepts_student_lora_overrides(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_student_lora",
            "model_type": "dual",
            "student_tuning_mode": "full",
            "student_lora_rank": 8,
            "student_lora_alpha": 32,
            "student_lora_dropout": 0.1,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.student_tuning_mode == "full"
    assert args.student_lora_rank == 8
    assert args.student_lora_alpha == 32
    assert args.student_lora_dropout == 0.1


def test_should_apply_student_gradient_checkpointing_keeps_lora_mode_enabled():
    args = _CliArgs(config="", env_file="")
    args.gradient_checkpointing = True
    args.student_tuning_mode = "lora"
    assert should_apply_student_gradient_checkpointing(args) is True


def test_should_apply_student_gradient_checkpointing_keeps_full_mode():
    args = _CliArgs(config="", env_file="")
    args.gradient_checkpointing = True
    args.student_tuning_mode = "full"
    assert should_apply_student_gradient_checkpointing(args) is True


def test_student_gradient_checkpointing_use_reentrant_for_lora():
    args = _CliArgs(config="", env_file="")
    args.student_tuning_mode = "lora"
    assert student_gradient_checkpointing_use_reentrant(args) is True


def test_student_gradient_checkpointing_use_reentrant_off_for_full():
    args = _CliArgs(config="", env_file="")
    args.student_tuning_mode = "full"
    assert student_gradient_checkpointing_use_reentrant(args) is False


def test_format_eta_renders_compact_hours():
    assert format_eta(7265) == "2h01m"


def test_format_eta_handles_missing_values():
    assert format_eta(None) == "unknown"


def test_maybe_scalar_to_float_handles_none_and_tensor_inputs():
    assert maybe_scalar_to_float(None) is None
    assert maybe_scalar_to_float(torch.tensor(1.5)) == 1.5
    assert maybe_scalar_to_float(2) == 2.0


def test_tensor_to_numpy_float32_handles_bfloat16_tensors_and_numpy_inputs():
    tensor_result = tensor_to_numpy_float32(torch.ones(2, 2, dtype=torch.bfloat16))
    assert isinstance(tensor_result, np.ndarray)
    assert tensor_result.dtype == np.float32

    array_result = tensor_to_numpy_float32(np.ones((2, 2), dtype=np.float64))
    assert isinstance(array_result, np.ndarray)
    assert array_result.dtype == np.float32
