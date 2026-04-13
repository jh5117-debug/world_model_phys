from physical_consistency.common.io import write_yaml
from physical_consistency.trainers.trd_v1 import build_args


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
    assert args.student_memory_efficient_modulation is True
    assert args.student_ffn_chunk_size == 2048


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
