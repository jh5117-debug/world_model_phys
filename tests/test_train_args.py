from physical_consistency.common.io import write_yaml
from physical_consistency.trainers.trd_v1 import build_args


class _CliArgs:
    config: str
    env_file: str
    experiment_name = ""
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
