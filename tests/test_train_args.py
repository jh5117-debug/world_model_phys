from physical_consistency.common.io import read_yaml, write_yaml
from physical_consistency.trainers.stage1_components import compute_scheduler_total_steps
from physical_consistency.trainers import trd_v1 as trd_v1_module
from physical_consistency.trainers.trd_v1 import (
    TRDTrainingRunner,
    _build_training_state_payload_from_bundle_state,
    _resolve_teacher_checkpoint,
    build_args,
    format_eta,
    maybe_scalar_to_float,
    should_apply_student_gradient_checkpointing,
    student_gradient_checkpointing_use_reentrant,
    tensor_to_numpy_float32,
)
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace


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
    assert args.teacher_backend == "vjepa2"
    assert args.teacher_dtype == "bfloat16"
    assert args.teacher_offload_after_encode is True
    assert args.teacher_model_variant == "vjepa2_1_vit_base_384"
    assert args.teacher_input_frames == 64
    assert args.teacher_drop_first_frame is False
    assert args.teacher_image_size == 384
    assert args.student_tuning_mode == "lora"
    assert args.student_lora_rank == 16
    assert args.student_lora_alpha == 16
    assert args.student_lora_dropout == 0.0
    assert args.student_memory_efficient_modulation is True
    assert args.student_ffn_chunk_size == 512
    assert args.wandb_relation_image_every_steps == 25
    assert args.num_frames == 81
    assert args.validation_every_steps == 0
    assert args.validation_every_epochs == 1
    assert args.validation_sample_steps == 70


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


class _DummyAccumulation:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyAccelerator:
    def __init__(self) -> None:
        self.is_main_process = True
        self.num_processes = 1
        self.sync_gradients = True
        self.device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        return None

    def accumulate(self, model):
        del model
        return _DummyAccumulation()

    def backward(self, loss) -> None:
        del loss

    def clip_grad_norm_(self, params, max_norm):
        del params, max_norm
        return torch.tensor(0.0)

    def log(self, payload, step=None) -> None:
        del payload, step
        return None

    def unwrap_model(self, model):
        return model

    def end_training(self) -> None:
        return None


class _DummyBundle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_state_dicts(self) -> dict[str, dict]:
        return {}


def test_train_runs_validation_at_each_epoch_end(tmp_path, monkeypatch):
    runner = object.__new__(TRDTrainingRunner)
    runner.args = _CliArgs(config="", env_file="")
    runner.args.output_dir = str(tmp_path / "checkpoints" / "exp")
    runner.args.stage1_ckpt_dir = "links/stage1_epoch2"
    runner.args.model_type = "dual"
    runner.args.num_epochs = 2
    runner.args.validation_every_steps = 0
    runner.args.validation_every_epochs = 1
    runner.args.save_every_n_epochs = 0
    runner.args.max_grad_norm = 1.0
    runner.args.gradient_accumulation_steps = 4
    runner.args.best_checkpoint_name = "best_videophy2"
    runner.accelerator = _DummyAccelerator()
    runner.global_step = 0
    runner.current_epoch = 0
    runner.run = None
    runner.teacher_checkpoint_path = ""
    runner.train_loader = None
    runner.val_loader = None
    runner.model_bundle = None
    runner.optimizer = None
    runner.scheduler = None
    runner.best_metrics_path = tmp_path / "best_videophy2.json"
    runner.best_checkpoint_path = tmp_path / "best_videophy2"
    runner._micro_step = 0
    runner.micro_steps_per_epoch = 1
    runner.optimizer_steps_per_epoch = 1
    runner.total_optimizer_steps = 2
    runner._train_start_time = None
    runner._last_optimizer_step_time = None

    validation_tags: list[str] = []

    def _fake_build_loaders():
        runner.train_loader = [{"video": torch.zeros(1)}]
        runner.val_loader = []
        runner.train_dataset_len = 1
        runner.micro_steps_per_epoch = 1
        runner.optimizer_steps_per_epoch = 1
        runner.total_optimizer_steps = 2

    def _fake_initialize_runtime(*, checkpoint_dir, resume_state):
        del checkpoint_dir, resume_state
        runner.model_bundle = _DummyBundle()
        runner.optimizer = torch.optim.SGD(runner.model_bundle.parameters(), lr=0.1)
        runner.scheduler = torch.optim.lr_scheduler.LambdaLR(runner.optimizer, lambda _: 1.0)

    def _fake_training_step(batch):
        del batch
        return {
            "loss_total": torch.tensor(1.0, requires_grad=True),
            "loss_fm": torch.tensor(0.8),
            "loss_trd": torch.tensor(0.2),
            "loss_trd_spatial": torch.tensor(0.1),
            "loss_trd_temporal": torch.tensor(0.1),
            "sample_sigma": torch.tensor(0.9),
            "sample_timestep": torch.tensor(900.0),
            "teacher_feat_norm": torch.tensor(1.0),
            "student_feat_norm": torch.tensor(1.0),
            "pred_target_cosine": torch.tensor(0.95),
            "active_branch_is_high": torch.tensor(0.0),
        }

    monkeypatch.setattr(trd_v1_module, "_resolve_teacher_checkpoint", lambda *_: "teacher.pt")
    monkeypatch.setattr(
        trd_v1_module,
        "save_dual_bundle_checkpoint",
        lambda **kwargs: Path(kwargs["args"].output_dir) / str(kwargs["tag"]),
    )

    runner.validate_runtime_stack = lambda: None
    runner._build_train_and_val_loaders = _fake_build_loaders
    runner._log_train_plan = lambda: None
    runner._initialize_training_runtime = _fake_initialize_runtime
    runner._set_train_mode = lambda: None
    runner._reset_gpu_peak_memory_stats = lambda: None
    runner._log_gpu_memory = lambda *args, **kwargs: None
    runner.training_step = _fake_training_step
    runner._log_progress = lambda *args, **kwargs: None
    runner._log_train_metrics = lambda *args, **kwargs: None
    runner._log_epoch_summary = lambda *args, **kwargs: None
    runner._write_lineage = lambda *args, **kwargs: None
    runner._write_pending_eval_commands = lambda *args, **kwargs: None
    runner.run_validation_cycle = lambda tag: validation_tags.append(tag)

    runner.train()

    assert validation_tags == ["epoch_1", "epoch_2"]


def test_build_args_accepts_teacher_backend_and_validation_epoch_overrides(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_teacher_backend",
            "model_type": "dual",
            "teacher_backend": "VideoMAEv2",
            "teacher_drop_first_frame": "true",
            "validation_every_epochs": 2,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.teacher_backend == "videomaev2"
    assert args.teacher_drop_first_frame is True
    assert args.validation_every_epochs == 2


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


def test_build_args_defaults_validation_sample_steps_to_sample_steps(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_val_sample_steps_default",
            "model_type": "dual",
            "sample_steps": 42,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.validation_sample_steps == 42


def test_build_args_accepts_validation_sample_steps_override(tmp_path):
    config_path = tmp_path / "train.yaml"
    env_path = tmp_path / "paths.env"
    write_yaml(
        config_path,
        {
            "experiment_name": "exp_val_sample_steps_override",
            "model_type": "dual",
            "sample_steps": 70,
            "validation_sample_steps": 35,
            "teacher_checkpoint_dir": str(tmp_path / "teacher"),
        },
    )
    env_path.write_text("", encoding="utf-8")

    args = build_args(_CliArgs(str(config_path), str(env_path)))
    assert args.sample_steps == 70
    assert args.validation_sample_steps == 35


def test_validation_generation_command_uses_validation_sample_steps(tmp_path):
    runner = object.__new__(TRDTrainingRunner)
    runner.args = SimpleNamespace(
        env_file="paths.env",
        manifest_mini_val="mini.csv",
        dataset_dir="dataset",
        base_model_dir="base",
        num_frames=81,
        validation_sample_steps=35,
        guide_scale=5.0,
        height=480,
        width=832,
        num_gpus=4,
        ulysses_size=4,
    )

    command = runner._validation_generation_command(
        tmp_path / "checkpoint",
        tmp_path / "generated",
        validation_seed=42,
    )

    assert command[command.index("--sample_steps") + 1] == "35"


def test_snapshot_validation_export_uses_validation_sample_steps(tmp_path, monkeypatch):
    runner = object.__new__(TRDTrainingRunner)
    runner.args = SimpleNamespace(
        output_root=str(tmp_path),
        experiment_name="exp",
        stage1_ckpt_dir=str(tmp_path / "stage1"),
        manifest_mini_val="mini.csv",
        validation_seed_list=[42],
        num_frames=81,
        sample_steps=70,
        validation_sample_steps=35,
        guide_scale=5.0,
        height=480,
        width=832,
        num_gpus=4,
        ulysses_size=4,
        base_model_dir="base",
        dataset_dir="dataset",
    )
    runner.global_step = 123
    runner.accelerator = _DummyAccelerator()
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    bundle_dir = tmp_path / "bundle"

    monkeypatch.setattr(
        trd_v1_module,
        "materialize_eval_checkpoint_bundle",
        lambda **kwargs: bundle_dir,
    )

    runner._export_validation_request(checkpoint_path, "epoch_1")

    eval_config = read_yaml(checkpoint_path / "validation_export" / "eval_trd_snapshot.yaml")
    assert eval_config["sample_steps"] == 35


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


def test_resolve_teacher_checkpoint_accepts_pt_files(tmp_path):
    checkpoint_dir = tmp_path / "teacher"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "vjepa2_1_vitb_dist_vitG_384.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    assert _resolve_teacher_checkpoint(str(checkpoint_dir)) == str(checkpoint_path)


def test_compute_scheduler_total_steps_uses_ceil_for_tail_accumulation():
    assert compute_scheduler_total_steps(dataset_len=1670, num_processes=8, grad_accum=4, num_epochs=5) == 265


def test_build_training_state_payload_from_bundle_state_uses_full_lora_tensors(tmp_path):
    stage1_dir = tmp_path / "stage1_ckpt"
    stage1_dir.mkdir()
    bundle_state = {
        "low_projector.proj.weight": torch.randn(768, 16),
        "low_projector.proj.bias": torch.randn(768),
        "high_projector.proj.weight": torch.randn(768, 16),
        "high_projector.proj.bias": torch.randn(768),
        "low_model.blocks.0.self_attn.q.lora_A.weight": torch.randn(16, 5120),
        "low_model.blocks.0.self_attn.q.lora_B.weight": torch.randn(5120, 16),
        "high_model.blocks.1.self_attn.k.lora_A.weight": torch.randn(16, 5120),
        "high_model.blocks.1.self_attn.k.lora_B.weight": torch.randn(5120, 16),
        "low_model.blocks.0.self_attn.q.weight": torch.randn(5120, 5120),
    }

    payload = _build_training_state_payload_from_bundle_state(
        bundle_state=bundle_state,
        student_tuning_mode="lora",
        global_step=105,
        epoch=1,
        tag="epoch_1",
        optimizer_state={"state": {"dummy": 1}},
        scheduler_state={"last_epoch": 105},
        student_base_checkpoint_dir=stage1_dir,
    )

    assert payload["global_step"] == 105
    assert payload["epoch"] == 1
    assert payload["tag"] == "epoch_1"
    assert payload["optimizer"] == {"state": {"dummy": 1}}
    assert payload["scheduler"] == {"last_epoch": 105}
    assert payload["student_base_checkpoint_dir"] == str(stage1_dir.resolve())
    assert payload["low_projector"]["proj.weight"].shape == (768, 16)
    assert payload["high_projector"]["proj.bias"].shape == (768,)
    assert set(payload["low_lora"]) == {
        "blocks.0.self_attn.q.lora_A.weight",
        "blocks.0.self_attn.q.lora_B.weight",
    }
    assert set(payload["high_lora"]) == {
        "blocks.1.self_attn.k.lora_A.weight",
        "blocks.1.self_attn.k.lora_B.weight",
    }
    assert payload["low_lora"]["blocks.0.self_attn.q.lora_A.weight"].shape == (16, 5120)
    assert payload["high_lora"]["blocks.1.self_attn.k.lora_B.weight"].shape == (5120, 16)
    assert all(tensor.numel() > 0 for tensor in payload["low_lora"].values())
    assert all(tensor.numel() > 0 for tensor in payload["high_lora"].values())


def test_gather_optimizer_resume_state_collects_all_deepspeed_ranks(monkeypatch):
    runner = object.__new__(TRDTrainingRunner)
    runner.accelerator = _DummyAccelerator()
    runner.accelerator.distributed_type = trd_v1_module.DistributedType.DEEPSPEED
    runner.accelerator.num_processes = 4
    runner.accelerator.is_main_process = True
    runner.optimizer = type(
        "_DummyOptimizer",
        (),
        {"state_dict": lambda self: {"rank": 0, "state": {"dummy": 1}}},
    )()

    monkeypatch.setattr(trd_v1_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trd_v1_module.dist, "is_initialized", lambda: True)

    def _fake_gather_object(obj, object_gather_list, dst):
        del dst
        object_gather_list[:] = [
            obj,
            {"rank": 1, "state": {"dummy": 2}},
            {"rank": 2, "state": {"dummy": 3}},
            {"rank": 3, "state": {"dummy": 4}},
        ]

    monkeypatch.setattr(trd_v1_module.dist, "gather_object", _fake_gather_object)

    gathered = runner._gather_optimizer_resume_state()

    assert isinstance(gathered, list)
    assert [item["rank"] for item in gathered] == [0, 1, 2, 3]
