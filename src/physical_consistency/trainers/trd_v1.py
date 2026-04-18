"""Standalone TRD-v1 trainer that does not modify the shared Stage-1 code."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import logging
import math
import os
import re
import time
import traceback
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import DistributedType
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)

from physical_consistency.common.defaults import CONFIG_DIR, PROJECT_ROOT
from physical_consistency.common.io import ensure_dir, read_json, read_yaml, write_json, write_yaml
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.seed import set_seed
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.common.summary_tables import format_videophy2_summary
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.lineage.contract import LineageRecord, verify_stage1_checkpoint
from physical_consistency.losses.trd import TRDLossOutput, TokenRelationDistillationLoss
from physical_consistency.teachers.videomaev2 import VideoMAEv2Teacher
from physical_consistency.teachers.vjepa2 import VJEPA21Teacher
from physical_consistency.trainers.hooks import BlockFeatureHook
from physical_consistency.trainers.stage1_components import (
    MODEL_SUBFOLDERS,
    LingBotStage1Helper,
    apply_gradient_checkpointing,
    build_dataloader,
    collect_lora_local_loss,
    collect_lora_parameter_loss,
    collect_lora_synthetic_matmul_loss,
    compute_scheduler_total_steps,
    export_pretrained_state_dict,
    extract_lora_state_dict,
    get_model_subfolder,
    load_lora_state_dict,
    move_optimizer_state,
    prune_checkpoint_dir,
)
from physical_consistency.wandb_utils.media import relation_matrix_image
from physical_consistency.wandb_utils.session import init_wandb_run, log_dict

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_cuda_probe_flags() -> None:
    """Apply process-wide CUDA debug toggles before model construction."""

    if not _env_flag("PC_DISABLE_TF32"):
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        LOGGER.debug("Failed to set float32 matmul precision", exc_info=True)
    LOGGER.info(
        "PC_DISABLE_TF32=1: disabled CUDA TF32 matmul/cudnn paths "
        "(matmul.allow_tf32=%s, cudnn.allow_tf32=%s)",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def should_apply_student_gradient_checkpointing(args: argparse.Namespace) -> bool:
    """Return whether block-level student checkpointing should be enabled."""
    if not getattr(args, "gradient_checkpointing", False):
        return False
    return True


def student_gradient_checkpointing_use_reentrant(
    args: argparse.Namespace,
    distributed_type: DistributedType | None = None,
) -> bool:
    """Return the checkpoint engine choice for the student blocks.

    Reentrant checkpointing can avoid some DeepSpeed metadata edge cases, but
    DDP cannot safely handle LoRA parameters reused inside reentrant checkpoint
    replays without static-graph handling. Prefer non-reentrant under DDP unless
    the caller explicitly overrides the setting.
    """
    override = getattr(args, "student_checkpoint_use_reentrant", None)
    if override is not None:
        return bool(override)
    if (
        distributed_type == DistributedType.MULTI_GPU
        and getattr(args, "student_tuning_mode", "full") == "lora"
    ):
        return False
    return getattr(args, "student_tuning_mode", "full") == "lora"


def maybe_scalar_to_float(value: torch.Tensor | float | int | None) -> float | None:
    """Convert scalar-like tensors and Python numbers to float while preserving missing values."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _safe_call_runtime_method(obj: Any, method_name: str, label: str) -> None:
    """Best-effort cleanup for optional DeepSpeed/optimizer teardown hooks."""
    if obj is None:
        return
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:
        LOGGER.debug("Failed to call %s.%s during runtime release", label, method_name, exc_info=True)


def _safe_clear_attr(obj: Any, attr_name: str, label: str) -> None:
    """Break common reference cycles without depending on a specific DeepSpeed version."""
    if obj is None or not hasattr(obj, attr_name):
        return
    try:
        setattr(obj, attr_name, None)
    except Exception:
        LOGGER.debug("Failed to clear %s.%s during runtime release", label, attr_name, exc_info=True)


def _candidate_deepspeed_engines(accelerator: accelerate.Accelerator, model_bundle: Any) -> list[Any]:
    """Collect DeepSpeed engine objects that may hold old ZeRO partitions."""
    candidates: list[Any] = []
    wrapped = getattr(accelerator, "deepspeed_engine_wrapped", None)
    candidates.append(getattr(wrapped, "engine", None))
    candidates.append(model_bundle)
    candidates.extend(getattr(accelerator, "_models", []) or [])

    engines: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        engine = getattr(candidate, "engine", candidate)
        if engine is None:
            continue
        if not (
            hasattr(engine, "destroy")
            or hasattr(engine, "optimizer")
            or engine.__class__.__name__.lower().startswith("deepspeed")
        ):
            continue
        ident = id(engine)
        if ident in seen:
            continue
        seen.add(ident)
        engines.append(engine)
    return engines


def _destroy_deepspeed_engines(accelerator: accelerate.Accelerator, model_bundle: Any) -> None:
    """Aggressively release DeepSpeed engine references before rebuilding a runtime."""
    for engine in _candidate_deepspeed_engines(accelerator, model_bundle):
        optimizer = getattr(engine, "optimizer", None)
        _safe_call_runtime_method(optimizer, "release_ipg_buffers", "deepspeed_optimizer")
        _safe_call_runtime_method(optimizer, "destroy", "deepspeed_optimizer")
        _safe_call_runtime_method(engine, "destroy", "deepspeed_engine")
        for attr_name in (
            "optimizer",
            "client_optimizer",
            "lr_scheduler",
            "module",
            "module_parameters",
            "fp16_groups",
            "fp32_groups",
        ):
            _safe_clear_attr(engine, attr_name, "deepspeed_engine")

    if hasattr(accelerator, "deepspeed_engine_wrapped"):
        accelerator.deepspeed_engine_wrapped = None
    for attr_name in ("_models", "_optimizers", "_schedulers", "_dataloaders"):
        if hasattr(accelerator, attr_name):
            setattr(accelerator, attr_name, [])


def _clear_torch_cuda_cache() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        LOGGER.debug("torch.cuda.ipc_collect failed during runtime release", exc_info=True)


def tensor_to_numpy_float32(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensors to float32 numpy arrays for logging utilities that don't support bf16."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return value.detach().float().cpu().numpy()


_STEP_LABEL_PATTERN = re.compile(r"^step_(\d+)_(.+)$")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _wandb_disabled(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "wandb_mode", "") or os.environ.get("WANDB_MODE", "")).strip().lower()
    env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    return _env_flag("PC_DISABLE_WANDB") or mode in {"disabled", "disable", "off", "none"} or env_mode in {
        "disabled",
        "disable",
        "off",
        "none",
    }


def format_eta(seconds: float | None) -> str:
    """Render a compact ETA string for progress logs."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _build_teacher_encoder(
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    device: torch.device,
):
    """Instantiate the configured frozen teacher with minimal trainer branching."""
    teacher_backend = str(getattr(args, "teacher_backend", "external")).strip().lower()
    if teacher_backend in {"external", "videomaev2", "videomae", "videorepa"}:
        return VideoMAEv2Teacher(
            repo_dir=args.teacher_repo_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            model_dtype=args.teacher_dtype,
            offload_after_encode=args.teacher_offload_after_encode,
            model_variant=args.teacher_model_variant,
            image_size=args.teacher_image_size,
            align_video_resolution=(args.teacher_height, args.teacher_width),
            pretrained_num_frames=args.teacher_pretrained_frames,
            teacher_input_frames=args.teacher_input_frames,
            drop_first_frame=args.teacher_drop_first_frame,
        )
    if teacher_backend in {"vjepa2", "v-jepa2", "vjepa2.1", "v-jepa-2.1", "vjepa_2_1"}:
        return VJEPA21Teacher(
            repo_dir=args.teacher_repo_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            model_dtype=args.teacher_dtype,
            offload_after_encode=args.teacher_offload_after_encode,
            model_variant=args.teacher_model_variant,
            image_size=args.teacher_image_size,
            teacher_input_frames=args.teacher_input_frames,
            drop_first_frame=args.teacher_drop_first_frame,
        )
    raise ValueError(f"Unsupported teacher_backend: {args.teacher_backend}")


class StudentProjector(nn.Module):
    """Project student tokens to the teacher feature dimension."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class DualBranchTrainingBundle(nn.Module):
    """Wrap both Stage-1 branches so ZeRO-3 sees one trainable model."""

    def __init__(
        self,
        *,
        low_model: nn.Module,
        high_model: nn.Module,
        low_projector: nn.Module,
        high_projector: nn.Module,
        student_target_block: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.low_model = low_model
        self.high_model = high_model
        self.low_projector = low_projector
        self.high_projector = high_projector
        self.patch_size = patch_size

        self.low_hook = BlockFeatureHook()
        self.high_hook = BlockFeatureHook()
        self.low_hook.attach(self.low_model.blocks[student_target_block])
        self.high_hook.attach(self.high_model.blocks[student_target_block])

    def forward(
        self,
        *,
        branch: str,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        y: torch.Tensor,
        dit_cond: dict[str, tuple[torch.Tensor, ...]],
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_model, active_projector, active_hook = self._active_components(branch)
        self.low_hook.clear()
        self.high_hook.clear()

        pred = active_model(
            [noisy_latent],
            t=timestep,
            context=context,
            seq_len=seq_len,
            y=[y],
            dit_cond_dict=dit_cond,
        )[0]
        if active_hook.latest is None:
            raise RuntimeError(f"Feature hook did not capture student tokens for branch={branch}")

        student_tokens = self._reshape_student_tokens(active_hook.latest, lat_f, lat_h, lat_w)
        student_tokens = active_projector(student_tokens)
        return pred, student_tokens

    def projector_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "low_projector": self.low_projector.state_dict(),
            "high_projector": self.high_projector.state_dict(),
        }

    def lora_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "low_lora": extract_lora_state_dict(self.low_model),
            "high_lora": extract_lora_state_dict(self.high_model),
        }

    def training_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        payload = self.projector_state_dicts()
        low_lora = extract_lora_state_dict(self.low_model)
        high_lora = extract_lora_state_dict(self.high_model)
        if low_lora or high_lora:
            payload.update(
                {
                    "low_lora": low_lora,
                    "high_lora": high_lora,
                }
            )
        return payload

    def _active_components(self, branch: str) -> tuple[nn.Module, nn.Module, BlockFeatureHook]:
        if branch == "high":
            return self.high_model, self.high_projector, self.high_hook
        return self.low_model, self.low_projector, self.low_hook

    def _reshape_student_tokens(
        self,
        tokens: torch.Tensor,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        channels = tokens.shape[-1]
        pooled_h = lat_h // self.patch_size[1]
        pooled_w = lat_w // self.patch_size[2]
        return tokens.view(1, lat_f, pooled_h * pooled_w, channels)


class SingleBranchTrainingBundle(nn.Module):
    """Wrap one Stage-1 branch for memory-probed single-branch training."""

    def __init__(
        self,
        *,
        branch: str,
        model: nn.Module,
        projector: nn.Module,
        student_target_block: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if branch not in {"low", "high"}:
            raise ValueError(f"Unsupported single branch: {branch}")
        self.branch = branch
        self.model = model
        self.projector = projector
        self.patch_size = patch_size

        self.hook = BlockFeatureHook()
        self.hook.attach(self.model.blocks[student_target_block])

    def forward(
        self,
        *,
        branch: str,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        y: torch.Tensor,
        dit_cond: dict[str, tuple[torch.Tensor, ...]],
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if branch != self.branch:
            raise RuntimeError(f"Single-branch bundle for {self.branch} received branch={branch}")
        self.hook.clear()

        pred = self.model(
            [noisy_latent],
            t=timestep,
            context=context,
            seq_len=seq_len,
            y=[y],
            dit_cond_dict=dit_cond,
        )[0]
        if self.hook.latest is None:
            raise RuntimeError(f"Feature hook did not capture student tokens for branch={branch}")

        student_tokens = self._reshape_student_tokens(self.hook.latest, lat_f, lat_h, lat_w)
        student_tokens = self.projector(student_tokens)
        return pred, student_tokens

    def projector_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {f"{self.branch}_projector": self.projector.state_dict()}

    def lora_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {f"{self.branch}_lora": extract_lora_state_dict(self.model)}

    def training_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        payload = self.projector_state_dicts()
        lora = extract_lora_state_dict(self.model)
        if lora:
            payload[f"{self.branch}_lora"] = lora
        return payload

    def _reshape_student_tokens(
        self,
        tokens: torch.Tensor,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        channels = tokens.shape[-1]
        pooled_h = lat_h // self.patch_size[1]
        pooled_w = lat_w // self.patch_size[2]
        return tokens.view(1, lat_f, pooled_h * pooled_w, channels)


def _extract_prefixed_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    *,
    key_filter: Callable[[str], bool] | None = None,
    required: bool = True,
) -> dict[str, torch.Tensor]:
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped_key = key[len(prefix) :]
            if key_filter is not None and not key_filter(stripped_key):
                continue
            extracted[stripped_key] = value.cpu()
    if required and not extracted:
        raise KeyError(f"No state_dict entries matched prefix={prefix!r}")
    return extracted


def _is_lora_adapter_key(key: str) -> bool:
    return ".lora_A.weight" in key or ".lora_B.weight" in key


def _build_training_state_payload_from_bundle_state(
    *,
    bundle_state: dict[str, torch.Tensor],
    student_tuning_mode: str,
    global_step: int,
    epoch: int,
    tag: str | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    student_base_checkpoint_dir: str | Path | None = None,
    model_type: str = "dual",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "global_step": global_step,
        "epoch": epoch,
    }
    branch_prefixes = (
        {"low": "low_projector.", "high": "high_projector."}
        if model_type == "dual"
        else {model_type: "projector."}
    )
    for branch, prefix in branch_prefixes.items():
        payload[f"{branch}_projector"] = _extract_prefixed_state_dict(bundle_state, prefix)
    if tag is not None:
        payload["tag"] = tag
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler"] = scheduler_state

    if student_tuning_mode == "lora":
        model_prefixes = (
            {"low": "low_model.", "high": "high_model."}
            if model_type == "dual"
            else {model_type: "model."}
        )
        for branch, prefix in model_prefixes.items():
            lora = _extract_prefixed_state_dict(
                bundle_state,
                prefix,
                key_filter=_is_lora_adapter_key,
                required=False,
            )
            if lora:
                payload[f"{branch}_lora"] = lora
        if student_base_checkpoint_dir is not None:
            payload["student_base_checkpoint_dir"] = str(Path(student_base_checkpoint_dir).resolve())

    return payload


def _save_branch_checkpoint(
    *,
    branch_model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    model_dir: Path,
) -> None:
    ensure_dir(model_dir)
    torch.save(
        export_pretrained_state_dict(branch_model, state_dict, prefix=prefix),
        model_dir / "diffusion_pytorch_model.bin",
    )
    if hasattr(branch_model, "save_config"):
        branch_model.save_config(model_dir)


def save_dual_bundle_checkpoint(
    *,
    accelerator,
    model_bundle: nn.Module,
    args,
    tag: str,
    extra_training_state: dict[str, Any] | Callable[[dict[str, torch.Tensor], nn.Module], dict[str, Any] | None] | None = None,
    training_state_filename: str = "resume_state.pt",
) -> Path:
    """Save a wrapped TRD checkpoint from one dual- or single-branch container."""
    save_dir = Path(args.output_dir) / tag
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    bundle_state = accelerator.get_state_dict(model_bundle)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if bundle_state is None:
            raise RuntimeError("Accelerator returned no state_dict for the TRD bundle on the main process")
        unwrapped = accelerator.unwrap_model(model_bundle)
        resolved_training_state = extra_training_state
        if callable(extra_training_state):
            resolved_training_state = extra_training_state(bundle_state, unwrapped)
        if hasattr(unwrapped, "low_model") and hasattr(unwrapped, "high_model"):
            _save_branch_checkpoint(
                branch_model=unwrapped.low_model,
                state_dict=bundle_state,
                prefix="low_model.",
                model_dir=save_dir / "low_noise_model",
            )
            _save_branch_checkpoint(
                branch_model=unwrapped.high_model,
                state_dict=bundle_state,
                prefix="high_model.",
                model_dir=save_dir / "high_noise_model",
            )
        elif hasattr(unwrapped, "model") and hasattr(unwrapped, "branch"):
            _save_branch_checkpoint(
                branch_model=unwrapped.model,
                state_dict=bundle_state,
                prefix="model.",
                model_dir=save_dir / get_model_subfolder(unwrapped.branch),
            )
        else:
            raise RuntimeError(f"Unsupported TRD bundle type for saving: {type(unwrapped)!r}")
        if resolved_training_state is not None:
            training_only = save_dir / "training_only"
            training_only.mkdir(parents=True, exist_ok=True)
            torch.save(resolved_training_state, training_only / training_state_filename)
    accelerator.wait_for_everyone()
    return save_dir


class TRDTrainingRunner:
    """End-to-end trainer for dual-branch Stage-1 continuation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.disable_wandb = _wandb_disabled(args)
        project_config = ProjectConfiguration(
            project_dir=str(Path(args.output_dir)),
            logging_dir=str(Path(args.output_root) / "logs" / "wandb"),
        )
        ddp_kwargs_payload: dict[str, Any] = {
            "find_unused_parameters": bool(args.student_ddp_find_unused_parameters),
        }
        if "static_graph" in inspect.signature(DistributedDataParallelKwargs).parameters:
            ddp_kwargs_payload["static_graph"] = bool(args.student_ddp_static_graph)
        elif bool(args.student_ddp_static_graph):
            LOGGER.warning(
                "student_ddp_static_graph=true was requested, but this accelerate version "
                "does not expose DistributedDataParallelKwargs.static_graph"
            )
        ddp_kwargs = DistributedDataParallelKwargs(**ddp_kwargs_payload)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl",
            timeout=timedelta(hours=args.distributed_timeout_hours),
        )
        grad_accum_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_plugin=grad_accum_plugin,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            log_with=None if self.disable_wandb else "wandb",
            project_config=project_config,
        )
        self.helper = LingBotStage1Helper(args)
        self.trd_loss = TokenRelationDistillationLoss(
            relation_tokens=args.relation_tokens,
            margin=args.margin,
            lambda_spatial=args.lambda_spatial,
            lambda_temporal=args.lambda_temporal,
        )
        self.global_step = 0
        self.current_epoch = 0
        self.run = None
        self._tracking_initialized = False
        self.teacher_checkpoint_path = ""
        self.train_dataset_len = 0
        self.train_loader_raw = None
        self.train_loader = None
        self.val_loader = None
        self.model_bundle = None
        self.teacher = None
        self._param_grad_trace_handles: list[Any] = []
        self._param_grad_trace_counts: dict[str, int] = {}
        self.best_metrics_path = Path(args.output_dir) / "best_videophy2.json"
        self.best_checkpoint_path = Path(args.output_dir) / args.best_checkpoint_name
        self.visible_gpu_list = os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            ",".join(str(idx) for idx in range(args.num_gpus)),
        )
        self.best_metrics: dict[str, float] | None = None
        self._logged_teacher_encode_offload_memory = False
        self._logged_student_sequence_geometry = False
        self._logged_train_plan = False
        self._micro_step = 0
        self.micro_steps_per_epoch = 0
        self.optimizer_steps_per_epoch = 0
        self.total_optimizer_steps = 0
        self._train_start_time: float | None = None
        self._last_optimizer_step_time: float | None = None

    def _should_trace_training_phase(self) -> bool:
        return self._micro_step <= 1 or _env_flag("PC_TRAIN_PHASE_TRACE")

    def _phase_trace_rank_allowed(self) -> bool:
        return self.accelerator.is_main_process or _env_flag("PC_TRAIN_PHASE_TRACE_ALL_RANKS")

    def _trace_training_phase(
        self,
        label: str,
        *,
        tensors: dict[str, torch.Tensor | None] | None = None,
        scalars: dict[str, torch.Tensor | float | int | None] | None = None,
    ) -> None:
        if not self._should_trace_training_phase() or not self._phase_trace_rank_allowed():
            return

        device = self.accelerator.device
        if torch.cuda.is_available() and device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                LOGGER.exception("[PHASE] cuda synchronize failed at %s", label)
                raise

        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        parts = [
            f"label={label}",
            f"rank={rank}",
            f"local_rank={local_rank}",
            f"epoch={self.current_epoch}",
            f"micro_step={self._micro_step}",
            f"global_step={self.global_step}",
        ]
        if torch.cuda.is_available() and device.type == "cuda":
            parts.extend(
                [
                    f"allocated={torch.cuda.memory_allocated(device) / (1024**3):.2f}GiB",
                    f"reserved={torch.cuda.memory_reserved(device) / (1024**3):.2f}GiB",
                    f"max_allocated={torch.cuda.max_memory_allocated(device) / (1024**3):.2f}GiB",
                ]
            )

        if tensors:
            for name, tensor in tensors.items():
                if tensor is None:
                    parts.append(f"{name}=None")
                    continue
                if not torch.is_tensor(tensor):
                    parts.append(f"{name}=<non_tensor>")
                    continue
                parts.append(
                    f"{name}=shape{tuple(tensor.shape)} dtype={tensor.dtype} "
                    f"device={tensor.device} requires_grad={tensor.requires_grad}"
                )

        if scalars:
            for name, value in scalars.items():
                if value is None:
                    parts.append(f"{name}=None")
                    continue
                try:
                    if torch.is_tensor(value):
                        scalar = float(value.detach().float().item())
                    else:
                        scalar = float(value)
                    parts.append(f"{name}={scalar:.6g}")
                    parts.append(f"{name}_finite={math.isfinite(scalar)}")
                except Exception:
                    parts.append(f"{name}=<unavailable>")

        LOGGER.info("[PHASE] %s", " ".join(parts))

    def _student_checkpoint_use_reentrant(self) -> bool:
        use_reentrant = student_gradient_checkpointing_use_reentrant(
            self.args,
            self.accelerator.distributed_type,
        )
        override = getattr(self.args, "student_checkpoint_use_reentrant", None)
        is_ddp_lora = (
            self.accelerator.distributed_type == DistributedType.MULTI_GPU
            and getattr(self.args, "student_tuning_mode", "full") == "lora"
        )
        if self.accelerator.is_main_process and is_ddp_lora and should_apply_student_gradient_checkpointing(self.args):
            if override is None and not use_reentrant:
                LOGGER.warning(
                    "[DDP DIAG] Auto-selected non-reentrant checkpointing for DDP+LoRA. "
                    "Reentrant checkpointing can mark LoRA parameters ready twice in DDP. "
                    "Use --student_checkpoint_use_reentrant true only with --student_ddp_static_graph true "
                    "or for a targeted diagnostic."
                )
            elif use_reentrant:
                LOGGER.warning(
                    "[DDP DIAG] DDP+LoRA is using reentrant checkpointing. If backward fails with "
                    "'Expected to mark a variable ready only once', rerun with "
                    "--student_checkpoint_use_reentrant false or --student_ddp_static_graph true."
                )
        return use_reentrant

    def _iter_student_branch_models(self) -> list[tuple[str, nn.Module]]:
        bundle = self.model_bundle
        if isinstance(bundle, DualBranchTrainingBundle):
            return [("low", bundle.low_model), ("high", bundle.high_model)]
        if isinstance(bundle, SingleBranchTrainingBundle):
            return [(bundle.branch, bundle.model)]
        return []

    def _named_parameters_including_duplicates(self, module: nn.Module) -> list[tuple[str, nn.Parameter]]:
        try:
            return list(module.named_parameters(remove_duplicate=False))
        except TypeError:
            return list(module.named_parameters())

    def _format_counter(self, counter: Counter) -> str:
        if not counter:
            return "none"
        return ",".join(f"{key}:{value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))

    def _log_student_graph_diagnostics(self, label: str) -> None:
        if not getattr(self.args, "student_diagnose_training_graph", True):
            return
        if not self.accelerator.is_main_process:
            return
        use_reentrant = student_gradient_checkpointing_use_reentrant(
            self.args,
            self.accelerator.distributed_type,
        )
        LOGGER.info(
            "[STUDENT DIAG] label=%s distributed_type=%s ddp_find_unused_parameters=%s "
            "ddp_static_graph=%s gradient_checkpointing=%s checkpoint_use_reentrant=%s "
            "checkpoint_mode=%s tuning_mode=%s lora_merge_mode=%s",
            label,
            self.accelerator.distributed_type,
            self.args.student_ddp_find_unused_parameters,
            self.args.student_ddp_static_graph,
            self.args.gradient_checkpointing,
            use_reentrant,
            self.args.student_memory_efficient_checkpoint_mode,
            self.args.student_tuning_mode,
            getattr(self.args, "student_lora_merge_mode", "inplace"),
        )
        for branch, model in self._iter_student_branch_models():
            block_container = getattr(model, "blocks", [])
            checkpoint_modes = Counter(
                getattr(block, "_pc_checkpoint_mode", "unpatched") for block in block_container
            )
            lora_merge_modes: Counter = Counter()
            lora_modules = 0
            for module in model.modules():
                if module.__class__.__name__ == "LoRALinear":
                    lora_modules += 1
                    lora_merge_modes[getattr(module, "merge_mode", "unknown")] += 1
            named_params = self._named_parameters_including_duplicates(model)
            trainable_tensors = [(name, param) for name, param in named_params if param.requires_grad]
            trainable_param_count = sum(param.numel() for _, param in trainable_tensors)
            seen_param_ids: dict[int, str] = {}
            duplicate_names: list[str] = []
            for name, param in named_params:
                param_id = id(param)
                if param_id in seen_param_ids:
                    duplicate_names.append(f"{seen_param_ids[param_id]}=={name}")
                else:
                    seen_param_ids[param_id] = name
            LOGGER.info(
                "[STUDENT DIAG] branch=%s blocks=%s checkpoint_modes=%s lora_modules=%s "
                "lora_merge_modes=%s trainable_tensors=%s trainable_params=%s duplicate_parameter_refs=%s",
                branch,
                len(block_container),
                self._format_counter(checkpoint_modes),
                lora_modules,
                self._format_counter(lora_merge_modes),
                len(trainable_tensors),
                trainable_param_count,
                len(duplicate_names),
            )
            if duplicate_names:
                LOGGER.warning(
                    "[STUDENT DIAG] branch=%s duplicate_parameter_examples=%s",
                    branch,
                    ";".join(duplicate_names[:8]),
                )
        if (
            self.accelerator.distributed_type == DistributedType.MULTI_GPU
            and self.args.student_tuning_mode == "lora"
            and self.args.gradient_checkpointing
            and use_reentrant
            and not self.args.student_ddp_static_graph
        ):
            LOGGER.warning(
                "[STUDENT DIAG] Risk detected: DDP+LoRA+reentrant checkpointing without static_graph. "
                "This matches PyTorch's 'marked ready twice' failure mode."
            )

    def _install_param_grad_trace(self) -> None:
        if not getattr(self.args, "student_param_grad_trace", False):
            return
        if self.model_bundle is None:
            return
        if self._param_grad_trace_handles:
            return
        pattern_text = str(getattr(self.args, "student_param_grad_trace_pattern", "lora")).strip().lower()
        patterns = [item.strip() for item in pattern_text.split(",") if item.strip()]
        trace_all = not patterns or "*" in patterns

        def _matches(name: str) -> bool:
            lowered = name.lower()
            return trace_all or any(pattern in lowered for pattern in patterns)

        watched = 0
        for name, param in self.model_bundle.named_parameters():
            if not param.requires_grad or not _matches(name):
                continue
            watched += 1

            def _make_hook(param_name: str):
                def _hook(grad: torch.Tensor):
                    count = self._param_grad_trace_counts.get(param_name, 0) + 1
                    self._param_grad_trace_counts[param_name] = count
                    if count > 1:
                        LOGGER.warning(
                            "[PARAM GRAD TRACE] duplicate_grad_hook rank=%s local_rank=%s "
                            "micro_step=%s global_step=%s count=%s name=%s grad_shape=%s grad_dtype=%s",
                            os.environ.get("RANK", "?"),
                            os.environ.get("LOCAL_RANK", "?"),
                            self._micro_step,
                            self.global_step,
                            count,
                            param_name,
                            tuple(grad.shape),
                            grad.dtype,
                        )
                    return grad

                return _hook

            self._param_grad_trace_handles.append(param.register_hook(_make_hook(name)))
        if self.accelerator.is_main_process:
            LOGGER.info(
                "[PARAM GRAD TRACE] installed hooks watched_tensors=%s pattern=%s",
                watched,
                pattern_text or "*",
            )

    def _reset_param_grad_trace_counts(self) -> None:
        counts = getattr(self, "_param_grad_trace_counts", None)
        if counts:
            counts.clear()

    def initialize_tracking(self) -> None:
        """Create the shared W&B run before model loading."""
        if self._tracking_initialized:
            return
        if self.disable_wandb:
            if self.accelerator.is_main_process:
                LOGGER.info(
                    "PC_DISABLE_WANDB/WANDB_MODE disabled: skipping W&B tracker initialization"
                )
            self._tracking_initialized = True
            return
        run_config = vars(self.args).copy()
        self.run = init_wandb_run(
            accelerator=self.accelerator,
            project=self.args.project_name,
            entity=self.args.wandb_entity,
            run_name=f"{self.args.experiment_name}_{self.args.model_type}",
            config=run_config,
            wandb_dir=self.args.wandb_dir,
            tags=["physical_consistency", self.args.model_type, "trd_v1", "videorepa_inspired"],
            group=self.args.run_group,
            job_type=f"train_{self.args.model_type}",
            mode=self.args.wandb_mode,
        )
        self._define_wandb_metrics()
        self._tracking_initialized = True

    def _define_wandb_metrics(self) -> None:
        if not self.accelerator.is_main_process or self.run is None:
            return
        try:
            import wandb

            wandb.define_metric("progress/micro_step")
            wandb.define_metric("progress/global_step")
            wandb.define_metric("progress/epoch")
            wandb.define_metric("train/*", step_metric="train/global_step")
            wandb.define_metric("val/*", step_metric="progress/global_step")
            wandb.define_metric("videophy2/*", step_metric="progress/global_step")
            wandb.define_metric("epoch/*", step_metric="progress/global_step")
            wandb.define_metric("progress/*", step_metric="progress/global_step")
            wandb.define_metric("runtime/optimizer/*", step_metric="progress/global_step")
            wandb.define_metric("runtime/gpu_mem/*", step_metric="progress/micro_step")
        except Exception:
            LOGGER.warning("Failed to define W&B metrics", exc_info=True)

    def validate_runtime_stack(self) -> None:
        """Reject unsupported runtime combinations before training starts."""
        if self.args.model_type not in {"low", "high", "dual"}:
            raise RuntimeError(f"Unsupported TRD-v1 model_type: {self.args.model_type}")

    def train(self) -> None:
        """Main training loop."""
        self.validate_runtime_stack()
        if self.accelerator.is_main_process:
            ensure_dir(self.args.output_dir)
            if self.best_metrics_path.exists():
                try:
                    self.best_metrics = read_json(self.best_metrics_path).get("metrics", {})
                except Exception:
                    self.best_metrics = None
        self.accelerator.wait_for_everyone()

        self.teacher_checkpoint_path = _resolve_teacher_checkpoint(
            self.args.teacher_checkpoint_dir,
            self.args.teacher_checkpoint_path,
        )
        self._build_train_and_val_loaders()
        self._log_train_plan()
        self._initialize_training_runtime(checkpoint_dir=self.args.stage1_ckpt_dir, resume_state=None)
        self._train_start_time = time.perf_counter()
        self._last_optimizer_step_time = self._train_start_time

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch + 1
            self._set_train_mode()
            epoch_metrics: list[dict[str, float]] = []
            for batch in self.train_loader:
                self._micro_step += 1
                self._reset_gpu_peak_memory_stats()
                self._log_gpu_memory(f"step_{self._micro_step}_start", emit_console=False)
                metrics: dict[str, torch.Tensor] | None = None
                with self.accelerator.accumulate(self.model_bundle):
                    self._trace_training_phase("before_training_step")
                    metrics = self.training_step(batch)
                    self._trace_training_phase(
                        "after_training_step",
                        scalars={
                            "loss_total": metrics["loss_total"],
                            "loss_fm": metrics["loss_fm"],
                            "loss_trd": metrics["loss_trd"],
                        },
                    )
                    self._log_gpu_memory(f"step_{self._micro_step}_after_forward", emit_console=False)
                    self._trace_training_phase("before_backward", scalars={"loss_total": metrics["loss_total"]})
                    self._reset_param_grad_trace_counts()
                    self.accelerator.backward(metrics["loss_total"])
                    self._trace_training_phase("after_backward")
                    # The non-detached loss keeps the autograd graph alive, which
                    # can retain old DeepSpeed parameters across pause_external
                    # validation/restoration cycles.
                    metrics["loss_total"] = metrics["loss_total"].detach()
                    self._log_gpu_memory(f"step_{self._micro_step}_after_backward", emit_console=False)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self._all_trainable_params(),
                            self.args.max_grad_norm,
                        )
                    else:
                        grad_norm = torch.tensor(0.0, device=self.accelerator.device)
                    self.optimizer.step()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()
                    self._log_gpu_memory(f"step_{self._micro_step}_after_optimizer_step", emit_console=False)
                    self.optimizer.zero_grad()
                    self._log_gpu_memory(f"step_{self._micro_step}_after_zero_grad", emit_console=False)

                epoch_metrics.append(self._scalarize_metrics(metrics))
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        self._log_progress(metrics, grad_norm)
                        self._log_train_metrics(metrics, self.scheduler, epoch, grad_norm)
                    if self.args.validation_every_steps > 0 and self.global_step % self.args.validation_every_steps == 0:
                        self.run_validation_cycle(tag=f"step_{self.global_step}")

                del metrics
                del batch
                if self.args.max_train_micro_steps > 0 and self._micro_step >= self.args.max_train_micro_steps:
                    if self.accelerator.is_main_process:
                        LOGGER.info(
                            "[MEM PROBE] reached max_train_micro_steps=%s global_step=%s micro_step=%s peak_mem=%.2fGiB; exiting before checkpoint/validation",
                            self.args.max_train_micro_steps,
                            self.global_step,
                            self._micro_step,
                            self._current_peak_allocated_gib(),
                        )
                    self.accelerator.wait_for_everyone()
                    return

            if self.args.save_every_n_epochs > 0 and self.current_epoch % self.args.save_every_n_epochs == 0:
                epoch_path = save_dual_bundle_checkpoint(
                    accelerator=self.accelerator,
                    model_bundle=self.model_bundle,
                    args=self.args,
                    tag=f"epoch_{self.current_epoch}",
                    extra_training_state=self._training_state_payload_factory(),
                    training_state_filename="projectors.pt",
                )
                if self.accelerator.is_main_process:
                    self._write_lineage(epoch_path)
                    self._log_epoch_summary(epoch, epoch_metrics, epoch_path)
            elif self.accelerator.is_main_process:
                self._log_epoch_summary(epoch, epoch_metrics, None)

            if self.args.validation_every_epochs > 0 and self.current_epoch % self.args.validation_every_epochs == 0:
                self.run_validation_cycle(tag=f"epoch_{self.current_epoch}")

        final_path = save_dual_bundle_checkpoint(
            accelerator=self.accelerator,
            model_bundle=self.model_bundle,
            args=self.args,
            tag="final",
            extra_training_state=self._training_state_payload_factory(),
            training_state_filename="projectors.pt",
        )
        if self.accelerator.is_main_process:
            self._write_lineage(final_path)
            self._write_pending_eval_commands(final_path)
        self.accelerator.end_training()

    def _build_train_and_val_loaders(self) -> None:
        train_loader = build_dataloader(
            self.args.dataset_dir,
            split="train",
            num_frames=self.args.num_frames,
            height=self.args.height,
            width=self.args.width,
            repeat=self.args.dataset_repeat,
            shuffle=True,
            num_workers=4,
        )
        val_loader = build_dataloader(
            self.args.dataset_dir,
            split="val",
            num_frames=self.args.num_frames,
            height=self.args.height,
            width=self.args.width,
            repeat=1,
            shuffle=False,
            num_workers=2,
        )
        self.train_dataset_len = len(train_loader.dataset)
        self.micro_steps_per_epoch = math.ceil(self.train_dataset_len / self.accelerator.num_processes)
        self.optimizer_steps_per_epoch = max(
            math.ceil(self.micro_steps_per_epoch / self.args.gradient_accumulation_steps),
            1,
        )
        self.total_optimizer_steps = max(self.optimizer_steps_per_epoch * self.args.num_epochs, 1)
        self.train_loader_raw = train_loader
        self.train_loader = None
        self.val_loader = val_loader

    def _log_train_plan(self) -> None:
        if self._logged_train_plan or not self.accelerator.is_main_process:
            return
        LOGGER.info(
            "[TRAIN PLAN] epochs=%s micro_steps_per_epoch=%s optimizer_steps_per_epoch=%s total_optimizer_steps=%s grad_accum=%s dataset_samples=%s world_size=%s",
            self.args.num_epochs,
            self.micro_steps_per_epoch,
            self.optimizer_steps_per_epoch,
            self.total_optimizer_steps,
            self.args.gradient_accumulation_steps,
            self.train_dataset_len,
            self.accelerator.num_processes,
        )
        log_dict(
            0,
            {
                "progress/epoch": 0,
                "progress/global_step": 0,
                "progress/micro_step": 0,
                "progress/epochs_total": self.args.num_epochs,
                "progress/micro_steps_per_epoch": self.micro_steps_per_epoch,
                "progress/optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "progress/total_optimizer_steps": self.total_optimizer_steps,
                "progress/grad_accumulation_steps": self.args.gradient_accumulation_steps,
            },
            accelerator=self.accelerator,
        )
        self._logged_train_plan = True

    def _initialize_training_runtime(
        self,
        *,
        checkpoint_dir: str | Path,
        resume_state: dict[str, Any] | None,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        if self.args.model_type == "dual":
            low_model = self.helper.load_model(self.accelerator.device, "low", checkpoint_dir=checkpoint_dir)
            self._log_gpu_memory("after_low_model_load")
            high_model = self.helper.load_model(self.accelerator.device, "high", checkpoint_dir=checkpoint_dir)
            self._log_gpu_memory("after_high_model_load")

            student_dim = int(low_model.dim)
            low_projector = StudentProjector(student_dim, self.args.teacher_feature_dim)
            high_projector = StudentProjector(student_dim, self.args.teacher_feature_dim)
            if resume_state:
                low_projector.load_state_dict(resume_state["low_projector"])
                high_projector.load_state_dict(resume_state["high_projector"])

            if should_apply_student_gradient_checkpointing(self.args):
                use_reentrant = self._student_checkpoint_use_reentrant()
                if self.accelerator.is_main_process:
                    LOGGER.info(
                        "Applying block-level gradient checkpointing for student models "
                        "(use_reentrant=%s; memory_efficient_mode=%s; skip_feature_block=%s)",
                        use_reentrant,
                        self.args.student_memory_efficient_checkpoint_mode,
                        self.args.student_target_block,
                    )
                skip_feature_block = {int(self.args.student_target_block)}
                apply_gradient_checkpointing(
                    low_model,
                    "low_noise_model",
                    use_reentrant=use_reentrant,
                    skip_block_indices=skip_feature_block,
                    memory_efficient_mode=self.args.student_memory_efficient_checkpoint_mode,
                )
                apply_gradient_checkpointing(
                    high_model,
                    "high_noise_model",
                    use_reentrant=use_reentrant,
                    skip_block_indices=skip_feature_block,
                    memory_efficient_mode=self.args.student_memory_efficient_checkpoint_mode,
                )

            if self.args.student_tuning_mode == "lora" and resume_state:
                load_lora_state_dict(low_model, resume_state.get("low_lora", {}), model_name="low_noise_model")
                load_lora_state_dict(high_model, resume_state.get("high_lora", {}), model_name="high_noise_model")

            self.model_bundle = DualBranchTrainingBundle(
                low_model=low_model,
                high_model=high_model,
                low_projector=low_projector,
                high_projector=high_projector,
                student_target_block=self.args.student_target_block,
                patch_size=self.helper.patch_size,
            )
            self._log_gpu_memory("after_dual_bundle_construct")
        else:
            branch = self.args.model_type
            branch_model = self.helper.load_model(self.accelerator.device, branch, checkpoint_dir=checkpoint_dir)
            self._log_gpu_memory(f"after_{branch}_model_load")

            projector = StudentProjector(int(branch_model.dim), self.args.teacher_feature_dim)
            if resume_state:
                projector.load_state_dict(resume_state[f"{branch}_projector"])

            if should_apply_student_gradient_checkpointing(self.args):
                use_reentrant = self._student_checkpoint_use_reentrant()
                if self.accelerator.is_main_process:
                    LOGGER.info(
                        "Applying block-level gradient checkpointing for %s_noise_model "
                        "(use_reentrant=%s; memory_efficient_mode=%s; skip_feature_block=%s)",
                        branch,
                        use_reentrant,
                        self.args.student_memory_efficient_checkpoint_mode,
                        self.args.student_target_block,
                    )
                apply_gradient_checkpointing(
                    branch_model,
                    f"{branch}_noise_model",
                    use_reentrant=use_reentrant,
                    skip_block_indices={int(self.args.student_target_block)},
                    memory_efficient_mode=self.args.student_memory_efficient_checkpoint_mode,
                )

            if self.args.student_tuning_mode == "lora" and resume_state:
                load_lora_state_dict(
                    branch_model,
                    resume_state.get(f"{branch}_lora", {}),
                    model_name=f"{branch}_noise_model",
                )

            self.model_bundle = SingleBranchTrainingBundle(
                branch=branch,
                model=branch_model,
                projector=projector,
                student_target_block=self.args.student_target_block,
                patch_size=self.helper.patch_size,
            )
            self._log_gpu_memory("after_single_bundle_construct")

        optimizer = torch.optim.AdamW(
            self._all_trainable_params(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=compute_scheduler_total_steps(
                self.train_dataset_len,
                self.accelerator.num_processes,
                self.args.gradient_accumulation_steps,
                self.args.num_epochs,
            ),
            eta_min=1e-6,
        )

        self._log_student_graph_diagnostics("before_accelerator_prepare")
        self._log_gpu_memory("before_accelerator_prepare")
        (
            self.model_bundle,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model_bundle,
            optimizer,
            self.train_loader_raw,
            scheduler,
        )
        self._install_param_grad_trace()
        self._log_gpu_memory("after_accelerator_prepare")
        if resume_state:
            self.optimizer.load_state_dict(resume_state["optimizer"])
            if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
                wrapped_optimizer = self._wrapped_optimizer()
                move_optimizer_state(wrapped_optimizer, self.accelerator.device)
            self.scheduler.load_state_dict(resume_state["scheduler"])

        self.teacher = _build_teacher_encoder(
            args=self.args,
            checkpoint_path=self.teacher_checkpoint_path,
            device=self.accelerator.device,
        )
        if self.teacher.feature_dim != self.args.teacher_feature_dim:
            raise ValueError(
                "teacher_feature_dim mismatch: "
                f"config={self.args.teacher_feature_dim}, teacher={self.teacher.feature_dim}"
            )
        self._log_gpu_memory("after_teacher_load")

    def _log_gpu_memory(self, label: str, *, emit_console: bool = True) -> None:
        if not torch.cuda.is_available() or not self.accelerator.is_main_process:
            return
        device = self.accelerator.device
        if device.type != "cuda":
            return
        allocated_gib = torch.cuda.memory_allocated(device) / (1024**3)
        reserved_gib = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        if emit_console:
            LOGGER.info(
                "[GPU MEM] %s device=%s allocated=%.2f GiB reserved=%.2f GiB max_allocated=%.2f GiB",
                label,
                device,
                allocated_gib,
                reserved_gib,
                max_allocated_gib,
            )

        match = _STEP_LABEL_PATTERN.match(label)
        if match:
            micro_step = int(match.group(1))
            phase = match.group(2)
            step = self.global_step
            payload = {
                "progress/micro_step": micro_step,
                "progress/global_step": self.global_step,
                "progress/epoch": self.current_epoch,
                f"runtime/gpu_mem/{phase}/allocated_gib": allocated_gib,
                f"runtime/gpu_mem/{phase}/reserved_gib": reserved_gib,
                f"runtime/gpu_mem/{phase}/max_allocated_gib": max_allocated_gib,
            }
            log_dict(step, payload, accelerator=self.accelerator)

    def _reset_gpu_peak_memory_stats(self) -> None:
        if not torch.cuda.is_available():
            return
        device = self.accelerator.device
        if device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(device)

    def _current_peak_allocated_gib(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        device = self.accelerator.device
        if device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(device) / (1024**3)

    def _log_progress(
        self,
        metrics: dict[str, torch.Tensor],
        grad_norm: torch.Tensor | float | None,
    ) -> None:
        now = time.perf_counter()
        if self._train_start_time is None:
            self._train_start_time = now
        elapsed = now - self._train_start_time
        if self._last_optimizer_step_time is None:
            optimizer_step_time = elapsed
        else:
            optimizer_step_time = max(now - self._last_optimizer_step_time, 0.0)
        self._last_optimizer_step_time = now

        average_step_time = elapsed / max(self.global_step, 1)
        remaining_steps = max(self.total_optimizer_steps - self.global_step, 0)
        eta_seconds = average_step_time * remaining_steps if remaining_steps > 0 else 0.0
        progress_percent = 100.0 * self.global_step / max(self.total_optimizer_steps, 1)
        current_accum_slot = ((self._micro_step - 1) % self.args.gradient_accumulation_steps) + 1
        grad_norm_value = maybe_scalar_to_float(grad_norm)
        peak_allocated_gib = self._current_peak_allocated_gib()
        loss_total = float(metrics["loss_total"].detach().item())
        loss_fm = float(metrics["loss_fm"].detach().item())
        loss_trd = float(metrics["loss_trd"].detach().item())
        lr = float(self.scheduler.get_last_lr()[0])
        grad_norm_display = float("nan") if grad_norm_value is None else grad_norm_value

        LOGGER.info(
            "[PROGRESS] epoch=%s/%s global_step=%s/%s micro_step=%s accum=%s/%s loss_total=%.4f loss_fm=%.4f loss_trd=%.4f lr=%.3e grad_norm=%.3e peak_mem=%.2fGiB step_time=%.1fs eta=%s",
            self.current_epoch,
            self.args.num_epochs,
            self.global_step,
            self.total_optimizer_steps,
            self._micro_step,
            current_accum_slot,
            self.args.gradient_accumulation_steps,
            loss_total,
            loss_fm,
            loss_trd,
            lr,
            grad_norm_display,
            peak_allocated_gib,
            optimizer_step_time,
            format_eta(eta_seconds),
        )

        payload = {
            "progress/epoch": self.current_epoch,
            "progress/global_step": self.global_step,
            "progress/micro_step": self._micro_step,
            "progress/epochs_total": self.args.num_epochs,
            "progress/optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
            "progress/total_optimizer_steps": self.total_optimizer_steps,
            "progress/percent_complete": progress_percent,
            "runtime/optimizer/step_time_sec": optimizer_step_time,
            "runtime/optimizer/avg_step_time_sec": average_step_time,
            "runtime/optimizer/eta_hours": eta_seconds / 3600.0,
            "runtime/optimizer/remaining_steps": remaining_steps,
            "runtime/optimizer/peak_allocated_gib": peak_allocated_gib,
        }
        if grad_norm_value is not None:
            payload["runtime/optimizer/grad_norm"] = grad_norm_value
        log_dict(self.global_step, payload, accelerator=self.accelerator)

    def _log_student_sequence_geometry(self, *, num_frames: int, lat_f: int, lat_h: int, lat_w: int, seq_len: int) -> None:
        if self._logged_student_sequence_geometry or not self.accelerator.is_main_process:
            return
        LOGGER.info(
            "[SEQ GEOM] num_frames=%s latent_grid=(%s,%s,%s) patch_size=%s seq_len=%s",
            num_frames,
            lat_f,
            lat_h,
            lat_w,
            self.helper.patch_size,
            seq_len,
        )
        self._logged_student_sequence_geometry = True

    def _wrapped_optimizer(self):
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _all_trainable_params(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model_bundle.parameters() if parameter.requires_grad]

    def _set_train_mode(self) -> None:
        self.model_bundle.train()

    def training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute FM + TRD for one batch."""
        trd_backward_mode = self.args.trd_backward_mode
        if _env_flag("PC_LORA_SYNTHETIC_MATMUL_SKIP_FORWARD"):
            loss_fm = collect_lora_synthetic_matmul_loss(self.model_bundle)
            self._trace_training_phase(
                "lora_synthetic_matmul_skip_forward_probe",
                scalars={"loss_lora_synthetic": loss_fm},
            )
            zero = loss_fm.detach().new_zeros(())
            spatial_matrix = torch.zeros(
                self.args.relation_tokens,
                self.args.relation_tokens,
                dtype=torch.float32,
                device="cpu",
            )
            temporal_matrix = torch.zeros(1, 1, dtype=torch.float32, device="cpu")
            return {
                "loss_total": loss_fm,
                "loss_fm": loss_fm.detach(),
                "loss_trd": zero,
                "loss_trd_spatial": zero,
                "loss_trd_temporal": zero,
                "sample_sigma": zero,
                "sample_timestep": zero,
                "teacher_feat_norm": zero,
                "student_feat_norm": zero,
                "pred_target_cosine": zero,
                "active_branch_is_high": zero,
                "_spatial_student": spatial_matrix,
                "_spatial_teacher": spatial_matrix,
                "_temporal_student": temporal_matrix,
                "_temporal_teacher": temporal_matrix,
            }
        if _env_flag("PC_LORA_PARAM_ONLY_SKIP_FORWARD"):
            loss_fm = collect_lora_parameter_loss(self.model_bundle)
            self._trace_training_phase(
                "lora_param_only_skip_forward_probe",
                scalars={"loss_lora_param": loss_fm},
            )
            zero = loss_fm.detach().new_zeros(())
            spatial_matrix = torch.zeros(
                self.args.relation_tokens,
                self.args.relation_tokens,
                dtype=torch.float32,
                device="cpu",
            )
            temporal_matrix = torch.zeros(1, 1, dtype=torch.float32, device="cpu")
            return {
                "loss_total": loss_fm,
                "loss_fm": loss_fm.detach(),
                "loss_trd": zero,
                "loss_trd_spatial": zero,
                "loss_trd_temporal": zero,
                "sample_sigma": zero,
                "sample_timestep": zero,
                "teacher_feat_norm": zero,
                "student_feat_norm": zero,
                "pred_target_cosine": zero,
                "active_branch_is_high": zero,
                "_spatial_student": spatial_matrix,
                "_spatial_teacher": spatial_matrix,
                "_temporal_student": temporal_matrix,
                "_temporal_teacher": temporal_matrix,
            }
        video = batch["video"].to(self.accelerator.device)
        poses = batch["poses"]
        actions = batch["actions"]
        intrinsics = batch["intrinsics"]
        prompt = batch["prompt"]
        height, width = video.shape[2], video.shape[3]

        with torch.no_grad():
            video_latent = self.helper.encode_video(video)
            context = self.helper.encode_text(prompt)
            y = self.helper.prepare_y(video, video_latent)

        lat_f, lat_h, lat_w = video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
        seq_len = lat_f * lat_h * lat_w // (self.helper.patch_size[1] * self.helper.patch_size[2])
        self._log_student_sequence_geometry(
            num_frames=int(video.shape[1]),
            lat_f=int(lat_f),
            lat_h=int(lat_h),
            lat_w=int(lat_w),
            seq_len=int(seq_len),
        )
        with torch.no_grad():
            dit_cond = self.helper.prepare_control_signal(
                poses,
                actions,
                intrinsics,
                height,
                width,
                lat_f,
                lat_h,
                lat_w,
            )
            timestep_sample = self.helper.sample_timestep(self.args.model_type)
            noise = torch.randn_like(video_latent)
            noisy_latent = (1.0 - timestep_sample.sigma) * video_latent + timestep_sample.sigma * noise
            target = noise - video_latent
            teacher_features = None
            if trd_backward_mode != "off":
                self._trace_training_phase("before_teacher_encode", tensors={"video": video})
                teacher_features = self.teacher.encode(video.unsqueeze(0))
                self._trace_training_phase(
                    "after_teacher_encode",
                    tensors={"teacher_tokens": teacher_features.tokens},
                )
                if not self._logged_teacher_encode_offload_memory:
                    self._log_gpu_memory("after_teacher_encode")
                    self._logged_teacher_encode_offload_memory = True

        self._trace_training_phase(
            "before_student_forward",
            tensors={
                "noisy_latent": noisy_latent,
                "context": context,
                "y": y,
            },
            scalars={
                "sample_sigma": timestep_sample.sigma,
                "sample_timestep": timestep_sample.timestep.detach().float().mean(),
            },
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, student_tokens = self.model_bundle(
                branch=timestep_sample.branch,
                noisy_latent=noisy_latent,
                timestep=timestep_sample.timestep,
                context=context,
                seq_len=seq_len,
                y=y,
                dit_cond=dit_cond,
                lat_f=lat_f,
                lat_h=lat_h,
                lat_w=lat_w,
            )
        self._trace_training_phase(
            "after_student_forward",
            tensors={"pred": pred, "student_tokens": student_tokens},
        )

        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        if _env_flag("PC_LORA_PARAM_ONLY_LOSS"):
            loss_fm = collect_lora_parameter_loss(self.model_bundle)
            self._trace_training_phase("lora_param_only_loss_probe", scalars={"loss_lora_param": loss_fm})
        elif _env_flag("PC_LORA_LOCAL_LOSS"):
            loss_fm = collect_lora_local_loss(self.model_bundle)
            self._trace_training_phase("lora_local_loss_probe", scalars={"loss_lora_local": loss_fm})
        else:
            loss_fm = F.mse_loss(pred_rest.float(), target_rest.float()) * timestep_sample.weight
        self._trace_training_phase("after_fm_loss", scalars={"loss_fm": loss_fm})

        if trd_backward_mode == "off":
            zero = loss_fm.detach().new_zeros(())
            spatial_matrix = torch.zeros(
                self.args.relation_tokens,
                self.args.relation_tokens,
                dtype=torch.float32,
                device="cpu",
            )
            temporal_matrix = torch.zeros(max(int(lat_f), 1), max(int(lat_f), 1), dtype=torch.float32, device="cpu")
            trd_output = TRDLossOutput(
                total=zero,
                spatial=zero,
                temporal=zero,
                spatial_student=spatial_matrix,
                spatial_teacher=spatial_matrix,
                temporal_student=temporal_matrix,
                temporal_teacher=temporal_matrix,
            )
            loss_total = loss_fm
            teacher_feat_norm = zero
            self._trace_training_phase("trd_loss_off", scalars={"loss_total": loss_total})
        else:
            assert teacher_features is not None
            trd_student = student_tokens.detach() if trd_backward_mode == "detached" else student_tokens
            self._trace_training_phase(
                "before_trd_loss",
                tensors={"student_tokens": trd_student, "teacher_tokens": teacher_features.tokens},
            )
            trd_output = self.trd_loss(trd_student, teacher_features.tokens)
            loss_total = loss_fm + self.args.lambda_trd * trd_output.total
            teacher_feat_norm = teacher_features.tokens.norm(dim=-1).mean().detach()
        self._trace_training_phase(
            "after_trd_loss",
            scalars={
                "loss_total": loss_total,
                "loss_trd": trd_output.total,
                "loss_trd_spatial": trd_output.spatial,
                "loss_trd_temporal": trd_output.temporal,
            },
        )

        metrics = {
            "loss_total": loss_total,
            "loss_fm": loss_fm.detach(),
            "loss_trd": trd_output.total.detach(),
            "loss_trd_spatial": trd_output.spatial.detach(),
            "loss_trd_temporal": trd_output.temporal.detach(),
            "sample_sigma": torch.tensor(timestep_sample.sigma, device=self.accelerator.device),
            "sample_timestep": timestep_sample.timestep.detach().float().mean(),
            "teacher_feat_norm": teacher_feat_norm,
            "student_feat_norm": student_tokens.norm(dim=-1).mean().detach(),
            "pred_target_cosine": F.cosine_similarity(
                pred_rest.flatten().float().unsqueeze(0),
                target_rest.flatten().float().unsqueeze(0),
                dim=1,
            ).mean().detach(),
            "active_branch_is_high": torch.tensor(
                1 if timestep_sample.branch == "high" else 0,
                device=self.accelerator.device,
            ),
            "_spatial_student": trd_output.spatial_student.detach(),
            "_spatial_teacher": trd_output.spatial_teacher.detach(),
            "_temporal_student": trd_output.temporal_student.detach(),
            "_temporal_teacher": trd_output.temporal_teacher.detach(),
        }
        return metrics

    def run_validation_cycle(self, tag: str) -> None:
        """Run one synchronized validation cycle."""
        if self.args.validation_runtime_mode == "in_process":
            if not self._should_run_inprocess_validation():
                raise RuntimeError(
                    "validation_runtime_mode=in_process requires single-process non-DeepSpeed runtime."
                )
            self.run_light_validation(tag=tag)
            return
        if self.args.validation_runtime_mode == "snapshot_only":
            self._run_snapshot_only_validation_cycle(tag)
            return
        self._run_pause_external_validation_cycle(tag)

    def _run_snapshot_only_validation_cycle(self, tag: str) -> None:
        self.accelerator.wait_for_everyone()
        checkpoint_path = save_dual_bundle_checkpoint(
            accelerator=self.accelerator,
            model_bundle=self.model_bundle,
            args=self.args,
            tag=tag,
            extra_training_state=self._training_state_payload_factory(tag=tag),
            training_state_filename="projectors.pt",
        )
        if self.accelerator.is_main_process:
            self._write_lineage(checkpoint_path)
            self._export_validation_request(checkpoint_path, tag)
        self.accelerator.wait_for_everyone()

    def _run_pause_external_validation_cycle(self, tag: str) -> None:
        self.accelerator.wait_for_everyone()
        candidate_tag = f"_candidate_{tag}"
        candidate_path = save_dual_bundle_checkpoint(
            accelerator=self.accelerator,
            model_bundle=self.model_bundle,
            args=self.args,
            tag=candidate_tag,
            extra_training_state=self._training_state_payload_factory(
                tag=tag,
                include_optimizer=True,
                include_scheduler=True,
            ),
            training_state_filename="resume_state.pt",
        )
        if self.accelerator.is_main_process:
            self._write_lineage(candidate_path)

        self.accelerator.wait_for_everyone()
        self._release_training_runtime()
        self.accelerator.wait_for_everyone()

        validation_error_path = candidate_path / "validation_error.txt"
        summary_path = candidate_path / "validation_summary.json"
        if self.accelerator.is_main_process:
            try:
                summary = self._run_external_videophy2_validation(candidate_path, tag)
                write_json(summary_path, summary)
            except Exception as exc:  # pragma: no cover - exercised on the real cluster
                validation_error_path.write_text(traceback.format_exc(), encoding="utf-8")
                LOGGER.warning("[VALIDATION FAILED] tag=%s error=%s", tag, exc, exc_info=True)

        self.accelerator.wait_for_everyone()
        self._restore_training_runtime(candidate_path)
        self.accelerator.wait_for_everyone()

        if validation_error_path.exists():
            error_message = validation_error_path.read_text(encoding="utf-8").strip() or "Validation failed"
            if self.accelerator.is_main_process:
                log_dict(
                    self.global_step,
                    {
                        "val/failed": 1,
                        "val/fail_fast": 1 if self.args.validation_fail_fast else 0,
                    },
                    accelerator=self.accelerator,
                )
                prune_checkpoint_dir(candidate_path)
            self.accelerator.wait_for_everyone()
            if self.args.validation_fail_fast:
                raise RuntimeError(error_message)
            if self.accelerator.is_main_process:
                LOGGER.warning(
                    "[VALIDATION NONFATAL] tag=%s failed; training will continue. Error saved to %s",
                    tag,
                    validation_error_path,
                )
            return

        if self.accelerator.is_main_process:
            summary = read_json(summary_path)
            self._retain_candidate_checkpoint(candidate_path, tag, summary)
        self.accelerator.wait_for_everyone()

    def _training_state_payload_factory(
        self,
        *,
        tag: str | None = None,
        include_optimizer: bool = False,
        include_scheduler: bool = False,
    ) -> Callable[[dict[str, torch.Tensor], nn.Module], dict[str, Any]]:
        optimizer_state = self._gather_optimizer_resume_state() if include_optimizer else None
        scheduler_state = self.scheduler.state_dict() if include_scheduler else None

        def _factory(bundle_state: dict[str, torch.Tensor], _unwrapped: nn.Module) -> dict[str, Any]:
            return _build_training_state_payload_from_bundle_state(
                bundle_state=bundle_state,
                student_tuning_mode=self.args.student_tuning_mode,
                global_step=self.global_step,
                epoch=self.current_epoch,
                tag=tag,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                student_base_checkpoint_dir=self.args.stage1_ckpt_dir,
                model_type=self.args.model_type,
            )

        return _factory

    def _gather_optimizer_resume_state(self) -> dict[str, Any] | list[dict[str, Any]]:
        optimizer_state = self.optimizer.state_dict()
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return optimizer_state
        if not dist.is_available() or not dist.is_initialized() or self.accelerator.num_processes <= 1:
            return [optimizer_state]

        gathered_states: list[dict[str, Any]] | None = [None] * self.accelerator.num_processes if self.accelerator.is_main_process else None
        dist.gather_object(
            optimizer_state,
            object_gather_list=gathered_states,
            dst=0,
        )
        if self.accelerator.is_main_process:
            return gathered_states
        return [optimizer_state]

    def _release_training_runtime(self) -> None:
        old_teacher = self.teacher
        old_model_bundle = self.model_bundle
        old_optimizer = self.optimizer
        old_train_loader = self.train_loader
        old_scheduler = self.scheduler
        self.teacher = None
        self.helper.release_runtime_components()
        _destroy_deepspeed_engines(self.accelerator, old_model_bundle)
        (
            self.model_bundle,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.free_memory(
            old_model_bundle,
            old_optimizer,
            old_train_loader,
            old_scheduler,
        )
        del old_teacher, old_model_bundle, old_optimizer, old_train_loader, old_scheduler
        _destroy_deepspeed_engines(self.accelerator, self.model_bundle)
        _clear_torch_cuda_cache()
        self._log_gpu_memory("after_runtime_release")

    def _restore_training_runtime(self, checkpoint_path: Path) -> None:
        resume_state = torch.load(
            checkpoint_path / "training_only" / "resume_state.pt",
            map_location="cpu",
        )
        checkpoint_dir: str | Path = checkpoint_path
        if self.args.student_tuning_mode == "lora":
            checkpoint_dir = resume_state.get("student_base_checkpoint_dir", self.args.stage1_ckpt_dir)
        self._initialize_training_runtime(checkpoint_dir=checkpoint_dir, resume_state=resume_state)
        self.global_step = int(resume_state.get("global_step", self.global_step))
        self.current_epoch = int(resume_state.get("epoch", self.current_epoch))

    def _run_external_videophy2_validation(self, checkpoint_path: Path, tag: str) -> dict[str, Any]:
        validation_root = ensure_dir(
            Path(self.args.output_root) / "runs" / "train_validation" / self.args.experiment_name / tag
        )
        generation_root = ensure_dir(validation_root / "generated")
        validation_seed = int(self.args.validation_seed_list[0])
        generation_checkpoint_path = self._materialize_eval_checkpoint_bundle(
            checkpoint_path,
            experiment_name=f"{self.args.experiment_name}_{tag}_generation",
        )

        run_command(
            self._validation_generation_command(generation_checkpoint_path, generation_root, validation_seed),
            cwd=PROJECT_ROOT,
            log_path=validation_root / "generation.log",
        )

        generated_manifest = generation_root / "lingbotstage1" / "generated_videos.csv"
        if not generated_manifest.exists():
            raise FileNotFoundError(f"Validation generation manifest missing: {generated_manifest}")

        videophy_experiment = f"{self.args.experiment_name}_{tag}"
        run_command(
            ["bash", str(PROJECT_ROOT / "scripts" / "run_videophy2_dataset_autoeval_parallel.sh")],
            cwd=PROJECT_ROOT,
            env=self._validation_videophy_env(
                experiment_name=videophy_experiment,
                manifest_path=generated_manifest,
                video_source_root=generation_root / "lingbotstage1",
                output_root=validation_root,
            ),
            log_path=validation_root / "videophy2.log",
        )

        summary_path = validation_root / "runs" / "eval" / "videophy2" / videophy_experiment / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Validation VideoPhy-2 summary missing: {summary_path}")
        summary = read_json(summary_path)
        self._emit_videophy2_summary(tag, summary)
        return summary

    def _validation_generation_command(
        self,
        checkpoint_path: Path,
        generation_root: Path,
        validation_seed: int,
    ) -> list[str]:
        return [
            "python",
            "-m",
            "physical_consistency.cli.run_lingbot_generation",
            "--env_file",
            self.args.env_file,
            "--manifest_path",
            self.args.manifest_mini_val,
            "--dataset_dir",
            self.args.dataset_dir,
            "--output_root",
            str(generation_root),
            "--base_model_dir",
            self.args.base_model_dir,
            "--stage1_ckpt_dir",
            str(checkpoint_path),
            "--val_inf_root",
            str(generation_root),
            "--seed",
            str(validation_seed),
            "--frame_num",
            str(self.args.num_frames),
            "--sample_steps",
            str(self.args.validation_sample_steps),
            "--guide_scale",
            str(self.args.guide_scale),
            "--height",
            str(self.args.height),
            "--width",
            str(self.args.width),
            "--models",
            "stage1",
            "--num_gpus",
            str(self.args.num_gpus),
            "--ulysses_size",
            str(self.args.ulysses_size),
        ]

    def _validation_videophy_env(
        self,
        *,
        experiment_name: str,
        manifest_path: Path,
        video_source_root: Path,
        output_root: Path,
    ) -> dict[str, str]:
        return {
            "ENV_FILE": self.args.env_file,
            "CONFIG_PATH": str(CONFIG_DIR / "videophy2_dataset_autoeval.yaml"),
            "EXPERIMENT_NAME": experiment_name,
            "MANIFEST": str(manifest_path),
            "VIDEO_SOURCE_ROOT": str(video_source_root),
            "VIDEO_SOURCE_MODE": "manifest_video_column",
            "MANIFEST_VIDEO_COLUMN": "candidate_videopath",
            "MANIFEST_CAPTION_COLUMN": "prompt",
            "OUTPUT_ROOT": str(output_root),
            "GPU_LIST": self.visible_gpu_list,
            "SEED": "0",
            "VIDEOPHY2_QUIET": "1",
            "VIDEOPHY2_SUMMARY_STDOUT": "0",
            "KILL_EXISTING_GPU_PIDS": "0",
        }

    def _allow_stage1_fallback_for_eval(self) -> bool:
        return self.args.model_type in {"low", "high"}

    def _materialize_eval_checkpoint_bundle(self, checkpoint_path: Path, *, experiment_name: str) -> Path:
        return materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=checkpoint_path,
            output_root=self.args.output_root,
            experiment_name=experiment_name,
            stage1_ckpt_dir=self.args.stage1_ckpt_dir,
            allow_stage1_fallback=self._allow_stage1_fallback_for_eval(),
        )

    def _emit_videophy2_summary(self, tag: str, summary: dict[str, Any]) -> None:
        rendered = format_videophy2_summary(
            summary,
            title="Lingbot_VideoREPA",
            include_per_seed=False,
        )
        print(rendered)
        self._log_videophy2_metrics(tag, summary)

    def _log_videophy2_metrics(self, tag: str, summary: dict[str, Any]) -> None:
        metrics = self._extract_videophy2_metrics(summary)
        log_dict(
            self.global_step,
            {
                "videophy2/sa_mean": metrics["sa_mean"],
                "videophy2/pc_mean": metrics["pc_mean"],
                "videophy2/joint_ge_4": metrics["joint"],
            },
            accelerator=self.accelerator,
        )
        if self.run is None:
            return
        try:
            import wandb

            table = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["SA Mean", metrics["sa_mean"]],
                    ["PC Mean", metrics["pc_mean"]],
                    ["Joint >= 4", metrics["joint"]],
                ],
            )
            wandb.log(
                {
                    "videophy2/tag": tag,
                    "videophy2/table": table,
                },
                step=self.global_step,
            )
        except Exception:
            LOGGER.warning("Failed to write VideoPhy-2 summary table to W&B", exc_info=True)

    def _extract_videophy2_metrics(self, summary: dict[str, Any]) -> dict[str, float]:
        means = summary.get("means", {})

        def _metric(name: str) -> float:
            payload = means.get(name, {})
            return float(payload.get("mean", 0.0) or 0.0)

        return {
            "sa_mean": _metric("sa_mean"),
            "pc_mean": _metric("pc_mean"),
            "joint": _metric("joint"),
        }

    def _retain_candidate_checkpoint(self, candidate_path: Path, tag: str, summary: dict[str, Any]) -> None:
        metrics = self._extract_videophy2_metrics(summary)
        record = {
            "tag": tag,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "metrics": metrics,
        }
        write_json(candidate_path / "validation_summary.json", summary)
        write_json(candidate_path / "validation_metrics.json", record)
        self._strip_resume_state(candidate_path)

        is_best = self._is_better_summary(metrics, self.best_metrics)
        log_dict(
            self.global_step,
            {
                "videophy2/is_best": 1 if is_best else 0,
            },
            accelerator=self.accelerator,
        )
        if is_best:
            prune_checkpoint_dir(self.best_checkpoint_path)
            candidate_path.rename(self.best_checkpoint_path)
            record["checkpoint_path"] = str(self.best_checkpoint_path)
            write_json(self.best_metrics_path, record)
            self.best_metrics = metrics
        else:
            prune_checkpoint_dir(candidate_path)

    def _is_better_summary(
        self,
        candidate_metrics: dict[str, float],
        current_best_metrics: dict[str, float] | None,
    ) -> bool:
        if current_best_metrics is None:
            return True
        candidate_key = (
            candidate_metrics["joint"],
            candidate_metrics["pc_mean"],
            candidate_metrics["sa_mean"],
        )
        best_key = (
            current_best_metrics.get("joint", 0.0),
            current_best_metrics.get("pc_mean", 0.0),
            current_best_metrics.get("sa_mean", 0.0),
        )
        return candidate_key > best_key

    def _strip_resume_state(self, checkpoint_path: Path) -> None:
        resume_state_path = checkpoint_path / "training_only" / "resume_state.pt"
        if resume_state_path.exists():
            resume_state_path.unlink()

    def _should_run_inprocess_validation(self) -> bool:
        return (
            self.accelerator.num_processes == 1
            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
        )

    def run_light_validation(self, tag: str) -> None:
        """Run lightweight loss validation inside the current training process."""
        del tag  # unused in the light path
        self.model_bundle.eval()
        sample_count = 0
        total_loss = 0.0
        if self.accelerator.is_main_process:
            with torch.no_grad():
                for batch in self.val_loader:
                    metrics = self.training_step(batch)
                    total_loss += float(metrics["loss_total"].detach().item())
                    sample_count += 1
                    if sample_count >= self.args.mini_val_max_samples:
                        break
            avg = total_loss / max(sample_count, 1)
            log_dict(
                self.global_step,
                {
                    "val/light_loss_total": avg,
                    "val/light_sample_count": sample_count,
                },
                accelerator=self.accelerator,
            )
        self._set_train_mode()

    def _scalarize_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        output: dict[str, float] = {}
        for key, value in metrics.items():
            if not torch.is_tensor(value) or value.ndim != 0:
                continue
            output[key] = float(value.detach().item())
        return output

    def _log_train_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        scheduler,
        epoch: int,
        grad_norm: torch.Tensor | float | None,
    ) -> None:
        payload = {
            "train/loss_total": float(metrics["loss_total"].detach().item()),
            "train/loss_fm": float(metrics["loss_fm"].detach().item()),
            "train/loss_trd": float(metrics["loss_trd"].detach().item()),
            "train/loss_trd_spatial": float(metrics["loss_trd_spatial"].detach().item()),
            "train/loss_trd_temporal": float(metrics["loss_trd_temporal"].detach().item()),
            "train/lr": float(scheduler.get_last_lr()[0]),
            "train/global_step": self.global_step,
            "train/epoch": epoch + 1,
            "progress/global_step": self.global_step,
            "progress/micro_step": self._micro_step,
            "progress/epoch": epoch + 1,
            "train/sample_sigma": float(metrics["sample_sigma"].detach().item()),
            "train/sample_timestep": float(metrics["sample_timestep"].detach().item()),
            "train/teacher_feat_norm": float(metrics["teacher_feat_norm"].detach().item()),
            "train/student_feat_norm": float(metrics["student_feat_norm"].detach().item()),
            "train/pred_target_cosine": float(metrics["pred_target_cosine"].detach().item()),
            "train/active_branch_is_high": float(metrics["active_branch_is_high"].detach().item()),
        }
        grad_norm_value = maybe_scalar_to_float(grad_norm)
        if grad_norm_value is not None:
            payload["train/grad_norm"] = grad_norm_value
        should_log_relation_images = (
            self.global_step == 1
            or self.global_step % self.args.wandb_relation_image_every_steps == 0
        )
        if should_log_relation_images:
            spatial_student = relation_matrix_image(
                tensor_to_numpy_float32(metrics["_spatial_student"]),
                "student_spatial",
            )
            spatial_teacher = relation_matrix_image(
                tensor_to_numpy_float32(metrics["_spatial_teacher"]),
                "teacher_spatial",
            )
            temporal_student = relation_matrix_image(
                tensor_to_numpy_float32(metrics["_temporal_student"]),
                "student_temporal",
            )
            temporal_teacher = relation_matrix_image(
                tensor_to_numpy_float32(metrics["_temporal_teacher"]),
                "teacher_temporal",
            )
            if spatial_student is not None:
                payload["train/spatial_relation_student"] = spatial_student
            if spatial_teacher is not None:
                payload["train/spatial_relation_teacher"] = spatial_teacher
            if temporal_student is not None:
                payload["train/temporal_relation_student"] = temporal_student
            if temporal_teacher is not None:
                payload["train/temporal_relation_teacher"] = temporal_teacher
        log_dict(self.global_step, payload, accelerator=self.accelerator)

    def _log_epoch_summary(self, epoch: int, epoch_metrics: list[dict[str, float]], checkpoint_path: Path | None) -> None:
        if not epoch_metrics:
            return
        loss_mean = sum(item["loss_total"] for item in epoch_metrics) / len(epoch_metrics)
        log_dict(
            self.global_step,
            {
                "epoch/loss_total_mean": loss_mean,
                "epoch/index": epoch + 1,
            },
            accelerator=self.accelerator,
        )
        if self.run is not None and checkpoint_path is not None:
            log_dict(
                self.global_step,
                {
                    "epoch/checkpoint_dir": str(checkpoint_path),
                    "epoch": epoch + 1,
                },
            )

    def _write_lineage(self, checkpoint_path: Path) -> None:
        record = LineageRecord.create(
            parent_stage1_ckpt=self.args.stage1_ckpt_dir,
            base_model_dir=self.args.base_model_dir,
            dataset_dir=self.args.dataset_dir,
            config_path=self.args.config_path,
            config_hash=self.args.config_hash,
            project_root=PROJECT_ROOT,
            notes=f"model_type={self.args.model_type}",
        )
        subfolders = (
            MODEL_SUBFOLDERS
            if self.args.model_type == "dual"
            else (get_model_subfolder(self.args.model_type),)
        )
        for subfolder in subfolders:
            branch_dir = checkpoint_path / subfolder
            if branch_dir.exists():
                record.write(branch_dir / "lineage.json")

    def _write_pending_eval_commands(self, checkpoint_path: Path) -> None:
        output_dir = checkpoint_path / "post_train_eval"
        ensure_dir(output_dir)
        bundle_dir = self._materialize_eval_checkpoint_bundle(
            checkpoint_path,
            experiment_name=f"{self.args.experiment_name}_final",
        )
        eval_config_path = output_dir / "eval_trd_final.yaml"
        eval_config = {
            "experiment_name": self.args.experiment_name,
            "seed_list": self.args.validation_seed_list,
            "split": "val",
            "manifest_path": self.args.manifest_full_val,
            "frame_num": self.args.num_frames,
            "sample_steps": self.args.validation_sample_steps,
            "guide_scale": self.args.guide_scale,
            "height": self.args.height,
            "width": self.args.width,
            "num_gpus": self.args.num_gpus,
            "ulysses_size": self.args.ulysses_size,
            "run_fid_fvd": True,
            "run_action_control": True,
            "run_videophy2": True,
            "base_model_dir": self.args.base_model_dir,
            "ft_ckpt_dir": str(bundle_dir),
            "output_root": self.args.output_root,
            "stage1_ckpt_dir": self.args.stage1_ckpt_dir,
            "allow_stage1_fallback": self._allow_stage1_fallback_for_eval(),
        }
        write_yaml(eval_config_path, eval_config)
        command_file = output_dir / "commands.txt"
        command_file.write_text(
            "\n".join(
                [
                    (
                        "python -m physical_consistency.cli.run_csgo_metrics "
                        f"--config {eval_config_path} "
                        f"--ft_ckpt_dir {bundle_dir} "
                        f"--experiment_name {self.args.experiment_name}"
                    ),
                    (
                        "python -m physical_consistency.cli.run_videophy2 "
                        f"--config {CONFIG_DIR / 'videophy2_eval.yaml'} "
                        f"--experiment_name {self.args.experiment_name} "
                        f"--manifest_csv {self.args.manifest_full_val} "
                        f"--generated_root {self.args.output_root}/runs/eval/{self.args.experiment_name}"
                    ),
                ]
            ),
            encoding="utf-8",
        )

    def _export_validation_request(self, checkpoint_path: Path, tag: str) -> None:
        export_dir = checkpoint_path / "validation_export"
        ensure_dir(export_dir)
        bundle_dir = self._materialize_eval_checkpoint_bundle(
            checkpoint_path,
            experiment_name=f"{self.args.experiment_name}_{tag}",
        )
        eval_config_path = export_dir / "eval_trd_snapshot.yaml"
        eval_config = {
            "experiment_name": f"{self.args.experiment_name}_{tag}",
            "seed_list": self.args.validation_seed_list,
            "split": "val",
            "manifest_path": self.args.manifest_mini_val,
            "frame_num": self.args.num_frames,
            "sample_steps": self.args.validation_sample_steps,
            "guide_scale": self.args.guide_scale,
            "height": self.args.height,
            "width": self.args.width,
            "num_gpus": self.args.num_gpus,
            "ulysses_size": self.args.ulysses_size,
            "run_fid_fvd": True,
            "run_action_control": True,
            "run_videophy2": True,
            "base_model_dir": self.args.base_model_dir,
            "ft_ckpt_dir": str(bundle_dir),
            "output_root": self.args.output_root,
            "stage1_ckpt_dir": self.args.stage1_ckpt_dir,
            "allow_stage1_fallback": self._allow_stage1_fallback_for_eval(),
        }
        write_yaml(eval_config_path, eval_config)
        request_payload = {
            "tag": tag,
            "global_step": self.global_step,
            "checkpoint_path": str(checkpoint_path),
            "eval_bundle_dir": str(bundle_dir),
            "manifest_mini_val": self.args.manifest_mini_val,
            "validation_seed_list": self.args.validation_seed_list,
            "eval_config_path": str(eval_config_path),
            "commands": [
                (
                    "python -m physical_consistency.cli.run_csgo_metrics "
                    f"--config {eval_config_path} "
                    f"--ft_ckpt_dir {bundle_dir} "
                    f"--experiment_name {self.args.experiment_name}_{tag}"
                ),
                (
                    "python -m physical_consistency.cli.run_videophy2 "
                    f"--config {CONFIG_DIR / 'videophy2_eval.yaml'} "
                    f"--experiment_name {self.args.experiment_name}_{tag} "
                    f"--manifest_csv {self.args.manifest_mini_val} "
                    f"--generated_root {self.args.output_root}/runs/eval/{self.args.experiment_name}_{tag}"
                ),
            ],
        }
        write_json(export_dir / "validation_request.json", request_payload)
        (export_dir / "run_validation.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "\n".join(request_payload["commands"])
            + "\n",
            encoding="utf-8",
        )
        log_dict(
            self.global_step,
            {
                "val/export_step": self.global_step,
                "val/export_seed_count": len(self.args.validation_seed_list),
            },
            accelerator=self.accelerator,
        )


def build_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command-line overrides."""
    payload = read_yaml(cli_args.config)
    for key, value in vars(cli_args).items():
        if key == "config" or value in ("", None):
            continue
        payload[key] = value

    path_cfg = resolve_path_config(SimpleNamespace(**payload), env_file=cli_args.env_file or None)
    payload.setdefault("base_model_dir", path_cfg.base_model_dir)
    payload.setdefault("stage1_ckpt_dir", path_cfg.stage1_ckpt_dir)
    payload.setdefault("dataset_dir", path_cfg.dataset_dir)
    payload.setdefault("lingbot_code_dir", path_cfg.lingbot_code_dir)
    payload.setdefault("output_root", path_cfg.output_root)
    payload.setdefault("wandb_dir", path_cfg.wandb_dir)
    payload.setdefault("teacher_repo_dir", path_cfg.videorepa_repo_dir)
    payload.setdefault("teacher_checkpoint_dir", path_cfg.teacher_ckpt_dir)

    payload.setdefault("wandb_entity", "")
    payload.setdefault("margin", 0.1)
    payload.setdefault("teacher_backend", "vjepa2")
    payload.setdefault("teacher_feature_dim", 768)
    payload.setdefault("teacher_height", 160)
    payload.setdefault("teacher_width", 240)
    payload.setdefault("teacher_pretrained_frames", 64)
    payload.setdefault("teacher_input_frames", 64)
    payload.setdefault("teacher_drop_first_frame", False)
    payload.setdefault("num_frames", 81)
    payload.setdefault("height", 480)
    payload.setdefault("width", 832)
    payload.setdefault("teacher_model_variant", "vjepa2_1_vit_base_384")
    payload.setdefault("teacher_dtype", "bfloat16")
    payload.setdefault("teacher_image_size", 384)
    payload.setdefault("teacher_offload_after_encode", True)
    payload.setdefault("student_tuning_mode", "lora")
    payload.setdefault("student_lora_rank", 16)
    payload.setdefault("student_lora_alpha", 16)
    payload.setdefault("student_lora_dropout", 0.0)
    payload.setdefault("student_lora_block_start", 0)
    payload.setdefault("student_lora_chunk_size", 0)
    payload.setdefault("student_lora_merge_mode", "inplace")
    payload.setdefault("student_memory_efficient_modulation", True)
    payload.setdefault("student_memory_efficient_checkpoint_mode", "full")
    payload.setdefault("student_ffn_chunk_size", 512)
    payload.setdefault("student_norm_chunk_size", 0)
    payload.setdefault("student_checkpoint_use_reentrant", None)
    payload.setdefault("student_ddp_find_unused_parameters", True)
    payload.setdefault("student_ddp_static_graph", False)
    payload.setdefault("student_diagnose_training_graph", True)
    payload.setdefault("student_param_grad_trace", False)
    payload.setdefault("student_param_grad_trace_pattern", "lora")
    payload.setdefault("gradient_checkpointing", True)
    payload.setdefault("validation_every_steps", 0)
    payload.setdefault("validation_every_epochs", 1)
    payload.setdefault("save_every_n_epochs", 0)
    payload.setdefault("max_train_micro_steps", 0)
    payload.setdefault("mini_val_max_samples", 8)
    payload.setdefault("student_target_block", 20)
    payload.setdefault("relation_tokens", 64)
    payload.setdefault("lambda_trd", 0.1)
    payload.setdefault("trd_backward_mode", "full")
    payload.setdefault("lambda_spatial", 1.0)
    payload.setdefault("lambda_temporal", 1.0)
    payload.setdefault("run_group", "trd_v1")
    payload.setdefault("wandb_mode", "online")
    payload.setdefault("wandb_relation_image_every_steps", 25)
    payload.setdefault("distributed_timeout_hours", 8)
    payload.setdefault("validation_runtime_mode", "pause_external")
    payload.setdefault("validation_fail_fast", False)
    payload.setdefault("validation_sample_steps", payload.get("sample_steps", 70))
    payload.setdefault("allow_deepspeed_feature_hook_experimental", False)
    payload.setdefault("best_checkpoint_name", "best_videophy2")
    payload["allow_deepspeed_feature_hook_experimental"] = _coerce_bool(
        payload["allow_deepspeed_feature_hook_experimental"]
    )
    payload["teacher_offload_after_encode"] = _coerce_bool(payload["teacher_offload_after_encode"])
    payload["teacher_drop_first_frame"] = _coerce_bool(payload["teacher_drop_first_frame"])
    payload["teacher_backend"] = str(payload["teacher_backend"]).strip().lower()
    payload["height"] = int(payload["height"])
    payload["width"] = int(payload["width"])
    if payload["height"] <= 0 or payload["width"] <= 0:
        raise ValueError(f"height and width must be positive, got {payload['height']}x{payload['width']}")
    payload["student_tuning_mode"] = str(payload["student_tuning_mode"]).strip().lower()
    if payload["student_tuning_mode"] not in {"full", "lora"}:
        raise ValueError(f"Unsupported student_tuning_mode: {payload['student_tuning_mode']}")
    if payload["student_lora_rank"] in ("", None):
        payload["student_lora_rank"] = 16
    else:
        payload["student_lora_rank"] = int(payload["student_lora_rank"])
    if payload["student_lora_alpha"] in ("", None):
        payload["student_lora_alpha"] = payload["student_lora_rank"]
    else:
        payload["student_lora_alpha"] = int(payload["student_lora_alpha"])
    if payload["student_lora_dropout"] in ("", None):
        payload["student_lora_dropout"] = 0.0
    else:
        payload["student_lora_dropout"] = float(payload["student_lora_dropout"])
    if payload["student_lora_block_start"] in ("", None):
        payload["student_lora_block_start"] = 0
    else:
        payload["student_lora_block_start"] = int(payload["student_lora_block_start"])
    if payload["student_lora_chunk_size"] in ("", None):
        payload["student_lora_chunk_size"] = 0
    else:
        payload["student_lora_chunk_size"] = int(payload["student_lora_chunk_size"])
    payload["student_lora_merge_mode"] = str(payload["student_lora_merge_mode"]).strip().lower().replace("-", "_")
    if payload["student_lora_merge_mode"] not in {"inplace", "out_of_place"}:
        raise ValueError(
            "student_lora_merge_mode must be one of inplace, out_of_place; "
            f"got {payload['student_lora_merge_mode']}"
        )
    if payload["student_tuning_mode"] == "lora":
        if payload["student_lora_rank"] <= 0:
            raise ValueError(f"student_lora_rank must be positive, got {payload['student_lora_rank']}")
        if payload["student_lora_alpha"] <= 0:
            raise ValueError(f"student_lora_alpha must be positive, got {payload['student_lora_alpha']}")
        if not 0.0 <= payload["student_lora_dropout"] < 1.0:
            raise ValueError(
                f"student_lora_dropout must be in [0, 1), got {payload['student_lora_dropout']}"
            )
        if payload["student_lora_block_start"] < 0:
            raise ValueError(
                f"student_lora_block_start must be non-negative, got {payload['student_lora_block_start']}"
            )
        if payload["student_lora_chunk_size"] < 0:
            raise ValueError(
                f"student_lora_chunk_size must be non-negative, got {payload['student_lora_chunk_size']}"
            )
    payload["student_memory_efficient_modulation"] = _coerce_bool(payload["student_memory_efficient_modulation"])
    payload["student_memory_efficient_checkpoint_mode"] = str(
        payload["student_memory_efficient_checkpoint_mode"]
    ).strip().lower()
    if payload["student_memory_efficient_checkpoint_mode"] not in {"full", "inner", "none"}:
        raise ValueError(
            "student_memory_efficient_checkpoint_mode must be one of full, inner, none; "
            f"got {payload['student_memory_efficient_checkpoint_mode']}"
        )
    payload["gradient_checkpointing"] = _coerce_bool(payload["gradient_checkpointing"])
    if payload["student_checkpoint_use_reentrant"] in ("", None):
        payload["student_checkpoint_use_reentrant"] = None
    else:
        payload["student_checkpoint_use_reentrant"] = _coerce_bool(payload["student_checkpoint_use_reentrant"])
    payload["student_ddp_find_unused_parameters"] = _coerce_bool(payload["student_ddp_find_unused_parameters"])
    payload["student_ddp_static_graph"] = _coerce_bool(payload["student_ddp_static_graph"])
    payload["student_diagnose_training_graph"] = _coerce_bool(payload["student_diagnose_training_graph"])
    payload["student_param_grad_trace"] = _coerce_bool(payload["student_param_grad_trace"])
    payload["student_param_grad_trace_pattern"] = str(payload["student_param_grad_trace_pattern"])
    payload["validation_fail_fast"] = _coerce_bool(payload["validation_fail_fast"])
    if payload["student_ffn_chunk_size"] in ("", None):
        payload["student_ffn_chunk_size"] = 512
    else:
        payload["student_ffn_chunk_size"] = int(payload["student_ffn_chunk_size"])
    if payload["student_norm_chunk_size"] in ("", None):
        payload["student_norm_chunk_size"] = 0
    else:
        payload["student_norm_chunk_size"] = int(payload["student_norm_chunk_size"])
    if payload["student_norm_chunk_size"] < 0:
        raise ValueError(f"student_norm_chunk_size must be non-negative, got {payload['student_norm_chunk_size']}")
    if payload["wandb_relation_image_every_steps"] in ("", None):
        payload["wandb_relation_image_every_steps"] = 25
    else:
        payload["wandb_relation_image_every_steps"] = int(payload["wandb_relation_image_every_steps"])
    if payload["wandb_relation_image_every_steps"] <= 0:
        raise ValueError(
            f"wandb_relation_image_every_steps must be positive, got {payload['wandb_relation_image_every_steps']}"
        )
    if payload["validation_sample_steps"] in ("", None):
        payload["validation_sample_steps"] = int(payload.get("sample_steps", 70))
    else:
        payload["validation_sample_steps"] = int(payload["validation_sample_steps"])
    if payload["validation_sample_steps"] <= 0:
        raise ValueError(f"validation_sample_steps must be positive, got {payload['validation_sample_steps']}")
    if payload["save_every_n_epochs"] in ("", None):
        payload["save_every_n_epochs"] = 0
    else:
        payload["save_every_n_epochs"] = int(payload["save_every_n_epochs"])
    if payload["save_every_n_epochs"] < 0:
        raise ValueError(f"save_every_n_epochs must be non-negative, got {payload['save_every_n_epochs']}")
    if payload["max_train_micro_steps"] in ("", None):
        payload["max_train_micro_steps"] = 0
    else:
        payload["max_train_micro_steps"] = int(payload["max_train_micro_steps"])
    if payload["max_train_micro_steps"] < 0:
        raise ValueError(f"max_train_micro_steps must be non-negative, got {payload['max_train_micro_steps']}")
    payload["trd_backward_mode"] = str(payload["trd_backward_mode"]).strip().lower()
    if payload["trd_backward_mode"] not in {"full", "detached", "off"}:
        raise ValueError(
            "trd_backward_mode must be one of full, detached, off; "
            f"got {payload['trd_backward_mode']}"
        )

    payload["output_dir"] = str(Path(payload["output_root"]) / "checkpoints" / payload["experiment_name"])
    payload.setdefault("teacher_checkpoint_path", "")
    payload["config_path"] = str(Path(cli_args.config).resolve())
    payload["config_hash"] = hashlib.sha256(Path(cli_args.config).read_bytes()).hexdigest()[:16]
    payload["env_file"] = str(cli_args.env_file)
    payload["subfolder"] = get_model_subfolder(payload["model_type"]) if payload["model_type"] in {"low", "high"} else ""
    return argparse.Namespace(**payload)


def _resolve_teacher_checkpoint(teacher_dir: str, teacher_checkpoint_path: str = "") -> str:
    if teacher_checkpoint_path:
        path = Path(teacher_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
        return str(path)
    path = Path(teacher_dir)
    if path.is_file():
        return str(path)
    candidates = sorted(
        list(path.rglob("*.pt")) + list(path.rglob("*.pth"))
    )
    if not candidates:
        raise FileNotFoundError(f"No teacher checkpoint .pt/.pth found under {teacher_dir}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Multiple teacher checkpoints found. "
            f"Please pass --teacher_checkpoint_path explicitly. Candidates: {', '.join(str(item) for item in candidates[:5])}"
        )
    return str(candidates[0])


def _coerce_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot coerce to bool: {value}")


def _require_existing_path(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} is empty. Pass --{label} or set it in the env file.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def parse_args() -> argparse.Namespace:
    """CLI args for train entry."""
    parser = argparse.ArgumentParser(description="Train TRD-v1 in isolation.")
    parser.add_argument("--config", type=str, default=str(CONFIG_DIR / "train_trd_v1.yaml"))
    parser.add_argument("--env_file", type=str, default=str(CONFIG_DIR / "path_config_cluster.env"))
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--model_type", type=str, default="", choices=["low", "high", "dual"])
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--lingbot_code_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--wandb_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--ulysses_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--teacher_backend", type=str, default="")
    parser.add_argument("--teacher_repo_dir", type=str, default="")
    parser.add_argument("--teacher_checkpoint_dir", type=str, default="")
    parser.add_argument("--teacher_checkpoint_path", type=str, default="")
    parser.add_argument("--teacher_model_variant", type=str, default="")
    parser.add_argument("--teacher_input_frames", type=int, default=None)
    parser.add_argument("--teacher_image_size", type=int, default=None)
    parser.add_argument("--teacher_feature_dim", type=int, default=None)
    parser.add_argument("--teacher_drop_first_frame", type=str, default="")
    parser.add_argument("--teacher_dtype", type=str, default="")
    parser.add_argument("--teacher_offload_after_encode", type=str, default="")
    parser.add_argument("--student_tuning_mode", type=str, default="")
    parser.add_argument("--student_lora_rank", type=int, default=None)
    parser.add_argument("--student_lora_alpha", type=int, default=None)
    parser.add_argument("--student_lora_dropout", type=float, default=None)
    parser.add_argument("--student_lora_block_start", type=int, default=None)
    parser.add_argument("--student_lora_chunk_size", type=int, default=None)
    parser.add_argument("--student_lora_merge_mode", type=str, default="")
    parser.add_argument("--gradient_checkpointing", type=str, default="")
    parser.add_argument("--student_checkpoint_use_reentrant", type=str, default="")
    parser.add_argument("--student_ddp_find_unused_parameters", type=str, default="")
    parser.add_argument("--student_ddp_static_graph", type=str, default="")
    parser.add_argument("--student_diagnose_training_graph", type=str, default="")
    parser.add_argument("--student_param_grad_trace", type=str, default="")
    parser.add_argument("--student_param_grad_trace_pattern", type=str, default="")
    parser.add_argument("--student_memory_efficient_modulation", type=str, default="")
    parser.add_argument("--student_memory_efficient_checkpoint_mode", type=str, default="")
    parser.add_argument("--student_ffn_chunk_size", type=int, default=None)
    parser.add_argument("--student_norm_chunk_size", type=int, default=None)
    parser.add_argument("--trd_backward_mode", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="")
    parser.add_argument("--wandb_relation_image_every_steps", type=int, default=None)
    parser.add_argument("--validation_every_steps", type=int, default=None)
    parser.add_argument("--validation_every_epochs", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--max_train_micro_steps", type=int, default=None)
    parser.add_argument("--validation_sample_steps", type=int, default=None)
    parser.add_argument("--validation_runtime_mode", type=str, default="")
    parser.add_argument("--validation_fail_fast", type=str, default="")
    parser.add_argument("--allow_deepspeed_feature_hook_experimental", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """CLI main entry."""
    cli_args = parse_args()
    args = build_args(cli_args)
    _require_existing_path("base_model_dir", args.base_model_dir)
    _require_existing_path("stage1_ckpt_dir", args.stage1_ckpt_dir)
    _require_existing_path("dataset_dir", args.dataset_dir)
    _require_existing_path("lingbot_code_dir", args.lingbot_code_dir)
    _require_existing_path("teacher_repo_dir", args.teacher_repo_dir)
    if args.teacher_checkpoint_path:
        _require_existing_path("teacher_checkpoint_path", args.teacher_checkpoint_path)
    else:
        _require_existing_path("teacher_checkpoint_dir", args.teacher_checkpoint_dir)
    configure_logging(Path(args.output_root) / "logs" / f"train_{args.experiment_name}_{args.model_type}.log")
    _configure_cuda_probe_flags()
    set_seed(args.seed)

    runner = TRDTrainingRunner(args)
    runner.initialize_tracking()
    try:
        ok, errors = verify_stage1_checkpoint(args.stage1_ckpt_dir)
        if not ok:
            for error in errors:
                LOGGER.error(error)
            raise SystemExit(1)
        runner.train()
    except Exception as exc:
        LOGGER.exception("Training aborted: %s", exc)
        log_dict(
            0,
            {
                "runtime/error": str(exc),
                "runtime/failed_before_train": 1,
            },
            accelerator=runner.accelerator,
        )
        raise


if __name__ == "__main__":
    main()
