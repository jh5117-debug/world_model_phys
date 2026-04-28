"""Pure Stage-1 PhysInOne camera-only trainer."""

from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import accelerate
import torch
import torch.nn.functional as F
from accelerate import DistributedType
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
from torch.utils.data import DataLoader

from physical_consistency.common.io import ensure_dir, write_json
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.trainers.stage1_components import (
    LingBotStage1Helper,
    apply_gradient_checkpointing,
    compute_scheduler_total_steps,
    configure_stage1_precision_env,
    export_pretrained_state_dict,
    resolve_stage1_low_precision_dtype,
)

from .config import Stage1PhysInOneConfig
from .dataset import PhysInOneCamDataset
from .eval import run_stage1_videophy2_eval

LOGGER = logging.getLogger(__name__)


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _isoformat_local(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


@dataclass(slots=True)
class BranchTrainResult:
    """Outputs from one branch-specific Stage-1 run."""

    branch: str
    final_branch_dir: str
    final_eval_bundle_dir: str
    output_dir: str
    started_at: str
    finished_at: str
    duration_seconds: float
    global_step: int
    micro_step: int


class Stage1BranchTrainer:
    """Train one LingBot MoE branch with pure camera conditioning and FM loss."""

    def __init__(
        self,
        cfg: Stage1PhysInOneConfig,
        *,
        branch: str,
        source_checkpoint_dir: str,
        companion_checkpoint_dir: str,
    ) -> None:
        if branch not in {"low", "high"}:
            raise ValueError(f"Unsupported branch: {branch}")
        self.cfg = cfg
        self.branch = branch
        self.source_checkpoint_dir = str(Path(source_checkpoint_dir).resolve())
        self.companion_checkpoint_dir = str(Path(companion_checkpoint_dir).resolve())
        self.precision_policy = configure_stage1_precision_env(
            cfg.student_precision_profile,
            cfg.student_low_precision_dtype,
        )
        LOGGER.info(
            "[Stage1][%s] Initializing accelerator (launcher_env rank=%s local_rank=%s world_size=%s)",
            branch,
            os.environ.get("RANK", ""),
            os.environ.get("LOCAL_RANK", ""),
            os.environ.get("WORLD_SIZE", ""),
        )

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=bool(cfg.student_ddp_find_unused_parameters)
        )
        init_kwargs = InitProcessGroupKwargs(
            backend="nccl",
            timeout=timedelta(hours=cfg.distributed_timeout_hours),
        )
        grad_accum_plugin = GradientAccumulationPlugin(
            num_steps=cfg.gradient_accumulation_steps,
            sync_each_batch=True,
        )
        project_config = ProjectConfiguration(
            project_dir=str(Path(cfg.output_dir)),
            logging_dir=str(Path(cfg.output_root) / "logs" / "wandb"),
        )
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_plugin=grad_accum_plugin,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            log_with=None,
            project_config=project_config,
        )
        LOGGER.info(
            "[Stage1][%s] Accelerator ready (distributed_type=%s num_processes=%s device=%s is_main=%s)",
            branch,
            self.accelerator.distributed_type,
            self.accelerator.num_processes,
            self.accelerator.device,
            self.accelerator.is_main_process,
        )
        self.helper = LingBotStage1Helper(cfg)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.global_step = 0
        self.micro_step = 0
        self.total_optimizer_steps = 0
        self.output_dir = Path(cfg.output_dir) / f"{branch}_phase"
        ensure_dir(self.output_dir)

    def run(self) -> BranchTrainResult:
        started_at = _now_local()
        LOGGER.info(
            "[Stage1][%s] Starting branch run at %s (source_ckpt=%s companion_ckpt=%s output_dir=%s)",
            self.branch,
            _isoformat_local(started_at),
            self.source_checkpoint_dir,
            self.companion_checkpoint_dir,
            self.output_dir,
        )
        LOGGER.info(
            "[Stage1][%s] Building dataset from %s (frames=%s size=%sx%s repeat=%s workers=%s)",
            self.branch,
            self.cfg.dataset_dir,
            self.cfg.num_frames,
            self.cfg.height,
            self.cfg.width,
            self.cfg.dataset_repeat,
            self.cfg.num_workers,
        )
        dataset = PhysInOneCamDataset(
            self.cfg.dataset_dir,
            split="train",
            num_frames=self.cfg.num_frames,
            height=self.cfg.height,
            width=self.cfg.width,
            repeat=self.cfg.dataset_repeat,
        )
        LOGGER.info("[Stage1][%s] Dataset ready with %s samples", self.branch, len(dataset))
        LOGGER.info("[Stage1][%s] Constructing raw dataloader", self.branch)
        raw_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=lambda items: items[0],
        )
        train_dataset_len = len(dataset)

        LOGGER.info("[Stage1][%s] Loading student model", self.branch)
        self.model = self.helper.load_model(
            self.accelerator.device,
            self.branch,
            checkpoint_dir=self.source_checkpoint_dir,
            control_type="cam",
        )
        LOGGER.info("[Stage1][%s] Student model loaded", self.branch)
        if self.cfg.gradient_checkpointing:
            LOGGER.info(
                "[Stage1][%s] Applying gradient checkpointing (mode=%s)",
                self.branch,
                self.cfg.student_memory_efficient_checkpoint_mode,
            )
            use_reentrant = self._student_checkpoint_use_reentrant()
            apply_gradient_checkpointing(
                self.model,
                f"{self.branch}_noise_model",
                use_reentrant=use_reentrant,
                skip_block_indices=set(),
                memory_efficient_mode=self.cfg.student_memory_efficient_checkpoint_mode,
            )
            LOGGER.info("[Stage1][%s] Gradient checkpointing ready", self.branch)

        LOGGER.info("[Stage1][%s] Building optimizer and scheduler", self.branch)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=compute_scheduler_total_steps(
                train_dataset_len,
                self.accelerator.num_processes,
                self.cfg.gradient_accumulation_steps,
                self.cfg.num_epochs,
            ),
            eta_min=1.0e-6,
        )
        self.total_optimizer_steps = compute_scheduler_total_steps(
            train_dataset_len,
            self.accelerator.num_processes,
            self.cfg.gradient_accumulation_steps,
            self.cfg.num_epochs,
        )
        LOGGER.info("[Stage1][%s] Preparing model/optimizer/dataloader with accelerator", self.branch)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            optimizer,
            raw_loader,
            scheduler,
        )
        LOGGER.info("[Stage1][%s] Accelerator.prepare complete", self.branch)
        self.accelerator.wait_for_everyone()
        LOGGER.info("[Stage1][%s] Post-prepare barrier complete", self.branch)

        last_eval_bundle = ""
        for epoch in range(self.cfg.num_epochs):
            epoch_index = epoch + 1
            self.model.train()
            for batch in self.train_loader:
                self.micro_step += 1
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.training_step(batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        if self.accelerator.is_main_process:
                            LOGGER.info(
                                "[Stage1][%s] epoch=%s/%s step=%s/%s loss_fm=%.6f lr=%.3e sigma=%.4f",
                                self.branch,
                                epoch_index,
                                self.cfg.num_epochs,
                                self.global_step,
                                self.total_optimizer_steps,
                                float(metrics["loss_fm"]),
                                float(self.scheduler.get_last_lr()[0]),
                                float(metrics["sample_sigma"]),
                            )
                if self.cfg.max_train_micro_steps > 0 and self.micro_step >= self.cfg.max_train_micro_steps:
                    break

            should_save = self.cfg.save_every_n_epochs > 0 and epoch_index % self.cfg.save_every_n_epochs == 0
            should_eval = (
                self.cfg.videophy2_eval.enabled
                and epoch_index % self.cfg.videophy2_eval.every_n_epochs == 0
            )
            checkpoint_root = None
            if should_save or should_eval:
                checkpoint_root = self._save_branch_checkpoint(tag=f"epoch_{epoch_index}")
            if should_eval and checkpoint_root is not None:
                last_eval_bundle = self._run_epoch_eval(epoch_index, checkpoint_root)
            if self.cfg.max_train_micro_steps > 0 and self.micro_step >= self.cfg.max_train_micro_steps:
                break

        final_branch_dir = self._save_branch_checkpoint(tag="final")
        final_bundle = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=final_branch_dir,
            output_root=self.output_dir / "eval_bundles",
            experiment_name=f"{self.cfg.experiment_name}_{self.branch}_final",
            companion_ckpt_dir=self.companion_checkpoint_dir,
        )
        if self.cfg.videophy2_eval.enabled:
            run_stage1_videophy2_eval(
                self.cfg.videophy2_eval,
                bundle_dir=final_bundle,
                output_dir=self.output_dir,
                experiment_name=self.cfg.experiment_name,
                epoch=self.cfg.num_epochs,
                branch=f"{self.branch}_final",
            )
        finished_at = _now_local()
        duration_seconds = (finished_at - started_at).total_seconds()
        if self.accelerator.is_main_process:
            write_json(
                self.output_dir / "branch_summary.json",
                {
                    "branch": self.branch,
                    "source_checkpoint_dir": self.source_checkpoint_dir,
                    "companion_checkpoint_dir": self.companion_checkpoint_dir,
                    "final_branch_dir": str(final_branch_dir),
                    "final_eval_bundle_dir": str(final_bundle),
                    "last_eval_bundle_dir": last_eval_bundle,
                    "started_at": _isoformat_local(started_at),
                    "finished_at": _isoformat_local(finished_at),
                    "duration_seconds": duration_seconds,
                    "global_step": self.global_step,
                    "micro_step": self.micro_step,
                    "config_path": self.cfg.config_path,
                    "config_hash": self.cfg.config_hash,
                },
            )
            LOGGER.info(
                "[Stage1][%s] Branch finished at %s (duration_seconds=%.1f global_step=%s micro_step=%s final_branch_dir=%s final_eval_bundle_dir=%s)",
                self.branch,
                _isoformat_local(finished_at),
                duration_seconds,
                self.global_step,
                self.micro_step,
                final_branch_dir,
                final_bundle,
            )
        self._release_runtime()
        return BranchTrainResult(
            branch=self.branch,
            final_branch_dir=str(final_branch_dir),
            final_eval_bundle_dir=str(final_bundle),
            output_dir=str(self.output_dir),
            started_at=_isoformat_local(started_at),
            finished_at=_isoformat_local(finished_at),
            duration_seconds=duration_seconds,
            global_step=self.global_step,
            micro_step=self.micro_step,
        )

    def training_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        video = batch["video"].to(self.accelerator.device)
        poses = batch["poses"]
        intrinsics = batch["intrinsics"]
        prompt = batch["prompt"]
        height, width = int(video.shape[2]), int(video.shape[3])
        source_height = int(batch.get("source_height", height))
        source_width = int(batch.get("source_width", width))

        with torch.no_grad():
            video_latent = self.helper.encode_video(video)
            context = self.helper.encode_text(prompt)
            y = self.helper.prepare_y(video, video_latent)
            lat_f, lat_h, lat_w = video_latent.shape[1], video_latent.shape[2], video_latent.shape[3]
            seq_len = lat_f * lat_h * lat_w // (self.helper.patch_size[1] * self.helper.patch_size[2])
            dit_cond = self.helper.prepare_control_signal(
                poses,
                None,
                intrinsics,
                height,
                width,
                lat_f,
                lat_h,
                lat_w,
                control_type="cam",
                source_height=source_height,
                source_width=source_width,
            )
            timestep_sample = self.helper.sample_timestep(self.branch)
            noise = torch.randn_like(video_latent)
            noisy_latent = (1.0 - timestep_sample.sigma) * video_latent + timestep_sample.sigma * noise
            target = noise - video_latent

        device_type = self.accelerator.device.type
        if os.environ.get("PC_STAGE1_FORCE_FP32", "").strip().lower() in {"1", "true", "yes", "on"}:
            autocast_ctx = torch.autocast(device_type=device_type, enabled=False)
        else:
            autocast_ctx = (
                torch.amp.autocast(device_type=device_type, dtype=resolve_stage1_low_precision_dtype())
                if device_type == "cuda"
                else torch.autocast(device_type=device_type, enabled=False)
            )
        with autocast_ctx:
            pred = self.model(
                [noisy_latent],
                t=timestep_sample.timestep,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond,
            )[0]
        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        loss_fm = F.mse_loss(pred_rest.float(), target_rest.float()) * timestep_sample.weight
        return loss_fm, {
            "loss_fm": float(loss_fm.detach().item()),
            "sample_sigma": float(timestep_sample.sigma),
        }

    def _student_checkpoint_use_reentrant(self) -> bool:
        if self.cfg.student_checkpoint_use_reentrant is not None:
            return bool(self.cfg.student_checkpoint_use_reentrant)
        if (
            self.accelerator.distributed_type in {DistributedType.MULTI_GPU, DistributedType.DEEPSPEED}
            and self.cfg.student_tuning_mode == "lora"
        ):
            return False
        return self.cfg.student_tuning_mode == "lora"

    def _save_branch_checkpoint(self, *, tag: str) -> Path:
        save_root = self.output_dir / "branches" / tag
        save_root.mkdir(parents=True, exist_ok=True)
        branch_dir = save_root / ("low_noise_model" if self.branch == "low" else "high_noise_model")
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.get_state_dict(self.model)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            ensure_dir(branch_dir)
            torch.save(
                export_pretrained_state_dict(unwrapped, state_dict, prefix=""),
                branch_dir / "diffusion_pytorch_model.bin",
            )
            if hasattr(unwrapped, "save_config"):
                unwrapped.save_config(branch_dir)
        self.accelerator.wait_for_everyone()
        return save_root

    def _run_epoch_eval(self, epoch: int, branch_checkpoint_root: Path) -> str:
        bundle_dir = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=branch_checkpoint_root,
            output_root=self.output_dir / "eval_bundles",
            experiment_name=f"{self.cfg.experiment_name}_{self.branch}_epoch_{epoch}",
            companion_ckpt_dir=self.companion_checkpoint_dir,
        )
        run_stage1_videophy2_eval(
            self.cfg.videophy2_eval,
            bundle_dir=bundle_dir,
            output_dir=self.output_dir,
            experiment_name=self.cfg.experiment_name,
            epoch=epoch,
            branch=self.branch,
        )
        return str(bundle_dir)

    def _release_runtime(self) -> None:
        self.helper.release_runtime_components()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
