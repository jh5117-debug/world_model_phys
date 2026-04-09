"""Standalone TRD-v1 trainer that does not modify the shared Stage-1 code."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from datetime import timedelta

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import DistributedType
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)

from physical_consistency.common.defaults import CONFIG_DIR, PROJECT_ROOT
from physical_consistency.common.io import ensure_dir, read_yaml, write_json, write_yaml
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.seed import set_seed
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.lineage.contract import LineageRecord, verify_stage1_checkpoint
from physical_consistency.losses.trd import TokenRelationDistillationLoss
from physical_consistency.teachers.videomaev2 import VideoMAEv2Teacher
from physical_consistency.trainers.hooks import BlockFeatureHook
from physical_consistency.trainers.stage1_components import (
    LingBotStage1Helper,
    apply_gradient_checkpointing,
    build_dataloader,
    compute_scheduler_total_steps,
    save_accelerate_checkpoint,
)
from physical_consistency.wandb_utils.media import relation_matrix_image
from physical_consistency.wandb_utils.session import init_wandb_run, log_dict

LOGGER = logging.getLogger(__name__)


class StudentProjector(nn.Module):
    """Project student tokens to the teacher feature dimension."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class TRDTrainingRunner:
    """End-to-end trainer for `low` or `high` Stage-1 continuation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        project_config = ProjectConfiguration(
            project_dir=str(Path(args.output_dir)),
            logging_dir=str(Path(args.output_root) / "logs" / "wandb"),
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl",
            timeout=timedelta(hours=args.distributed_timeout_hours),
        )
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            log_with="wandb",
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
        self.run = None
        self._tracking_initialized = False

    def initialize_tracking(self) -> None:
        """Create the single shared W&B run before model loading."""
        if self._tracking_initialized:
            return
        run_config = vars(self.args).copy()
        self.run = init_wandb_run(
            accelerator=self.accelerator,
            project=self.args.project_name,
            run_name=f"{self.args.experiment_name}_{self.args.model_type}",
            config=run_config,
            wandb_dir=self.args.wandb_dir,
            tags=["physical_consistency", self.args.model_type, "trd_v1", "videorepa_inspired"],
            group=self.args.run_group,
            job_type=f"train_{self.args.model_type}",
            mode=self.args.wandb_mode,
        )
        self._tracking_initialized = True

    def validate_runtime_stack(self) -> None:
        """Reject unsafe runtime combinations before the actual training starts."""
        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            and not self.args.allow_deepspeed_feature_hook_experimental
        ):
            raise RuntimeError(
                "DeepSpeed + forward-hook TRD is disabled by default in this subproject. "
                "The hook-based student feature capture path has not been validated on the real 8-GPU cluster "
                "under ZeRO sharding. Use the default DDP accelerate config, or explicitly opt in with "
                "--allow_deepspeed_feature_hook_experimental true after cluster smoke testing."
            )

    def train(self) -> None:
        """Main training loop."""
        self.validate_runtime_stack()
        if self.accelerator.is_main_process:
            ensure_dir(self.args.output_dir)
        self.accelerator.wait_for_everyone()

        teacher_checkpoint_path = _resolve_teacher_checkpoint(
            self.args.teacher_checkpoint_dir,
            self.args.teacher_checkpoint_path,
        )
        self.model = self.helper.load_model(self.accelerator.device, self.args.model_type)
        self.teacher = VideoMAEv2Teacher(
            repo_dir=self.args.teacher_repo_dir,
            checkpoint_path=teacher_checkpoint_path,
            device=self.accelerator.device,
            model_variant=self.args.teacher_model_variant,
            image_size=self.args.teacher_image_size,
            align_video_resolution=(self.args.teacher_height, self.args.teacher_width),
            pretrained_num_frames=self.args.teacher_pretrained_frames,
            teacher_input_frames=self.args.teacher_input_frames,
            drop_first_frame=self.args.teacher_drop_first_frame,
        )
        student_dim = int(self.model.dim)
        self.projector = StudentProjector(student_dim, self.teacher.feature_dim)

        self.feature_hook = BlockFeatureHook()
        self.feature_hook.attach(self.model.blocks[self.args.student_target_block])

        if self.args.gradient_checkpointing:
            apply_gradient_checkpointing(self.model, self.args.model_type)

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
        self.val_loader = val_loader
        train_dataset_len = len(train_loader.dataset)

        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        total_steps = compute_scheduler_total_steps(
            train_dataset_len,
            self.accelerator.num_processes,
            self.args.gradient_accumulation_steps,
            self.args.num_epochs,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        self.model, self.projector, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model,
            self.projector,
            optimizer,
            train_loader,
            scheduler,
        )
        self.val_loader = val_loader

        for epoch in range(self.args.num_epochs):
            self.model.train()
            epoch_metrics: list[dict[str, float]] = []
            for batch in train_loader:
                with self.accelerator.accumulate(self.model):
                    metrics = self.training_step(batch)
                    self.accelerator.backward(metrics["loss_total"])
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.projector.parameters()),
                            self.args.max_grad_norm,
                        )
                    else:
                        grad_norm = torch.tensor(0.0, device=self.accelerator.device)
                    optimizer.step()
                    if self.accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad()

                epoch_metrics.append(self._scalarize_metrics(metrics))
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        self._log_train_metrics(metrics, scheduler, epoch, grad_norm)
                    if self.global_step % self.args.validation_every_steps == 0:
                        self.run_validation_cycle(tag=f"step_{self.global_step}")

            checkpoint_path = save_accelerate_checkpoint(
                accelerator=self.accelerator,
                model=self.model,
                args=self.args,
                subfolder=self.args.subfolder,
                tag=f"epoch_{epoch + 1}",
                extra_training_state={
                    "projector": self.accelerator.unwrap_model(self.projector).state_dict(),
                    "global_step": self.global_step,
                    "epoch": epoch + 1,
                },
            )
            if self.accelerator.is_main_process:
                self._write_lineage(checkpoint_path)
                self._log_epoch_summary(epoch, epoch_metrics, checkpoint_path)

        final_path = save_accelerate_checkpoint(
            accelerator=self.accelerator,
            model=self.model,
            args=self.args,
            subfolder=self.args.subfolder,
            tag="final",
            extra_training_state={
                "projector": self.accelerator.unwrap_model(self.projector).state_dict(),
                "global_step": self.global_step,
                "epoch": self.args.num_epochs,
            },
        )
        if self.accelerator.is_main_process:
            self._write_lineage(final_path)
            self._write_pending_eval_commands(final_path)
        self.accelerator.end_training()

    def training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute FM + TRD for one batch."""
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
            teacher_features = self.teacher.encode(video.unsqueeze(0))

        self.feature_hook.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(
                [noisy_latent],
                t=timestep_sample.timestep,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond,
            )[0]
        if self.feature_hook.latest is None:
            raise RuntimeError("Feature hook did not capture student tokens.")

        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        loss_fm = F.mse_loss(pred_rest.float(), target_rest.float()) * timestep_sample.weight

        student_tokens = self._reshape_student_tokens(self.feature_hook.latest, lat_f, lat_h, lat_w)
        student_tokens = self.projector(student_tokens)
        trd_output = self.trd_loss(student_tokens, teacher_features.tokens)
        loss_total = loss_fm + self.args.lambda_trd * trd_output.total

        metrics = {
            "loss_total": loss_total,
            "loss_fm": loss_fm.detach(),
            "loss_trd": trd_output.total.detach(),
            "loss_trd_spatial": trd_output.spatial.detach(),
            "loss_trd_temporal": trd_output.temporal.detach(),
            "sample_sigma": torch.tensor(timestep_sample.sigma, device=self.accelerator.device),
            "sample_timestep": timestep_sample.timestep.detach().float().mean(),
            "teacher_feat_norm": teacher_features.tokens.norm(dim=-1).mean().detach(),
            "student_feat_norm": student_tokens.norm(dim=-1).mean().detach(),
            "pred_target_cosine": F.cosine_similarity(
                pred_rest.flatten().float().unsqueeze(0),
                target_rest.flatten().float().unsqueeze(0),
                dim=1,
            ).mean().detach(),
            "_spatial_student": trd_output.spatial_student,
            "_spatial_teacher": trd_output.spatial_teacher,
            "_temporal_student": trd_output.temporal_student,
            "_temporal_teacher": trd_output.temporal_teacher,
        }
        metrics["loss_total"] = loss_total
        return metrics

    def run_validation_cycle(self, tag: str) -> None:
        """Run synchronized light validation and export deterministic benchmark bundles."""
        self.accelerator.wait_for_everyone()
        if self._should_run_inprocess_validation():
            self.run_light_validation(tag=tag)
        elif self.accelerator.is_main_process:
            log_dict(
                self.global_step,
                {
                    "val/inprocess_skipped": 1,
                    "val/inprocess_skip_reason": self._validation_skip_reason(),
                },
                accelerator=self.accelerator,
            )
        self.accelerator.wait_for_everyone()
        checkpoint_path = save_accelerate_checkpoint(
            accelerator=self.accelerator,
            model=self.model,
            args=self.args,
            subfolder=self.args.subfolder,
            tag=tag,
            extra_training_state={
                "projector": self.accelerator.unwrap_model(self.projector).state_dict(),
                "global_step": self.global_step,
                "tag": tag,
            },
        )
        if self.accelerator.is_main_process:
            self._write_lineage(checkpoint_path)
            self._export_validation_request(checkpoint_path, tag)
        self.accelerator.wait_for_everyone()

    def _should_run_inprocess_validation(self) -> bool:
        return (
            self.args.validation_runtime_mode == "in_process"
            and self.accelerator.num_processes == 1
            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
        )

    def _validation_skip_reason(self) -> str:
        if self.args.validation_runtime_mode != "in_process":
            return "snapshot_only_mode"
        if self.accelerator.num_processes != 1:
            return "multi_process_runtime"
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            return "deepspeed_not_supported"
        return "unknown"

    def run_light_validation(self, tag: str) -> None:
        """Run lightweight loss validation inside the current training process."""
        self.model.eval()
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
                    "val/tag_index": self.global_step,
                },
                accelerator=self.accelerator,
            )
        self.model.train()

    def _reshape_student_tokens(
        self,
        tokens: torch.Tensor,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        channels = tokens.shape[-1]
        pooled_h = lat_h // self.helper.patch_size[1]
        pooled_w = lat_w // self.helper.patch_size[2]
        return tokens.view(1, lat_f, pooled_h * pooled_w, channels)

    def _scalarize_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        """Drop non-scalar tensors before epoch aggregation."""
        output: dict[str, float] = {}
        for key, value in metrics.items():
            if not torch.is_tensor(value):
                continue
            if value.ndim != 0:
                continue
            output[key] = float(value.detach().item())
        return output

    def _log_train_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        scheduler,
        epoch: int,
        grad_norm: torch.Tensor,
    ) -> None:
        payload = {
            "train/loss_total": float(metrics["loss_total"].detach().item()),
            "train/loss_fm": float(metrics["loss_fm"].detach().item()),
            "train/loss_trd": float(metrics["loss_trd"].detach().item()),
            "train/loss_trd_spatial": float(metrics["loss_trd_spatial"].detach().item()),
            "train/loss_trd_temporal": float(metrics["loss_trd_temporal"].detach().item()),
            "train/lr": float(scheduler.get_last_lr()[0]),
            "train/grad_norm": float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm),
            "train/global_step": self.global_step,
            "train/epoch": epoch + 1,
            "train/sample_sigma": float(metrics["sample_sigma"].detach().item()),
            "train/sample_timestep": float(metrics["sample_timestep"].detach().item()),
            "train/teacher_feat_norm": float(metrics["teacher_feat_norm"].detach().item()),
            "train/student_feat_norm": float(metrics["student_feat_norm"].detach().item()),
            "train/pred_target_cosine": float(metrics["pred_target_cosine"].detach().item()),
        }
        spatial_student = relation_matrix_image(metrics["_spatial_student"].numpy(), "student_spatial")
        spatial_teacher = relation_matrix_image(metrics["_spatial_teacher"].numpy(), "teacher_spatial")
        temporal_student = relation_matrix_image(metrics["_temporal_student"].numpy(), "student_temporal")
        temporal_teacher = relation_matrix_image(metrics["_temporal_teacher"].numpy(), "teacher_temporal")
        if spatial_student is not None:
            payload["train/spatial_relation_student"] = spatial_student
        if spatial_teacher is not None:
            payload["train/spatial_relation_teacher"] = spatial_teacher
        if temporal_student is not None:
            payload["train/temporal_relation_student"] = temporal_student
        if temporal_teacher is not None:
            payload["train/temporal_relation_teacher"] = temporal_teacher
        log_dict(self.global_step, payload, accelerator=self.accelerator)

    def _log_epoch_summary(self, epoch: int, epoch_metrics: list[dict[str, float]], checkpoint_path: Path) -> None:
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
        if self.run is not None:
            artifact_payload = {
                "epoch/checkpoint_dir": str(checkpoint_path),
                "epoch": epoch + 1,
            }
            log_dict(self.global_step, artifact_payload)

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
        branch_dir = checkpoint_path / self.args.subfolder
        record.write(branch_dir / "lineage.json")

    def _write_pending_eval_commands(self, checkpoint_path: Path) -> None:
        branch_dir = checkpoint_path / self.args.subfolder
        output_dir = branch_dir / "post_train_eval"
        ensure_dir(output_dir)
        bundle_dir = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=checkpoint_path,
            output_root=self.args.output_root,
            experiment_name=f"{self.args.experiment_name}_final",
            stage1_ckpt_dir=self.args.stage1_ckpt_dir,
            allow_stage1_fallback=True,
        )
        eval_config_path = output_dir / "eval_trd_final.yaml"
        eval_config = {
            "experiment_name": self.args.experiment_name,
            "seed_list": self.args.validation_seed_list,
            "split": "val",
            "manifest_path": self.args.manifest_full_val,
            "frame_num": self.args.num_frames,
            "sample_steps": self.args.sample_steps,
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
            "allow_stage1_fallback": True,
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
        """Export exact benchmark commands and metadata for this validation snapshot."""
        branch_dir = checkpoint_path / self.args.subfolder
        export_dir = branch_dir / "validation_export"
        ensure_dir(export_dir)
        bundle_dir = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=checkpoint_path,
            output_root=self.args.output_root,
            experiment_name=f"{self.args.experiment_name}_{tag}",
            stage1_ckpt_dir=self.args.stage1_ckpt_dir,
            allow_stage1_fallback=True,
        )
        eval_config_path = export_dir / "eval_trd_snapshot.yaml"
        eval_config = {
            "experiment_name": f"{self.args.experiment_name}_{tag}",
            "seed_list": self.args.validation_seed_list,
            "split": "val",
            "manifest_path": self.args.manifest_mini_val,
            "frame_num": self.args.num_frames,
            "sample_steps": self.args.sample_steps,
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
            "allow_stage1_fallback": True,
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
            "note": (
                "This export is the deterministic benchmark handoff for the current "
                "training snapshot. Full CSGO/VideoPhy2 benchmark execution is kept "
                "outside the training process because the shared evaluation stack "
                "requires the same GPUs already occupied by 8-GPU training."
            ),
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

    payload.setdefault("margin", 0.1)
    payload.setdefault("teacher_height", 160)
    payload.setdefault("teacher_width", 240)
    payload.setdefault("teacher_pretrained_frames", 16)
    payload.setdefault("teacher_input_frames", 49)
    payload.setdefault("teacher_drop_first_frame", True)
    payload.setdefault("teacher_model_variant", "vit_base_patch16_224")
    payload.setdefault("validation_every_steps", 300)
    payload.setdefault("mini_val_max_samples", 8)
    payload.setdefault("student_target_block", 20)
    payload.setdefault("relation_tokens", 64)
    payload.setdefault("lambda_trd", 0.1)
    payload.setdefault("lambda_spatial", 1.0)
    payload.setdefault("lambda_temporal", 1.0)
    payload.setdefault("run_group", "trd_v1")
    payload.setdefault("wandb_mode", "online")
    payload.setdefault("distributed_timeout_hours", 8)
    payload.setdefault("validation_runtime_mode", "snapshot_only")
    payload.setdefault("allow_deepspeed_feature_hook_experimental", False)
    payload["allow_deepspeed_feature_hook_experimental"] = _coerce_bool(
        payload["allow_deepspeed_feature_hook_experimental"]
    )

    payload["experiment_name"] = payload["experiment_name"]
    payload["subfolder"] = "low_noise_model" if payload["model_type"] == "low" else "high_noise_model"
    payload["output_dir"] = str(Path(payload["output_root"]) / "checkpoints" / payload["experiment_name"])
    payload.setdefault("teacher_checkpoint_path", "")
    payload["config_path"] = str(Path(cli_args.config).resolve())
    payload["config_hash"] = hashlib.sha256(Path(cli_args.config).read_bytes()).hexdigest()[:16]
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
    candidates = sorted(path.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No teacher checkpoint .pth found under {teacher_dir}")
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
    parser.add_argument("--model_type", type=str, default="", choices=["low", "high"])
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
    parser.add_argument("--teacher_repo_dir", type=str, default="")
    parser.add_argument("--teacher_checkpoint_dir", type=str, default="")
    parser.add_argument("--teacher_checkpoint_path", type=str, default="")
    parser.add_argument("--validation_runtime_mode", type=str, default="")
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
