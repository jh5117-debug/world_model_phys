"""Standalone TRD-v1 trainer that does not modify the shared Stage-1 code."""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import accelerate
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
from physical_consistency.common.io import ensure_dir, read_json, read_yaml, write_json, write_yaml
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.seed import set_seed
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.common.summary_tables import format_videophy2_summary
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.lineage.contract import LineageRecord, verify_stage1_checkpoint
from physical_consistency.losses.trd import TokenRelationDistillationLoss
from physical_consistency.teachers.videomaev2 import VideoMAEv2Teacher
from physical_consistency.trainers.hooks import BlockFeatureHook
from physical_consistency.trainers.stage1_components import (
    MODEL_SUBFOLDERS,
    LingBotStage1Helper,
    apply_gradient_checkpointing,
    build_dataloader,
    compute_scheduler_total_steps,
    get_model_subfolder,
    move_optimizer_state,
    prune_checkpoint_dir,
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
    """End-to-end trainer for dual-branch Stage-1 continuation."""

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
        self.current_epoch = 0
        self.run = None
        self._tracking_initialized = False
        self.teacher_checkpoint_path = ""
        self.train_dataset_len = 0
        self.train_loader = None
        self.val_loader = None
        self.best_metrics_path = Path(args.output_dir) / "best_videophy2.json"
        self.best_checkpoint_path = Path(args.output_dir) / args.best_checkpoint_name
        self.visible_gpu_list = os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            ",".join(str(idx) for idx in range(args.num_gpus)),
        )
        self.best_metrics: dict[str, float] | None = None

    def initialize_tracking(self) -> None:
        """Create the shared W&B run before model loading."""
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
        """Reject unsupported runtime combinations before training starts."""
        if self.args.model_type != "dual":
            raise RuntimeError(
                "TRD-v1 now trains the Stage-1 dual model only. "
                "Use model_type=dual and the dual training wrapper."
            )
        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            and not self.args.allow_deepspeed_feature_hook_experimental
        ):
            raise RuntimeError(
                "DeepSpeed + forward-hook TRD is disabled by default in this subproject. "
                "Use the default DDP accelerate config, or explicitly opt in with "
                "--allow_deepspeed_feature_hook_experimental true after cluster smoke testing."
            )

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
        self._initialize_training_runtime(checkpoint_dir=self.args.stage1_ckpt_dir, resume_state=None)

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch + 1
            self._set_train_mode()
            epoch_metrics: list[dict[str, float]] = []
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.low_model, self.high_model):
                    metrics = self.training_step(batch)
                    self.accelerator.backward(metrics["loss_total"])
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
                    self.optimizer.zero_grad()

                epoch_metrics.append(self._scalarize_metrics(metrics))
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.accelerator.is_main_process:
                        self._log_train_metrics(metrics, self.scheduler, epoch, grad_norm)
                    if self.args.validation_every_steps > 0 and self.global_step % self.args.validation_every_steps == 0:
                        self.run_validation_cycle(tag=f"step_{self.global_step}")

            if self.args.save_every_n_epochs > 0 and self.current_epoch % self.args.save_every_n_epochs == 0:
                epoch_path = save_accelerate_checkpoint(
                    accelerator=self.accelerator,
                    model={
                        "low_noise_model": self.low_model,
                        "high_noise_model": self.high_model,
                    },
                    args=self.args,
                    tag=f"epoch_{self.current_epoch}",
                    extra_training_state={
                        "low_projector": self.accelerator.unwrap_model(self.low_projector).state_dict(),
                        "high_projector": self.accelerator.unwrap_model(self.high_projector).state_dict(),
                        "global_step": self.global_step,
                        "epoch": self.current_epoch,
                    },
                    training_state_filename="projectors.pt",
                )
                if self.accelerator.is_main_process:
                    self._write_lineage(epoch_path)
                    self._log_epoch_summary(epoch, epoch_metrics, epoch_path)
            elif self.accelerator.is_main_process:
                self._log_epoch_summary(epoch, epoch_metrics, None)

        final_path = save_accelerate_checkpoint(
            accelerator=self.accelerator,
            model={
                "low_noise_model": self.low_model,
                "high_noise_model": self.high_model,
            },
            args=self.args,
            tag="final",
            extra_training_state={
                "low_projector": self.accelerator.unwrap_model(self.low_projector).state_dict(),
                "high_projector": self.accelerator.unwrap_model(self.high_projector).state_dict(),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            },
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
        self.train_loader = self.accelerator.prepare(train_loader)
        self.val_loader = val_loader

    def _initialize_training_runtime(
        self,
        *,
        checkpoint_dir: str | Path,
        resume_state: dict[str, Any] | None,
    ) -> None:
        self.low_model = self.helper.load_model(self.accelerator.device, "low", checkpoint_dir=checkpoint_dir)
        self.high_model = self.helper.load_model(self.accelerator.device, "high", checkpoint_dir=checkpoint_dir)
        self.teacher = VideoMAEv2Teacher(
            repo_dir=self.args.teacher_repo_dir,
            checkpoint_path=self.teacher_checkpoint_path,
            device=self.accelerator.device,
            model_variant=self.args.teacher_model_variant,
            image_size=self.args.teacher_image_size,
            align_video_resolution=(self.args.teacher_height, self.args.teacher_width),
            pretrained_num_frames=self.args.teacher_pretrained_frames,
            teacher_input_frames=self.args.teacher_input_frames,
            drop_first_frame=self.args.teacher_drop_first_frame,
        )
        student_dim = int(self.low_model.dim)
        self.low_projector = StudentProjector(student_dim, self.teacher.feature_dim)
        self.high_projector = StudentProjector(student_dim, self.teacher.feature_dim)
        if resume_state:
            self.low_projector.load_state_dict(resume_state["low_projector"])
            self.high_projector.load_state_dict(resume_state["high_projector"])

        self.low_hook = BlockFeatureHook()
        self.high_hook = BlockFeatureHook()
        self.low_hook.attach(self.low_model.blocks[self.args.student_target_block])
        self.high_hook.attach(self.high_model.blocks[self.args.student_target_block])

        if self.args.gradient_checkpointing:
            apply_gradient_checkpointing(self.low_model, "low_noise_model")
            apply_gradient_checkpointing(self.high_model, "high_noise_model")

        optimizer = torch.optim.AdamW(
            list(self.low_model.parameters())
            + list(self.high_model.parameters())
            + list(self.low_projector.parameters())
            + list(self.high_projector.parameters()),
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

        (
            self.low_model,
            self.high_model,
            self.low_projector,
            self.high_projector,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.low_model,
            self.high_model,
            self.low_projector,
            self.high_projector,
            optimizer,
            scheduler,
        )
        if resume_state:
            self.optimizer.load_state_dict(resume_state["optimizer"])
            wrapped_optimizer = self._wrapped_optimizer()
            move_optimizer_state(wrapped_optimizer, self.accelerator.device)
            self.scheduler.load_state_dict(resume_state["scheduler"])

    def _wrapped_optimizer(self):
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _all_trainable_params(self) -> list[torch.nn.Parameter]:
        return (
            list(self.low_model.parameters())
            + list(self.high_model.parameters())
            + list(self.low_projector.parameters())
            + list(self.high_projector.parameters())
        )

    def _set_train_mode(self) -> None:
        self.low_model.train()
        self.high_model.train()
        self.low_projector.train()
        self.high_projector.train()

    def _active_components(self, branch: str):
        if branch == "high":
            return self.high_model, self.high_projector, self.high_hook
        return self.low_model, self.low_projector, self.low_hook

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

        active_model, active_projector, active_hook = self._active_components(timestep_sample.branch)
        self.low_hook.clear()
        self.high_hook.clear()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = active_model(
                [noisy_latent],
                t=timestep_sample.timestep,
                context=context,
                seq_len=seq_len,
                y=[y],
                dit_cond_dict=dit_cond,
            )[0]
        if active_hook.latest is None:
            raise RuntimeError("Feature hook did not capture student tokens.")

        pred_rest = pred[:, 1:]
        target_rest = target[:, 1:]
        loss_fm = F.mse_loss(pred_rest.float(), target_rest.float()) * timestep_sample.weight

        student_tokens = self._reshape_student_tokens(active_hook.latest, lat_f, lat_h, lat_w)
        student_tokens = active_projector(student_tokens)
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
            "active_branch_is_high": torch.tensor(
                1 if timestep_sample.branch == "high" else 0,
                device=self.accelerator.device,
            ),
            "_spatial_student": trd_output.spatial_student,
            "_spatial_teacher": trd_output.spatial_teacher,
            "_temporal_student": trd_output.temporal_student,
            "_temporal_teacher": trd_output.temporal_teacher,
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
        checkpoint_path = save_accelerate_checkpoint(
            accelerator=self.accelerator,
            model={
                "low_noise_model": self.low_model,
                "high_noise_model": self.high_model,
            },
            args=self.args,
            tag=tag,
            extra_training_state={
                "low_projector": self.accelerator.unwrap_model(self.low_projector).state_dict(),
                "high_projector": self.accelerator.unwrap_model(self.high_projector).state_dict(),
                "global_step": self.global_step,
                "tag": tag,
            },
            training_state_filename="projectors.pt",
        )
        if self.accelerator.is_main_process:
            self._write_lineage(checkpoint_path)
            self._export_validation_request(checkpoint_path, tag)
        self.accelerator.wait_for_everyone()

    def _run_pause_external_validation_cycle(self, tag: str) -> None:
        self.accelerator.wait_for_everyone()
        candidate_tag = f"_candidate_{tag}"
        candidate_path = save_accelerate_checkpoint(
            accelerator=self.accelerator,
            model={
                "low_noise_model": self.low_model,
                "high_noise_model": self.high_model,
            },
            args=self.args,
            tag=candidate_tag,
            extra_training_state=self._build_resume_state_payload(tag),
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
                validation_error_path.write_text(str(exc), encoding="utf-8")

        self.accelerator.wait_for_everyone()
        self._restore_training_runtime(candidate_path)
        self.accelerator.wait_for_everyone()

        if validation_error_path.exists():
            error_message = validation_error_path.read_text(encoding="utf-8").strip() or "Validation failed"
            if self.accelerator.is_main_process:
                prune_checkpoint_dir(candidate_path)
            self.accelerator.wait_for_everyone()
            raise RuntimeError(error_message)

        if self.accelerator.is_main_process:
            summary = read_json(summary_path)
            self._retain_candidate_checkpoint(candidate_path, tag, summary)
        self.accelerator.wait_for_everyone()

    def _build_resume_state_payload(self, tag: str) -> dict[str, Any]:
        return {
            "low_projector": self.accelerator.unwrap_model(self.low_projector).state_dict(),
            "high_projector": self.accelerator.unwrap_model(self.high_projector).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "tag": tag,
        }

    def _release_training_runtime(self) -> None:
        self.low_hook = None
        self.high_hook = None
        self.teacher = None
        self.helper.release_runtime_components()
        (
            self.low_model,
            self.high_model,
            self.low_projector,
            self.high_projector,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.free_memory(
            self.low_model,
            self.high_model,
            self.low_projector,
            self.high_projector,
            self.optimizer,
            self.scheduler,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_training_runtime(self, checkpoint_path: Path) -> None:
        resume_state = torch.load(
            checkpoint_path / "training_only" / "resume_state.pt",
            map_location="cpu",
        )
        self._initialize_training_runtime(checkpoint_dir=checkpoint_path, resume_state=resume_state)
        self.global_step = int(resume_state.get("global_step", self.global_step))
        self.current_epoch = int(resume_state.get("epoch", self.current_epoch))

    def _run_external_videophy2_validation(self, checkpoint_path: Path, tag: str) -> dict[str, Any]:
        validation_root = ensure_dir(
            Path(self.args.output_root) / "runs" / "train_validation" / self.args.experiment_name / tag
        )
        generation_root = ensure_dir(validation_root / "generated")
        validation_seed = int(self.args.validation_seed_list[0])

        run_command(
            self._validation_generation_command(checkpoint_path, generation_root, validation_seed),
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
            str(self.args.sample_steps),
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
        self.low_model.eval()
        self.high_model.eval()
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
            "train/active_branch_is_high": float(metrics["active_branch_is_high"].detach().item()),
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
        for subfolder in MODEL_SUBFOLDERS:
            record.write(checkpoint_path / subfolder / "lineage.json")

    def _write_pending_eval_commands(self, checkpoint_path: Path) -> None:
        output_dir = checkpoint_path / "post_train_eval"
        ensure_dir(output_dir)
        bundle_dir = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=checkpoint_path,
            output_root=self.args.output_root,
            experiment_name=f"{self.args.experiment_name}_final",
            stage1_ckpt_dir=self.args.stage1_ckpt_dir,
            allow_stage1_fallback=False,
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
            "allow_stage1_fallback": False,
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
        bundle_dir = materialize_eval_checkpoint_bundle(
            ft_ckpt_dir=checkpoint_path,
            output_root=self.args.output_root,
            experiment_name=f"{self.args.experiment_name}_{tag}",
            stage1_ckpt_dir=self.args.stage1_ckpt_dir,
            allow_stage1_fallback=False,
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
            "allow_stage1_fallback": False,
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
    payload.setdefault("validation_runtime_mode", "pause_external")
    payload.setdefault("allow_deepspeed_feature_hook_experimental", False)
    payload.setdefault("best_checkpoint_name", "best_videophy2")
    payload["allow_deepspeed_feature_hook_experimental"] = _coerce_bool(
        payload["allow_deepspeed_feature_hook_experimental"]
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
