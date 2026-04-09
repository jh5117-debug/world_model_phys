"""Reusable Stage-1 helpers for isolated physical-consistency training."""

from __future__ import annotations

import csv
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


class CSGODataset(Dataset):
    """Load preprocessed CSGO clips for training or validation."""

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        repeat: int = 1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.repeat = repeat

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        with open(csv_path, "r", encoding="utf-8") as handle:
            self.samples = list(csv.DictReader(handle))
        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")
        LOGGER.info("Loaded %s %s samples (repeat=%s)", len(self.samples), split, repeat)

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        import cv2

        sample = self.samples[index % len(self.samples)]
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])
        video_path = os.path.join(clip_dir, "video.mp4")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        video_tensor = torch.stack(frames, dim=1)

        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        actions = np.load(os.path.join(clip_dir, "action.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))

        return {
            "clip_name": os.path.basename(clip_dir),
            "video": video_tensor,
            "prompt": sample["prompt"],
            "poses": torch.from_numpy(self._pad_or_truncate(poses)).float(),
            "actions": torch.from_numpy(self._pad_or_truncate(actions)).float(),
            "intrinsics": torch.from_numpy(self._pad_or_truncate(intrinsics)).float(),
        }

    def _pad_or_truncate(self, array: np.ndarray) -> np.ndarray:
        if len(array) >= self.num_frames:
            return array[: self.num_frames]
        rep_shape = (self.num_frames - len(array),) + (1,) * (array.ndim - 1)
        pad = np.tile(array[-1:], rep_shape)
        return np.concatenate([array, pad], axis=0)


@dataclass(slots=True)
class TimestepSample:
    """One sampled timestep record."""

    index: int
    sigma: float
    timestep: torch.Tensor
    weight: float


class LingBotStage1Helper:
    """Load models and shared components the same way as Stage-1 training."""

    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device("cpu")
        self.boundary = 0.947
        self.num_train_timesteps = 1000
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.shift = 10.0
        self._t5_cache: dict[str, list[torch.Tensor]] = {}
        self._build_schedule()

    def _build_schedule(self) -> None:
        sigmas_linear = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.shift * sigmas_linear / (1 + (self.shift - 1) * sigmas_linear)
        self.timesteps_schedule = self.sigmas * self.num_train_timesteps
        max_timestep = self.boundary * self.num_train_timesteps
        self.low_noise_indices = torch.where(self.timesteps_schedule < max_timestep)[0]
        self.high_noise_indices = torch.where(self.timesteps_schedule >= max_timestep)[0]

        y = torch.exp(-2 * ((self.timesteps_schedule - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
        y_shifted = y - y.min()
        training_weights = y_shifted * (self.num_train_timesteps / y_shifted.sum())
        self.low_noise_weights = training_weights[self.low_noise_indices] / training_weights[self.low_noise_indices].mean()
        self.high_noise_weights = training_weights[self.high_noise_indices] / training_weights[self.high_noise_indices].mean()

    def bootstrap_imports(self) -> None:
        """Import LingBot modules lazily from the shared code checkout."""
        if self.args.lingbot_code_dir not in sys.path:
            sys.path.insert(0, self.args.lingbot_code_dir)
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.utils.cam_utils import (
            compute_relative_poses,
            get_Ks_transformed,
            get_plucker_embeddings,
            interpolate_camera_poses,
        )

        self.WanModel = WanModel
        self.T5EncoderModel = T5EncoderModel
        self.Wan2_1_VAE = Wan2_1_VAE
        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

    def load_model(self, device: torch.device, model_type: str):
        """Load the target Stage-1 model plus VAE and T5."""
        self.bootstrap_imports()
        self.device = device
        subfolder = "low_noise_model" if model_type == "low" else "high_noise_model"
        LOGGER.info("Loading %s from %s", subfolder, self.args.stage1_ckpt_dir)
        model = self.WanModel.from_pretrained(
            self.args.stage1_ckpt_dir,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            control_type="act",
        )
        model.train()
        self.vae = self.Wan2_1_VAE(
            vae_pth=os.path.join(self.args.base_model_dir, "Wan2.1_VAE.pth"),
            device=self.device,
        )
        self.t5 = self.T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=os.path.join(self.args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.args.base_model_dir, "google", "umt5-xxl"),
        )
        return model

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        return self.vae.encode([video_tensor.to(self.device)])[0]

    @torch.no_grad()
    def encode_text(self, prompt: str) -> list[torch.Tensor]:
        if prompt in self._t5_cache:
            return [tensor.to(self.device) for tensor in self._t5_cache[prompt]]
        self.t5.model.to(self.device)
        context = self.t5([prompt], self.device)
        self.t5.model.cpu()
        self._t5_cache[prompt] = [tensor.cpu() for tensor in context]
        return [tensor.to(self.device) for tensor in context]

    @torch.no_grad()
    def prepare_y(self, video_tensor: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        lat_h, lat_w = latent.shape[2], latent.shape[3]
        frame_total = video_tensor.shape[1]
        height, width = video_tensor.shape[2], video_tensor.shape[3]
        first_frame = video_tensor[:, 0:1]
        zeros = torch.zeros(3, frame_total - 1, height, width, device=video_tensor.device)
        y_latent = self.vae.encode([torch.concat([first_frame, zeros], dim=1).to(self.device)])[0]

        mask = torch.ones(1, frame_total, lat_h, lat_w, device=self.device)
        mask[:, 1:] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        mask = mask.transpose(1, 2)[0]
        return torch.concat([mask, y_latent])

    @torch.no_grad()
    def prepare_control_signal(
        self,
        poses: torch.Tensor,
        actions: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        lat_f: int,
        lat_h: int,
        lat_w: int,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        interpolate_camera_poses = self.cam_utils["interpolate_camera_poses"]
        compute_relative_poses = self.cam_utils["compute_relative_poses"]
        get_plucker_embeddings = self.cam_utils["get_plucker_embeddings"]
        get_Ks_transformed = self.cam_utils["get_Ks_transformed"]

        num_frames = poses.shape[0]
        ks = get_Ks_transformed(
            intrinsics,
            height_org=480,
            width_org=832,
            height_resize=height,
            width_resize=width,
            height_final=height,
            width_final=width,
        )
        ks_single = ks[0]
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, num_frames - 1, num_frames),
            src_rot_mat=poses[:, :3, :3].cpu().numpy(),
            src_trans_vec=poses[:, :3, 3].cpu().numpy(),
            tgt_indices=np.linspace(0, num_frames - 1, int((num_frames - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        ks_repeated = ks_single.repeat(len(c2ws_infer), 1).to(self.device)
        c2ws_infer = c2ws_infer.to(self.device)

        wasd = actions[::4].to(self.device)
        if len(wasd) > len(c2ws_infer):
            wasd = wasd[: len(c2ws_infer)]
        elif len(wasd) < len(c2ws_infer):
            wasd = torch.cat([wasd, wasd[-1:].repeat(len(c2ws_infer) - len(wasd), 1)], dim=0)

        plucker = get_plucker_embeddings(c2ws_infer, ks_repeated, height, width, only_rays_d=True)
        plucker = rearrange(
            plucker,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(height // lat_h),
            c2=int(width // lat_w),
        )[None]
        plucker = rearrange(plucker, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(torch.bfloat16)

        wasd_tensor = wasd[:, None, None, :].repeat(1, height, width, 1)
        wasd_tensor = rearrange(
            wasd_tensor,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(height // lat_h),
            c2=int(width // lat_w),
        )[None]
        wasd_tensor = rearrange(
            wasd_tensor,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        ).to(torch.bfloat16)

        return {"c2ws_plucker_emb": (torch.cat([plucker, wasd_tensor], dim=1),)}

    def sample_timestep(self, model_type: str) -> TimestepSample:
        """Sample a valid noise step for the selected model."""
        if model_type == "high":
            indices = self.high_noise_indices
            weights = self.high_noise_weights
        else:
            indices = self.low_noise_indices
            weights = self.low_noise_weights
        local_idx = torch.randint(len(indices), (1,)).item()
        idx = int(indices[local_idx].item())
        sigma = float(self.sigmas[idx].item())
        timestep = self.timesteps_schedule[idx].to(self.device).unsqueeze(0)
        weight = float(weights[local_idx].item())
        return TimestepSample(index=idx, sigma=sigma, timestep=timestep, weight=weight)


def apply_gradient_checkpointing(model, model_name: str = "model") -> None:
    """Apply the same DiT block checkpointing strategy as Stage-1."""
    from functools import wraps
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    patched = 0
    block_container = getattr(model, "blocks", None)
    if block_container is None:
        LOGGER.warning("No transformer blocks found for %s", model_name)
        return
    for block in block_container:
        original_forward = block.forward

        def _make_ckpt(fn):
            @wraps(fn)
            def _wrapped(x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
                return torch_checkpoint(
                    fn,
                    x,
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                    dit_cond_dict,
                    use_reentrant=False,
                )

            return _wrapped

        block.forward = _make_ckpt(original_forward)
        patched += 1
    LOGGER.info("Gradient checkpointing patched %s blocks for %s", patched, model_name)


def save_accelerate_checkpoint(
    *,
    accelerator,
    model,
    args,
    subfolder: str,
    tag: str,
    extra_training_state: dict | None = None,
) -> Path:
    """Save a checkpoint compatible with WanModel.from_pretrained."""
    save_dir = Path(args.output_dir) / tag
    model_dir = save_dir / subfolder
    if accelerator.is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(model, "save_16bit_model"):
        model.save_16bit_model(model_dir, "diffusion_pytorch_model.bin")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_config(model_dir)
    else:
        if accelerator.is_main_process:
            unwrapped.save_pretrained(model_dir, safe_serialization=False)
    if accelerator.is_main_process:
        if extra_training_state:
            training_only = save_dir / "training_only"
            training_only.mkdir(parents=True, exist_ok=True)
            torch.save(extra_training_state, training_only / f"{subfolder}_extras.pt")
    accelerator.wait_for_everyone()
    return save_dir


def build_dataloader(
    dataset_dir: str,
    *,
    split: str,
    num_frames: int,
    height: int,
    width: int,
    repeat: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a standard Stage-1 dataloader."""
    dataset = CSGODataset(
        dataset_dir,
        split=split,
        num_frames=num_frames,
        height=height,
        width=width,
        repeat=repeat,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda items: items[0],
    )


def compute_scheduler_total_steps(dataset_len: int, num_processes: int, grad_accum: int, num_epochs: int) -> int:
    """Mirror the optimizer-step counting logic from Stage-1."""
    iters_per_epoch = math.ceil(dataset_len / num_processes)
    steps_per_epoch = max(iters_per_epoch // grad_accum, 1)
    return max(num_epochs * steps_per_epoch, 1)
