"""Reusable Stage-1 helpers for isolated physical-consistency training."""

from __future__ import annotations

import csv
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


def _should_log_rank_zero() -> bool:
    rank = os.environ.get("RANK", "")
    return rank in {"", "0"}


def _patch_wan_rope_apply_to_preserve_dtype(wan_model_module) -> None:
    """Patch Wan RoPE to avoid a large fp32 q/k tensor before flash-attn."""

    original_rope_apply = getattr(wan_model_module, "rope_apply", None)
    if original_rope_apply is None:
        LOGGER.warning("Wan model module has no rope_apply; skipping RoPE dtype patch")
        return
    if getattr(wan_model_module, "_pc_rope_preserve_dtype_patched", False):
        return

    def _rope_apply_preserve_dtype(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        device_type = x.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            n, c = x.size(2), x.size(3) // 2
            split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

            output = []
            for i, (f, h, w) in enumerate(grid_sizes.tolist()):
                f, h, w = int(f), int(h), int(w)
                seq_len = f * h * w

                x_i = torch.view_as_complex(
                    x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2)
                )
                freqs_i = torch.cat(
                    [
                        split_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, 1, -1)

                x_i = torch.view_as_real(x_i * freqs_i.to(torch.complex64)).flatten(2)
                if seq_len < x.size(1):
                    x_i = torch.cat([x_i, x[i, seq_len:].to(dtype=x_i.dtype)], dim=0)
                output.append(x_i.to(out_dtype))
            return torch.stack(output)

    wan_model_module._pc_original_rope_apply = original_rope_apply
    wan_model_module.rope_apply = _rope_apply_preserve_dtype
    wan_model_module._pc_rope_preserve_dtype_patched = True
    if _should_log_rank_zero():
        LOGGER.info("Patched Wan rope_apply to preserve q/k dtype for training memory")


MODEL_SUBFOLDERS = ("low_noise_model", "high_noise_model")
MODEL_TYPE_TO_SUBFOLDER = {
    "low": "low_noise_model",
    "high": "high_noise_model",
}


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
    branch: str


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
        self.all_indices = torch.arange(self.num_train_timesteps)
        self.all_weights = training_weights / training_weights.mean()
        self.low_noise_weights = training_weights[self.low_noise_indices] / training_weights[self.low_noise_indices].mean()
        self.high_noise_weights = training_weights[self.high_noise_indices] / training_weights[self.high_noise_indices].mean()

    def bootstrap_imports(self) -> None:
        """Import LingBot modules lazily from the shared code checkout."""
        if self.args.lingbot_code_dir not in sys.path:
            sys.path.insert(0, self.args.lingbot_code_dir)
        import wan.modules.model as wan_model_module
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_1 import Wan2_1_VAE
        from wan.utils.cam_utils import (
            compute_relative_poses,
            get_Ks_transformed,
            get_plucker_embeddings,
            interpolate_camera_poses,
        )

        _patch_wan_rope_apply_to_preserve_dtype(wan_model_module)
        self.WanModel = WanModel
        self.T5EncoderModel = T5EncoderModel
        self.Wan2_1_VAE = Wan2_1_VAE
        self.cam_utils = {
            "interpolate_camera_poses": interpolate_camera_poses,
            "compute_relative_poses": compute_relative_poses,
            "get_plucker_embeddings": get_plucker_embeddings,
            "get_Ks_transformed": get_Ks_transformed,
        }

    def ensure_runtime_components(self, device: torch.device) -> None:
        """Load the shared VAE/T5 runtime once."""
        self.bootstrap_imports()
        self.device = device
        if getattr(self, "vae", None) is None:
            self.vae = self.Wan2_1_VAE(
                vae_pth=os.path.join(self.args.base_model_dir, "Wan2.1_VAE.pth"),
                device=self.device,
            )
        if getattr(self, "t5", None) is None:
            # Keep T5 on CPU by default and only move it to GPU on cache misses.
            self.t5 = self.T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(self.args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(self.args.base_model_dir, "google", "umt5-xxl"),
            )
            module = getattr(self.t5, "model", None)
            if module is not None and hasattr(module, "to"):
                module.to("cpu")

    def load_model(self, device: torch.device, model_type: str, checkpoint_dir: str | Path | None = None):
        """Load one target Stage-1 branch plus shared VAE/T5 runtime."""
        self.ensure_runtime_components(device)
        subfolder = get_model_subfolder(model_type)
        checkpoint_root = str(checkpoint_dir or self.args.stage1_ckpt_dir)
        LOGGER.info("Loading %s from %s", subfolder, checkpoint_root)
        model = self.WanModel.from_pretrained(
            checkpoint_root,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            control_type="act",
        )
        if getattr(self.args, "student_memory_efficient_modulation", True):
            apply_memory_efficient_wan_block_patch(
                model,
                subfolder,
                ffn_chunk_size=getattr(self.args, "student_ffn_chunk_size", None),
                norm_chunk_size=getattr(self.args, "student_norm_chunk_size", None),
            )
        if getattr(self.args, "student_tuning_mode", "full") == "lora":
            apply_lora_to_wan_model(
                model,
                model_name=subfolder,
                rank=getattr(self.args, "student_lora_rank", 16),
                alpha=getattr(self.args, "student_lora_alpha", 16),
                dropout=getattr(self.args, "student_lora_dropout", 0.0),
                block_start=getattr(self.args, "student_lora_block_start", 0),
                lora_chunk_size=getattr(self.args, "student_lora_chunk_size", None),
            )
        model.train()
        return model

    def release_runtime_components(self) -> None:
        """Drop non-trainable helper runtime objects to free GPU memory."""
        for attr_name in ("vae", "t5"):
            obj = getattr(self, attr_name, None)
            if obj is None:
                continue
            module = getattr(obj, "model", None)
            if module is not None and hasattr(module, "to"):
                try:
                    module.to("cpu")
                except Exception:
                    LOGGER.debug("Failed to move %s.model to CPU before release", attr_name, exc_info=True)
            elif hasattr(obj, "to"):
                try:
                    obj.to("cpu")
                except Exception:
                    LOGGER.debug("Failed to move %s to CPU before release", attr_name, exc_info=True)
            setattr(self, attr_name, None)
        self._t5_cache.clear()
        self.device = torch.device("cpu")

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

        # Wan consumes a 4-channel temporal mask aligned to the latent timeline, not the raw frame count.
        mask = torch.zeros(4, y_latent.shape[1], lat_h, lat_w, device=self.device, dtype=y_latent.dtype)
        mask[:, 0] = 1
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
        if model_type == "dual":
            indices = self.all_indices
            weights = self.all_weights
        elif model_type == "high":
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
        return TimestepSample(
            index=idx,
            sigma=sigma,
            timestep=timestep,
            weight=weight,
            branch=self.branch_for_timestep_index(idx),
        )

    def branch_for_timestep_index(self, timestep_index: int) -> str:
        """Return the model branch implied by one sampled timestep."""
        timestep_value = float(self.timesteps_schedule[timestep_index].item())
        boundary = self.boundary * self.num_train_timesteps
        return "high" if timestep_value >= boundary else "low"


def apply_gradient_checkpointing(
    model,
    model_name: str = "model",
    *,
    use_reentrant: bool = False,
) -> None:
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
                def _forward(*tensor_args):
                    return fn(*tensor_args, dit_cond_dict=dit_cond_dict)

                checkpoint_args = (x, e, seq_lens, grid_sizes, freqs, context, context_lens)
                if use_reentrant and not any(
                    torch.is_tensor(arg) and arg.requires_grad for arg in checkpoint_args
                ):
                    # Reentrant checkpointing drops parameter gradients when no input
                    # requires grad. LoRA tuning freezes the student inputs, so mark
                    # the actual hidden state as a local grad anchor.
                    checkpoint_args = (
                        x.detach().requires_grad_(True),
                        e,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        context,
                        context_lens,
                    )

                checkpoint_kwargs = {"use_reentrant": use_reentrant}
                if not use_reentrant:
                    # ZeRO-3 can expose sharded parameters as empty tensors during
                    # non-reentrant recompute, which trips PyTorch's metadata check
                    # even though DeepSpeed gathers them for the actual math.
                    checkpoint_kwargs["determinism_check"] = "none"

                return torch_checkpoint(
                    _forward,
                    *checkpoint_args,
                    **checkpoint_kwargs,
                )

            return _wrapped

        block.forward = _make_ckpt(original_forward)
        patched += 1
    if _should_log_rank_zero():
        LOGGER.info(
            "Gradient checkpointing patched %s blocks for %s (use_reentrant=%s)",
            patched,
            model_name,
            use_reentrant,
        )


def apply_memory_efficient_wan_block_patch(
    model,
    model_name: str = "model",
    *,
    ffn_chunk_size: int | None = None,
    norm_chunk_size: int | None = None,
) -> None:
    """Avoid full fp32 modulation tensors and optionally chunk FFN activations."""

    if ffn_chunk_size is not None and ffn_chunk_size <= 0:
        ffn_chunk_size = None
    if norm_chunk_size is not None and norm_chunk_size <= 0:
        norm_chunk_size = None

    norm_patched = 0
    if norm_chunk_size is not None:
        norm_patched = _patch_sequence_norm_forward(model, int(norm_chunk_size))

    def _squeeze_gate(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim >= 3 and tensor.shape[2] == 1:
            return tensor.squeeze(2)
        return tensor

    def _modulate(normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        normed = normed.to(target_dtype)
        shift = _squeeze_gate(shift).to(device=normed.device, dtype=target_dtype)
        scale = _squeeze_gate(scale).to(device=normed.device, dtype=target_dtype)
        return normed * (1 + scale) + shift

    def _slice_sequence(tensor: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        if tensor.ndim < 3 or tensor.shape[1] == 1:
            return tensor
        return tensor[:, start:stop]

    def _apply_gated_residual(base: torch.Tensor, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = _squeeze_gate(gate).to(device=value.device, dtype=value.dtype)
        return torch.addcmul(base, value, gate)

    def _run_module_layers(module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        # Running nested Sequentials layer-by-layer avoids a large ZeRO-3 fetch on the
        # outer container, which otherwise gathers the whole FFN at once.
        if isinstance(module, torch.nn.Sequential):
            value = inputs
            for submodule in module:
                value = _run_module_layers(submodule, value)
            return value
        return module(inputs)

    def _run_ffn_residual(
        ffn: torch.nn.Module,
        norm: torch.nn.Module,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if ffn_chunk_size is None or residual.shape[1] <= ffn_chunk_size:
            ffn_input = _modulate(norm(residual), shift, scale, target_dtype)
            y = _run_module_layers(ffn, ffn_input)
            return _apply_gated_residual(residual, y, gate)

        output = torch.empty_like(residual)
        for start in range(0, residual.shape[1], ffn_chunk_size):
            stop = min(start + ffn_chunk_size, residual.shape[1])
            residual_chunk = residual[:, start:stop]
            shift_chunk = _slice_sequence(shift, start, stop)
            scale_chunk = _slice_sequence(scale, start, stop)
            gate_chunk = _slice_sequence(gate, start, stop)
            ffn_input_chunk = _modulate(norm(residual_chunk), shift_chunk, scale_chunk, target_dtype)
            y_chunk = _run_module_layers(ffn, ffn_input_chunk)
            output[:, start:stop] = _apply_gated_residual(residual_chunk, y_chunk, gate_chunk)
        return output

    patched = 0
    block_container = getattr(model, "blocks", None)
    if block_container is None:
        LOGGER.warning("No transformer blocks found for %s", model_name)
        return

    for block in block_container:
        if getattr(block, "_pc_memory_efficient_modulation_patched", False):
            continue
        required_attrs = ("modulation", "norm1", "self_attn", "norm2", "ffn", "norm3", "cross_attn")
        if not all(hasattr(block, attr) for attr in required_attrs):
            continue

        def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None):
            target_dtype = x.dtype
            e = (
                self.modulation.unsqueeze(0).to(device=e.device, dtype=torch.float32)
                + e.to(dtype=torch.float32)
            ).chunk(6, dim=2)

            self_attn_input = _modulate(self.norm1(x), e[0], e[1], target_dtype)
            y = self.self_attn(self_attn_input, seq_lens, grid_sizes, freqs)
            x = _apply_gated_residual(x, y, e[2])

            cross_input = self.norm3(x).to(target_dtype)
            x = x + self.cross_attn(cross_input, context, context_lens)

            x = _run_ffn_residual(
                self.ffn,
                self.norm2,
                x,
                e[3],
                e[4],
                e[5],
                target_dtype,
            )
            return x

        block.forward = MethodType(_forward, block)
        block._pc_memory_efficient_modulation_patched = True
        patched += 1

    if patched:
        if _should_log_rank_zero():
            LOGGER.info(
                "Memory-efficient modulation patched %s blocks for %s (ffn_chunk_size=%s)",
                patched,
                model_name,
                ffn_chunk_size or "disabled",
            )
            if norm_patched:
                LOGGER.info(
                    "Memory-efficient Wan sequence norms patched %s modules for %s (norm_chunk_size=%s)",
                    norm_patched,
                    model_name,
                    norm_chunk_size,
                )
    else:
        LOGGER.warning("No Wan attention blocks matched modulation patch for %s", model_name)


def _patch_sequence_norm_forward(model: torch.nn.Module, chunk_size: int) -> int:
    """Chunk Wan norm modules that internally cast long sequences to fp32."""

    patched = 0
    target_class_names = {"WanRMSNorm", "WanLayerNorm"}
    for module in model.modules():
        if module.__class__.__name__ not in target_class_names:
            continue
        if getattr(module, "_pc_sequence_norm_chunk_patched", False):
            continue
        original_forward = module.forward

        def _make_forward(fn):
            def _forward(self, x: torch.Tensor):
                if not torch.is_tensor(x) or x.ndim < 3 or x.shape[1] <= chunk_size:
                    return fn(x)
                outputs = []
                for start in range(0, x.shape[1], chunk_size):
                    stop = min(start + chunk_size, x.shape[1])
                    outputs.append(fn(x[:, start:stop]))
                return torch.cat(outputs, dim=1)

            return _forward

        module.forward = MethodType(_make_forward(original_forward), module)
        module._pc_sequence_norm_chunk_patched = True
        patched += 1
    return patched


class LoRALinear(torch.nn.Module):
    """Standard LoRA wrapper for nn.Linear."""

    def __init__(
        self,
        base: torch.nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(self.alpha) / float(self.rank)
        self.chunk_size = int(chunk_size) if chunk_size is not None and chunk_size > 0 else None
        self.dropout = torch.nn.Dropout(float(dropout)) if dropout > 0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = torch.nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.chunk_size is None or x.ndim < 3 or x.shape[1] <= self.chunk_size:
            lora_out = self.lora_B(self.lora_A(self.dropout(x)))
            return base_out.add_(lora_out, alpha=self.scaling)

        for start in range(0, x.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, x.shape[1])
            lora_chunk = self.lora_B(self.lora_A(self.dropout(x[:, start:stop])))
            base_out[:, start:stop].add_(lora_chunk, alpha=self.scaling)
        return base_out


def _iter_lora_modules(model: torch.nn.Module) -> list[tuple[str, LoRALinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def apply_lora_to_wan_model(
    model: torch.nn.Module,
    *,
    model_name: str = "model",
    rank: int,
    alpha: int,
    dropout: float,
    target_prefixes: tuple[str, ...] = ("blocks",),
    block_start: int = 0,
    lora_chunk_size: int | None = None,
) -> None:
    """Replace selected Wan linear layers with standard LoRA adapters."""

    block_start = int(block_start)
    if block_start < 0:
        raise ValueError(f"LoRA block_start must be non-negative, got {block_start}")
    if lora_chunk_size is not None and lora_chunk_size <= 0:
        lora_chunk_size = None

    def _block_index(full_name: str) -> int | None:
        parts = full_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def _is_target_module(full_name: str) -> bool:
        if not any(full_name == prefix or full_name.startswith(f"{prefix}.") for prefix in target_prefixes):
            return False
        index = _block_index(full_name)
        return index is None or index >= block_start

    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not full_name or not _is_target_module(full_name):
            continue
        if ".base" in full_name or ".lora_" in full_name:
            continue
        if isinstance(module, LoRALinear) or not isinstance(module, torch.nn.Linear):
            continue
        parent_name, child_name = full_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        parent._modules[child_name] = LoRALinear(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            chunk_size=lora_chunk_size,
        )
        replaced += 1

    if replaced == 0:
        raise RuntimeError(f"No linear layers matched LoRA targets for {model_name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_params = 0
    total_params = 0
    for _, module in _iter_lora_modules(model):
        module.lora_A.weight.requires_grad = True
        module.lora_B.weight.requires_grad = True
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    model._pc_lora_config = {
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_prefixes": tuple(target_prefixes),
        "block_start": block_start,
        "lora_chunk_size": lora_chunk_size,
    }
    if _should_log_rank_zero():
        LOGGER.info(
            "Applied standard LoRA to %s linear layers for %s (rank=%s, alpha=%s, dropout=%.3f, block_start=%s, lora_chunk_size=%s, trainable=%s/%s)",
            replaced,
            model_name,
            rank,
            alpha,
            dropout,
            block_start,
            lora_chunk_size or "disabled",
            trainable_params,
            total_params,
        )


def extract_lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return just the trainable LoRA adapter tensors for one model."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_A.weight" in key or ".lora_B.weight" in key
    }


def load_lora_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], *, model_name: str = "model") -> None:
    """Load adapter-only weights into a LoRA-patched model."""
    if not state_dict:
        LOGGER.warning("No LoRA weights provided for %s", model_name)
        return
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise KeyError(f"Unexpected LoRA keys for {model_name}: {incompatible.unexpected_keys}")
    loaded = len(state_dict)
    LOGGER.info("Loaded %s LoRA tensors for %s", loaded, model_name)


def export_pretrained_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Export a standard checkpoint state_dict, merging LoRA weights when present."""
    exported: dict[str, torch.Tensor] = {}
    lora_modules = dict(_iter_lora_modules(model))
    skip_keys: set[str] = set()

    for module_name, _module in lora_modules.items():
        skip_keys.update(
            {
                f"{module_name}.base.weight",
                f"{module_name}.base.bias",
                f"{module_name}.lora_A.weight",
                f"{module_name}.lora_B.weight",
            }
        )

    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        if local_key in skip_keys:
            continue
        exported[local_key] = value.detach().cpu()

    for module_name, module in lora_modules.items():
        base_weight_key = f"{prefix}{module_name}.base.weight"
        lora_a_key = f"{prefix}{module_name}.lora_A.weight"
        lora_b_key = f"{prefix}{module_name}.lora_B.weight"
        base_weight = state_dict[base_weight_key]
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        merged_weight = (
            base_weight.float() + (lora_b.float() @ lora_a.float()) * module.scaling
        ).to(dtype=base_weight.dtype)
        exported[f"{module_name}.weight"] = merged_weight.detach().cpu()

        base_bias_key = f"{prefix}{module_name}.base.bias"
        if base_bias_key in state_dict:
            exported[f"{module_name}.bias"] = state_dict[base_bias_key].detach().cpu()

    return exported


def get_model_subfolder(model_type: str) -> str:
    """Map a logical model type to its checkpoint subfolder."""
    if model_type not in MODEL_TYPE_TO_SUBFOLDER:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return MODEL_TYPE_TO_SUBFOLDER[model_type]


def save_accelerate_checkpoint(
    *,
    accelerator,
    model,
    args,
    tag: str,
    extra_training_state: dict | None = None,
    training_state_filename: str = "resume_state.pt",
) -> Path:
    """Save one or more Stage-1 branches in WanModel.from_pretrained layout."""
    save_dir = Path(args.output_dir) / tag
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if isinstance(model, dict):
        model_items = list(model.items())
    else:
        subfolder = getattr(args, "subfolder", "")
        if not subfolder:
            raise ValueError("subfolder must be set when saving a single model")
        model_items = [(subfolder, model)]

    for subfolder, branch_model in model_items:
        model_dir = save_dir / subfolder
        if accelerator.is_main_process:
            model_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(branch_model)
        if hasattr(unwrapped, "save_16bit_model"):
            unwrapped.save_16bit_model(model_dir, "diffusion_pytorch_model.bin")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped.save_config(model_dir)
        elif accelerator.is_main_process:
            unwrapped.save_pretrained(model_dir, safe_serialization=False)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if extra_training_state:
            training_only = save_dir / "training_only"
            training_only.mkdir(parents=True, exist_ok=True)
            torch.save(extra_training_state, training_only / training_state_filename)
    accelerator.wait_for_everyone()
    return save_dir


def prune_checkpoint_dir(path: str | Path) -> None:
    """Delete one checkpoint directory if it exists."""
    root = Path(path)
    if root.exists():
        shutil.rmtree(root)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move optimizer state tensors between CPU and GPU."""
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


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
    steps_per_epoch = max(math.ceil(iters_per_epoch / grad_accum), 1)
    return max(num_epochs * steps_per_epoch, 1)
