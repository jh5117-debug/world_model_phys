"""Official V-JEPA 2.1 teacher wrapper loaded from a downloaded source tree."""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .base import TeacherEncoder, TeacherFeatures

LOGGER = logging.getLogger(__name__)

_EMA_ENCODER_VARIANTS = {
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
}


class VJEPA21Teacher(TeacherEncoder):
    """Frozen V-JEPA 2.1 teacher loaded from the official Meta source tree."""

    def __init__(
        self,
        *,
        repo_dir: str,
        checkpoint_path: str,
        device: torch.device,
        model_dtype: str | torch.dtype = torch.bfloat16,
        offload_after_encode: bool = True,
        model_variant: str = "vjepa2_1_vit_base_384",
        image_size: int = 384,
        teacher_input_frames: int = 64,
        drop_first_frame: bool = False,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model_dtype = self._resolve_model_dtype(model_dtype, device)
        self.offload_after_encode = bool(offload_after_encode and device.type == "cuda")
        self.storage_device = torch.device("cpu") if self.offload_after_encode else self.device
        self.model_variant = model_variant
        self.image_size = int(image_size)
        self.teacher_input_frames = int(teacher_input_frames)
        self.drop_first_frame = bool(drop_first_frame)

        backbones = self._load_backbones_module()
        factory = getattr(backbones, model_variant, None)
        if factory is None:
            raise AttributeError(f"Unsupported V-JEPA 2.1 model variant: {model_variant}")

        self.model, _ = factory(pretrained=False, num_frames=self.teacher_input_frames)
        self._load_checkpoint()
        self.model.to(device=self.storage_device, dtype=self.model_dtype)
        self._model_device = self.storage_device
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.feature_dim = int(self.model.embed_dim)
        self.normalize = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info(
            "Loaded V-JEPA 2.1 teacher from %s (variant=%s, dtype=%s, offload_after_encode=%s, storage_device=%s)",
            self.checkpoint_path,
            self.model_variant,
            self.model_dtype,
            self.offload_after_encode,
            self.storage_device,
        )

    def _load_backbones_module(self):
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"V-JEPA 2.1 repo directory not found: {self.repo_dir}")
        repo_dir_str = str(self.repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)
        return importlib.import_module("src.hub.backbones")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {self.checkpoint_path}")
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_key = "ema_encoder" if self.model_variant in _EMA_ENCODER_VARIANTS else "target_encoder"
        if checkpoint_key not in state_dict:
            available = ", ".join(sorted(str(key) for key in state_dict.keys()))
            raise KeyError(
                f"Checkpoint key {checkpoint_key!r} missing in {self.checkpoint_path}. Available keys: {available}"
            )
        encoder_state_dict = self._clean_backbone_key(dict(state_dict[checkpoint_key]))
        incompatible = self.model.load_state_dict(encoder_state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Failed to load official V-JEPA 2.1 encoder weights cleanly: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    @staticmethod
    def _clean_backbone_key(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            key = key.replace("backbone.", "")
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _resolve_model_dtype(model_dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
        if isinstance(model_dtype, torch.dtype):
            resolved = model_dtype
        else:
            normalized = str(model_dtype).strip().lower()
            supported = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
            }
            if normalized not in supported:
                raise ValueError(f"Unsupported V-JEPA 2.1 teacher dtype: {model_dtype}")
            resolved = supported[normalized]
        if device.type != "cuda" and resolved != torch.float32:
            LOGGER.warning(
                "V-JEPA 2.1 teacher dtype %s requested on %s; falling back to float32.",
                resolved,
                device,
            )
            return torch.float32
        return resolved

    def _autocast_context(self):
        if self.device.type != "cuda" or self.model_dtype == torch.float32:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.model_dtype)

    def _move_model_to(self, target_device: torch.device) -> None:
        target_device = torch.device(target_device)
        if self._model_device == target_device:
            return
        self.model.to(device=target_device)
        self._model_device = target_device
        if self.device.type == "cuda" and target_device.type != "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> TeacherFeatures:
        """Encode a normalized video tensor [B, 3, F, H, W] in [-1, 1]."""
        if video.ndim != 5:
            raise ValueError(f"Expected [B,3,F,H,W], got {tuple(video.shape)}")
        self._move_model_to(self.device)
        video = video.to(self.device, non_blocking=True)
        video = self._preprocess(video)
        if self.model_dtype != torch.float32:
            video = video.to(self.model_dtype)
        with self._autocast_context():
            tokens = self.model(video)
        tokens = self._reshape_tokens(tokens)
        if self.offload_after_encode:
            self._move_model_to(torch.device("cpu"))
        return TeacherFeatures(tokens=tokens)

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        bsz, channels, frames, height, width = video.shape
        if channels != 3:
            raise ValueError(f"Expected RGB video with 3 channels, got {channels}")

        sampled = video
        if self.drop_first_frame and sampled.shape[2] > 1:
            sampled = sampled[:, :, 1:]
            frames = sampled.shape[2]

        if self.teacher_input_frames > 0 and frames != self.teacher_input_frames:
            frame_ids = torch.linspace(0, frames - 1, self.teacher_input_frames, device=video.device)
            frame_ids = frame_ids.round().long().clamp_(0, frames - 1)
            sampled = sampled.index_select(dim=2, index=frame_ids)

        sampled = (sampled + 1.0) / 2.0
        sampled = sampled.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        short_side = int(round(self.image_size * 256 / 224))
        resize_height, resize_width = self._resize_short_side(height, width, short_side)
        sampled = F.interpolate(
            sampled,
            size=(resize_height, resize_width),
            mode="bicubic",
            align_corners=False,
        )
        sampled = self._center_crop(sampled, self.image_size)
        sampled = self.normalize(sampled)
        sampled = sampled.view(bsz, self.teacher_input_frames, channels, self.image_size, self.image_size)
        return sampled.permute(0, 2, 1, 3, 4)

    @staticmethod
    def _resize_short_side(height: int, width: int, short_side: int) -> tuple[int, int]:
        if height <= width:
            new_height = short_side
            new_width = int(round(width * short_side / height))
        else:
            new_width = short_side
            new_height = int(round(height * short_side / width))
        return max(new_height, short_side), max(new_width, short_side)

    @staticmethod
    def _center_crop(video: torch.Tensor, crop_size: int) -> torch.Tensor:
        _, _, height, width = video.shape
        top = max((height - crop_size) // 2, 0)
        left = max((width - crop_size) // 2, 0)
        cropped = video[:, :, top : top + crop_size, left : left + crop_size]
        if cropped.shape[-2:] != (crop_size, crop_size):
            cropped = F.interpolate(
                cropped,
                size=(crop_size, crop_size),
                mode="bicubic",
                align_corners=False,
            )
        return cropped

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        bsz, seq_len, channels = tokens.shape
        patch_size = int(self.model.patch_size if isinstance(self.model.patch_size, int) else self.model.patch_size[0])
        spatial_h = self.image_size // patch_size
        spatial_w = self.image_size // patch_size
        spatial = spatial_h * spatial_w
        if spatial <= 0 or seq_len % spatial != 0:
            raise ValueError(
                "V-JEPA 2.1 teacher token grid mismatch: "
                f"seq_len={seq_len}, spatial_h={spatial_h}, spatial_w={spatial_w}"
            )
        temporal = seq_len // spatial
        return tokens.view(bsz, temporal, spatial, channels)
