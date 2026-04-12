"""VideoMAEv2 teacher wrapper using the cloned VideoREPA implementation."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .base import TeacherEncoder, TeacherFeatures

LOGGER = logging.getLogger(__name__)


class VideoMAEv2Teacher(TeacherEncoder):
    """Frozen VideoMAEv2 teacher loaded from the official VideoREPA repo."""

    def __init__(
        self,
        *,
        repo_dir: str,
        checkpoint_path: str,
        device: torch.device,
        model_variant: str = "vit_base_patch16_224",
        image_size: int = 224,
        align_video_resolution: tuple[int, int] = (160, 240),
        pretrained_num_frames: int = 16,
        teacher_input_frames: int = 49,
        drop_first_frame: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model_variant = model_variant
        self.image_size = image_size
        self.align_video_resolution = align_video_resolution
        self.pretrained_num_frames = pretrained_num_frames
        self.teacher_input_frames = teacher_input_frames
        self.drop_first_frame = drop_first_frame

        module = self._load_module()
        factory = getattr(module, model_variant)
        self.model = factory(
            img_size=image_size,
            align_video_resolution=align_video_resolution,
            all_frames=pretrained_num_frames,
        ).to(device)
        self.model.from_pretrained(str(self.checkpoint_path))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.feature_dim = int(self.model.embed_dim)
        self.normalize = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        LOGGER.info("Loaded VideoMAEv2 teacher from %s", self.checkpoint_path)

    def _load_module(self):
        ssl_path = (
            self.repo_dir
            / "finetune"
            / "models"
            / "cogvideox_t2v_align"
            / "models"
            / "ssl"
            / "VideoMAEv2.py"
        )
        if not ssl_path.exists():
            raise FileNotFoundError(f"VideoMAEv2.py not found under {ssl_path}")
        spec = importlib.util.spec_from_file_location("pc_videomaev2", ssl_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {ssl_path}")
        module = importlib.util.module_from_spec(spec)
        module_name = spec.name
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            raise
        return module

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> TeacherFeatures:
        """Encode a normalized video tensor [B, 3, F, H, W] in [-1, 1]."""
        if video.ndim != 5:
            raise ValueError(f"Expected [B,3,F,H,W], got {tuple(video.shape)}")
        video = video.to(self.device)
        video = self._preprocess(video)
        tokens = self.model(video)
        tokens = self._reshape_tokens(tokens)
        return TeacherFeatures(tokens=tokens)

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        bsz, _, frames, _, _ = video.shape
        sampled = video
        if self.teacher_input_frames > 0 and frames != self.teacher_input_frames:
            frame_ids = torch.linspace(0, frames - 1, self.teacher_input_frames, device=video.device)
            frame_ids = frame_ids.round().long().clamp_(0, frames - 1)
            sampled = video.index_select(dim=2, index=frame_ids)
        if self.drop_first_frame and sampled.shape[2] > 1:
            sampled = sampled[:, :, 1:]
        sampled = (sampled + 1.0) / 2.0
        sampled = sampled.transpose(1, 2).flatten(0, 1)
        sampled = F.interpolate(
            sampled,
            size=self.align_video_resolution,
            mode="bicubic",
            align_corners=False,
        )
        sampled = self.normalize(sampled)
        sampled = sampled.reshape(bsz, -1, 3, *self.align_video_resolution).transpose(1, 2)
        return sampled

    def _reshape_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        bsz, seq_len, channels = tokens.shape
        patch = int(self.model.patch_size if isinstance(self.model.patch_size, int) else self.model.patch_size[0])
        spatial_h = self.align_video_resolution[0] // patch
        spatial_w = self.align_video_resolution[1] // patch
        spatial = spatial_h * spatial_w
        if spatial <= 0 or seq_len % spatial != 0:
            raise ValueError(
                "Teacher token grid mismatch: "
                f"seq_len={seq_len}, spatial_h={spatial_h}, spatial_w={spatial_w}"
            )
        temporal = seq_len // spatial
        tokens = tokens.view(bsz, temporal, spatial_h * spatial_w, channels)
        return tokens
