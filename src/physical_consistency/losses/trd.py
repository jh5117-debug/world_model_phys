"""Token Relation Distillation loss for Stage-1 physical-consistency tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TRDLossOutput:
    """Structured loss outputs for logging."""

    total: torch.Tensor
    spatial: torch.Tensor
    temporal: torch.Tensor
    spatial_student: torch.Tensor
    spatial_teacher: torch.Tensor
    temporal_student: torch.Tensor
    temporal_teacher: torch.Tensor


class TokenRelationDistillationLoss(nn.Module):
    """Compute VideoREPA-style relation alignment with adaptive pooling."""

    def __init__(
        self,
        *,
        relation_tokens: int = 64,
        margin: float = 0.1,
        lambda_spatial: float = 1.0,
        lambda_temporal: float = 1.0,
    ) -> None:
        super().__init__()
        self.relation_tokens = relation_tokens
        self.margin = margin
        self.lambda_spatial = lambda_spatial
        self.lambda_temporal = lambda_temporal

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> TRDLossOutput:
        """Compute spatial and temporal relation loss."""
        student = self._normalize(self._pool_tokens(self._match_time(student)))
        teacher = self._normalize(self._pool_tokens(self._match_time(teacher, target_frames=student.shape[1])))

        spatial_student = torch.einsum("bfnd,bfmd->bfnm", student, student)
        spatial_teacher = torch.einsum("bfnd,bfmd->bfnm", teacher, teacher)
        spatial = F.relu((spatial_student - spatial_teacher).abs() - self.margin).mean()

        temporal_student_feats = student.mean(dim=2)
        temporal_teacher_feats = teacher.mean(dim=2)
        temporal_student = torch.einsum("bfd,bgd->bfg", temporal_student_feats, temporal_student_feats)
        temporal_teacher = torch.einsum("bfd,bgd->bfg", temporal_teacher_feats, temporal_teacher_feats)
        diag_mask = 1.0 - torch.eye(temporal_student.shape[-1], device=temporal_student.device).unsqueeze(0)
        temporal = (
            F.relu((temporal_student - temporal_teacher).abs() - self.margin) * diag_mask
        ).sum() / diag_mask.sum().clamp_min(1.0)

        total = self.lambda_spatial * spatial + self.lambda_temporal * temporal
        return TRDLossOutput(
            total=total,
            spatial=spatial,
            temporal=temporal,
            spatial_student=spatial_student[0, 0].detach().cpu(),
            spatial_teacher=spatial_teacher[0, 0].detach().cpu(),
            temporal_student=temporal_student[0].detach().cpu(),
            temporal_teacher=temporal_teacher[0].detach().cpu(),
        )

    def _pool_tokens(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[2] == self.relation_tokens:
            return features
        bsz, frames, tokens, channels = features.shape
        flattened = features.reshape(bsz * frames, tokens, channels).transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(flattened, self.relation_tokens)
        return pooled.transpose(1, 2).reshape(bsz, frames, self.relation_tokens, channels)

    def _match_time(self, features: torch.Tensor, target_frames: int | None = None) -> torch.Tensor:
        target_frames = target_frames or features.shape[1]
        if features.shape[1] == target_frames:
            return features
        bsz, frames, tokens, channels = features.shape
        flattened = features.permute(0, 2, 3, 1).reshape(bsz * tokens, channels, frames)
        resized = F.interpolate(flattened, size=target_frames, mode="linear", align_corners=False)
        return resized.reshape(bsz, tokens, channels, target_frames).permute(0, 3, 1, 2)

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, dim=-1)
