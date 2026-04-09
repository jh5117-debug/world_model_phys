"""Teacher encoder interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TeacherFeatures:
    """Canonical teacher feature container."""

    tokens: torch.Tensor  # [B, F, N, C]


class TeacherEncoder:
    """Abstract teacher encoder."""

    feature_dim: int

    def encode(self, video: torch.Tensor) -> TeacherFeatures:
        """Encode video frames into teacher tokens."""
        raise NotImplementedError
