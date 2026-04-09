"""Plotting and media conversion helpers for W&B."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def relation_matrix_image(matrix: np.ndarray, title: str = "relation"):
    """Create a W&B image from a relation matrix if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import wandb
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return wandb.Image(buffer, caption=title)


def safe_video(path: str | Path, caption: str = ""):
    """Wrap a video path as a W&B video if the file exists."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        import wandb
    except Exception:
        return None
    return wandb.Video(str(file_path), caption=caption, fps=8)
