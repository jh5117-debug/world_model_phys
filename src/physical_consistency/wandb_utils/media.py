"""Plotting and media conversion helpers for W&B."""

from __future__ import annotations

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
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return wandb.Image(image, caption=title)


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
