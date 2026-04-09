"""Consistent project logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging once."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        resolved = Path(log_path)
        ensure_dir(resolved.parent)
        handlers.append(logging.FileHandler(resolved, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
