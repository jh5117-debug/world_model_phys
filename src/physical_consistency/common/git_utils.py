"""Helpers for recording git provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_commit(root: str | Path) -> str:
    """Return the current git commit if available."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(root),
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"
