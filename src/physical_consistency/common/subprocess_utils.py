"""Safe subprocess wrappers with logging."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Mapping

from .io import ensure_dir

LOGGER = logging.getLogger(__name__)


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
) -> None:
    """Run a command, teeing stdout/stderr to a file if requested."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})

    stdout_target = None
    try:
        if log_path is not None:
            resolved = Path(log_path)
            ensure_dir(resolved.parent)
            stdout_target = resolved.open("a", encoding="utf-8")
        LOGGER.info("Running command: %s", " ".join(command))
        subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=True,
            stdout=stdout_target or None,
            stderr=subprocess.STDOUT if stdout_target else None,
            text=True,
        )
    finally:
        if stdout_target is not None:
            stdout_target.close()
