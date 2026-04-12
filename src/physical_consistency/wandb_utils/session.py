"""W&B lifecycle helpers with rank-safe early init."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from physical_consistency.common.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def init_wandb_run(
    *,
    accelerator,
    project: str,
    entity: str = "",
    run_name: str,
    config: dict[str, Any],
    wandb_dir: str,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str = "online",
):
    """Initialize one shared W&B run via Accelerate trackers."""
    entity, project = _normalize_wandb_target(entity=entity, project=project)
    ensure_dir(Path(wandb_dir))
    accelerator.init_trackers(
        project_name=project,
        config=config,
        init_kwargs={
            "wandb": {
                "name": run_name,
                "entity": entity or None,
                "dir": wandb_dir,
                "tags": tags or [],
                "group": group,
                "job_type": job_type,
                "mode": mode,
            }
        },
    )
    run = None
    if accelerator.is_main_process:
        try:
            import wandb

            run = wandb.run
        except Exception:
            run = None
    LOGGER.info("Initialized W&B run: %s", run_name)
    return run


def _normalize_wandb_target(*, entity: str, project: str) -> tuple[str, str]:
    normalized_entity = (entity or "").strip()
    normalized_project = (project or "").strip()
    if "/" in normalized_project and not normalized_entity:
        normalized_entity, normalized_project = normalized_project.split("/", 1)
    return normalized_entity, normalized_project


def log_dict(step: int, payload: dict[str, Any] | None, *, accelerator=None) -> None:
    """Log a dict to W&B if payload exists."""
    if not payload:
        return
    if accelerator is not None:
        accelerator.log(payload, step=step)
        return
    try:
        import wandb
    except Exception:
        return
    if wandb.run is not None:
        wandb.log(payload, step=step)
