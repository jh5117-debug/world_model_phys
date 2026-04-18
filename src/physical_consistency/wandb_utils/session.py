"""W&B lifecycle helpers with rank-safe early init."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from collections.abc import Mapping
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
    """Log a dict to W&B and optionally a local JSONL file if payload exists."""
    if not payload:
        return
    _write_local_metrics(step, payload, accelerator=accelerator)
    if accelerator is not None:
        try:
            accelerator.log(payload, step=step)
        except Exception:
            LOGGER.debug("Skipping accelerator.log because no tracker is active", exc_info=True)
        return
    try:
        import wandb
    except Exception:
        return
    if wandb.run is not None:
        wandb.log(payload, step=step)


def _write_local_metrics(step: int, payload: dict[str, Any], *, accelerator=None) -> None:
    path_text = os.environ.get("PC_LOCAL_METRICS_PATH", "").strip()
    if not path_text:
        return
    if accelerator is not None and not getattr(accelerator, "is_main_process", True):
        return
    path = Path(path_text)
    try:
        ensure_dir(path.parent)
        record = {
            "time": time.time(),
            "step": int(step),
            "payload": _to_jsonable(payload),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        LOGGER.debug("Skipping local metrics write to %s", path, exc_info=True)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        try:
            tensor = value.detach()
            if getattr(tensor, "ndim", None) == 0:
                return float(tensor.item())
            return {
                "type": type(value).__name__,
                "shape": list(getattr(tensor, "shape", ())),
                "dtype": str(getattr(tensor, "dtype", "")),
            }
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
