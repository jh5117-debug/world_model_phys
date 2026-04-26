"""Optional official VideoPhy-2 validation hooks for Stage-1."""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from physical_consistency.common.io import read_json
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.common.summary_tables import format_videophy2_summary

from .config import VideoPhy2EvalConfig

LOGGER = logging.getLogger(__name__)


def run_stage1_videophy2_eval(
    cfg: VideoPhy2EvalConfig,
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    experiment_name: str,
    epoch: int,
    branch: str,
) -> dict | None:
    """Run config-driven official VideoPhy-2 generation / scoring commands."""

    if not cfg.enabled:
        return None
    if not cfg.generation_command and not cfg.score_command and not cfg.summary_json:
        LOGGER.warning(
            "Stage-1 VideoPhy-2 eval is enabled but no official generation/score/summary "
            "commands were configured; skipping this epoch."
        )
        return None

    context = {
        "bundle_dir": str(Path(bundle_dir).resolve()),
        "checkpoint_dir": str(Path(bundle_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "experiment_name": experiment_name,
        "epoch": str(epoch),
        "branch": branch,
    }
    log_root = Path(output_dir) / "videophy2" / f"{branch}_epoch_{epoch}"
    log_root.mkdir(parents=True, exist_ok=True)

    if cfg.generation_command:
        generation_command = Template(cfg.generation_command).safe_substitute(context)
        LOGGER.info("Running Stage-1 official generation hook: %s", generation_command)
        run_command(
            ["bash", "-lc", generation_command],
            cwd=cfg.working_dir or None,
            env=cfg.env,
            log_path=log_root / "generation.log",
        )

    if cfg.score_command:
        score_command = Template(cfg.score_command).safe_substitute(context)
        LOGGER.info("Running Stage-1 official VideoPhy-2 score hook: %s", score_command)
        run_command(
            ["bash", "-lc", score_command],
            cwd=cfg.working_dir or None,
            env=cfg.env,
            log_path=log_root / "score.log",
        )

    if not cfg.summary_json:
        return None
    summary_path = Path(Template(cfg.summary_json).safe_substitute(context))
    if not summary_path.exists():
        message = (
            "Stage-1 VideoPhy-2 summary_json was configured but not produced: "
            f"{summary_path}"
        )
        if cfg.fail_fast:
            raise FileNotFoundError(message)
        LOGGER.warning(message)
        return None

    summary = read_json(summary_path)
    LOGGER.info(
        "\n%s",
        format_videophy2_summary(
            summary,
            title=f"Stage-1 VideoPhy-2 Summary ({branch}, epoch={epoch})",
        ),
    )
    return summary
