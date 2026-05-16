"""Sequential low/high Stage-2 runner for Phy_CSGO action fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch.distributed as dist

from physical_consistency.common.defaults import CONFIG_DIR
from physical_consistency.common.io import write_json
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.trainers.trd_v1 import (
    build_args as trd_build_args,
    main as trd_main,
    parse_args as trd_parse_args,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(CONFIG_DIR / "train_stage2_phycsgo_act.yaml"))
    parser.add_argument("--env_file", type=str, default=str(CONFIG_DIR / "path_config_cluster.env"))
    return parser.parse_known_args()


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _resolve_trd_args(argv: list[str]):
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        return trd_build_args(trd_parse_args())
    finally:
        sys.argv = original_argv


def _run_trd(argv: list[str]):
    resolved = _resolve_trd_args(argv)
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        trd_main()
    finally:
        sys.argv = original_argv
    return resolved


def _select_phase_checkpoint(output_dir: Path) -> Path:
    """Prefer the validated best checkpoint, otherwise use the final checkpoint."""
    for name in ("best_physinone", "best_videophy2"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return output_dir / "final"


def main() -> None:
    known, passthrough = parse_args()
    path_cfg = resolve_path_config(argparse.Namespace(dataset_dir=""), env_file=known.env_file)
    existing_flags = set(item for item in passthrough if item.startswith("--"))
    common = ["--config", known.config, "--env_file", known.env_file, *passthrough]
    if "--dataset_dir" not in existing_flags:
        common.extend(["--dataset_dir", path_cfg.phycsgo_weighted_dir])
    if "--control_type" not in existing_flags:
        common.extend(["--control_type", "act"])
    if "--trd_backward_mode" not in existing_flags:
        common.extend(["--trd_backward_mode", "off"])

    probe_args = _resolve_trd_args([*common, "--model_type", "low"])
    configure_logging(
        Path(probe_args.output_root) / "logs" / f"train_{probe_args.experiment_name}_stage2_sequence.log"
    )

    base_experiment = probe_args.experiment_name
    stage_root = Path(probe_args.output_root) / "checkpoints" / base_experiment
    source_bundle_dir = str(Path(probe_args.stage1_ckpt_dir).resolve())

    low_args = _run_trd(
        [
            *common,
            "--model_type",
            "low",
            "--experiment_name",
            f"{base_experiment}_low",
            "--eval_companion_ckpt_dir",
            source_bundle_dir,
        ]
    )
    low_final_dir = Path(low_args.output_dir) / "final"
    low_selected_dir = _select_phase_checkpoint(Path(low_args.output_dir))

    high_args = _run_trd(
        [
            *common,
            "--model_type",
            "high",
            "--experiment_name",
            f"{base_experiment}_high",
            "--eval_companion_ckpt_dir",
            str(low_selected_dir),
        ]
    )
    high_final_dir = Path(high_args.output_dir) / "final"
    high_selected_dir = _select_phase_checkpoint(Path(high_args.output_dir))

    final_bundle = materialize_eval_checkpoint_bundle(
        ft_ckpt_dir=high_selected_dir,
        output_root=stage_root / "final_bundle_cache",
        experiment_name=f"{base_experiment}_stage2_final",
        companion_ckpt_dir=low_selected_dir,
    )
    final_bundle_link = stage_root / "stage2_final_bundle"
    if _is_main_process():
        stage_root.mkdir(parents=True, exist_ok=True)
        if final_bundle_link.is_symlink() or final_bundle_link.exists():
            final_bundle_link.unlink()
        os.symlink(final_bundle, final_bundle_link, target_is_directory=True)
        write_json(
            stage_root / "stage2_summary.json",
            {
                "experiment_name": base_experiment,
                "config": str(Path(known.config).resolve()),
                "env_file": str(Path(known.env_file).resolve()),
                "source_stage1_bundle_dir": source_bundle_dir,
                "dataset_dir": low_args.dataset_dir,
                "low_phase": {
                    "experiment_name": low_args.experiment_name,
                    "output_dir": low_args.output_dir,
                    "final_checkpoint_dir": str(low_final_dir),
                    "selected_checkpoint_dir": str(low_selected_dir),
                },
                "high_phase": {
                    "experiment_name": high_args.experiment_name,
                    "output_dir": high_args.output_dir,
                    "final_checkpoint_dir": str(high_final_dir),
                    "selected_checkpoint_dir": str(high_selected_dir),
                },
                "final_stage2_bundle_dir": str(final_bundle),
                "final_stage2_bundle_link": str(final_bundle_link),
            },
        )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
