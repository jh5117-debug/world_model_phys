"""Sequential low/high runner for pure Stage-1 PhysInOne camera training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch.distributed as dist

from physical_consistency.common.defaults import CONFIG_DIR
from physical_consistency.common.io import write_json
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle

from .config import Stage1PhysInOneConfig
from .eval import run_stage1_videophy2_eval
from .trainer import Stage1BranchTrainer


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure Stage-1 PhysInOne camera LoRA.")
    parser.add_argument("--config", type=str, default=str(CONFIG_DIR / "train_stage1_physinone_cam.yaml"))
    parser.add_argument("--env_file", type=str, default=str(CONFIG_DIR / "path_config_cluster.env"))
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--lingbot_code_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--wandb_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Stage1PhysInOneConfig.from_yaml(
        args.config,
        env_file=args.env_file,
        cli_args=args,
    )
    configure_logging(Path(cfg.output_root) / "logs" / f"train_{cfg.experiment_name}_stage1.log")

    low_result = Stage1BranchTrainer(
        cfg,
        branch="low",
        source_checkpoint_dir=cfg.base_model_dir,
        companion_checkpoint_dir=cfg.base_model_dir,
    ).run()

    high_result = Stage1BranchTrainer(
        cfg,
        branch="high",
        source_checkpoint_dir=cfg.base_model_dir,
        companion_checkpoint_dir=low_result.final_branch_dir,
    ).run()

    final_bundle = materialize_eval_checkpoint_bundle(
        ft_ckpt_dir=high_result.final_branch_dir,
        output_root=Path(cfg.output_dir) / "final_bundle_cache",
        experiment_name=f"{cfg.experiment_name}_stage1_final",
        companion_ckpt_dir=low_result.final_branch_dir,
    )
    final_bundle_link = Path(cfg.output_dir) / "stage1_final_bundle"
    if _is_main_process():
        if final_bundle_link.is_symlink() or final_bundle_link.exists():
            final_bundle_link.unlink()
        os.symlink(final_bundle, final_bundle_link, target_is_directory=True)
    if cfg.videophy2_eval.enabled:
        run_stage1_videophy2_eval(
            cfg.videophy2_eval,
            bundle_dir=final_bundle,
            output_dir=cfg.output_dir,
            experiment_name=cfg.experiment_name,
            epoch=cfg.num_epochs,
            branch="stage1_final",
        )

    if _is_main_process():
        write_json(
            Path(cfg.output_dir) / "stage1_summary.json",
            {
                "experiment_name": cfg.experiment_name,
                "config_path": cfg.config_path,
                "config_hash": cfg.config_hash,
                "base_model_dir": cfg.base_model_dir,
                "dataset_dir": cfg.dataset_dir,
                "low_phase": {
                    "final_branch_dir": low_result.final_branch_dir,
                    "final_eval_bundle_dir": low_result.final_eval_bundle_dir,
                },
                "high_phase": {
                    "final_branch_dir": high_result.final_branch_dir,
                    "final_eval_bundle_dir": high_result.final_eval_bundle_dir,
                },
                "final_stage1_bundle_dir": str(final_bundle),
                "final_stage1_bundle_link": str(final_bundle_link),
            },
        )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
