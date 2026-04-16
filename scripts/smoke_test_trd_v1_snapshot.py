#!/usr/bin/env python3
"""Smoke test the TRD snapshot-only validation path used on H20."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.io import read_yaml
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.seed import set_seed
from physical_consistency.lineage.contract import verify_stage1_checkpoint
from physical_consistency.trainers.trd_v1 import (
    TRDTrainingRunner,
    _require_existing_path,
    _resolve_teacher_checkpoint,
    build_args,
    parse_args,
)


def main() -> None:
    cli_args = parse_args()
    args = build_args(cli_args)
    args.experiment_name = f"{args.experiment_name}_smoke_snapshot"
    args.output_dir = str(Path(args.output_root) / "checkpoints" / args.experiment_name)

    if args.validation_runtime_mode != "snapshot_only":
        raise SystemExit(
            "smoke_test_trd_v1_snapshot.py expects validation_runtime_mode=snapshot_only; "
            f"got {args.validation_runtime_mode!r}"
        )

    _require_existing_path("base_model_dir", args.base_model_dir)
    _require_existing_path("stage1_ckpt_dir", args.stage1_ckpt_dir)
    _require_existing_path("dataset_dir", args.dataset_dir)
    _require_existing_path("lingbot_code_dir", args.lingbot_code_dir)
    _require_existing_path("teacher_repo_dir", args.teacher_repo_dir)
    if args.teacher_checkpoint_path:
        _require_existing_path("teacher_checkpoint_path", args.teacher_checkpoint_path)
    else:
        _require_existing_path("teacher_checkpoint_dir", args.teacher_checkpoint_dir)

    configure_logging(Path(args.output_root) / "logs" / f"smoke_snapshot_{args.experiment_name}_{args.model_type}.log")
    set_seed(args.seed)

    runner = TRDTrainingRunner(args)
    ok, errors = verify_stage1_checkpoint(args.stage1_ckpt_dir)
    if not ok:
        raise SystemExit("\n".join(errors))

    runner.teacher_checkpoint_path = _resolve_teacher_checkpoint(args.teacher_checkpoint_dir, args.teacher_checkpoint_path)
    runner._build_train_and_val_loaders()
    runner._log_train_plan()
    runner._initialize_training_runtime(checkpoint_dir=args.stage1_ckpt_dir, resume_state=None)
    runner.run_validation_cycle("smoke_snapshot")
    runner.accelerator.wait_for_everyone()

    if runner.accelerator.is_main_process:
        checkpoint_path = Path(args.output_dir) / "smoke_snapshot"
        eval_config_path = checkpoint_path / "validation_export" / "eval_trd_snapshot.yaml"
        request_path = checkpoint_path / "validation_export" / "validation_request.json"
        eval_config = read_yaml(eval_config_path)
        sample_steps = int(eval_config["sample_steps"])
        if sample_steps != int(args.validation_sample_steps):
            raise RuntimeError(
                f"snapshot export sample_steps={sample_steps}, expected {args.validation_sample_steps}"
            )
        if not request_path.exists():
            raise RuntimeError(f"missing validation request: {request_path}")
        print(
            "[SMOKE OK] snapshot validation export created "
            f"sample_steps={sample_steps} checkpoint={checkpoint_path}"
        )

    runner.accelerator.end_training()


if __name__ == "__main__":
    main()
