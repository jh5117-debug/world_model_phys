#!/usr/bin/env python3
"""Fast smoke test for the TRD save/release/restore path."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.seed import set_seed
from physical_consistency.lineage.contract import verify_stage1_checkpoint
from physical_consistency.trainers.trd_v1 import (
    TRDTrainingRunner,
    _require_existing_path,
    _resolve_teacher_checkpoint,
    build_args,
    parse_args,
    save_dual_bundle_checkpoint,
)


def main() -> None:
    cli_args = parse_args()
    args = build_args(cli_args)
    args.experiment_name = f"{args.experiment_name}_smoke_restore"
    args.output_dir = str(Path(args.output_root) / "checkpoints" / args.experiment_name)

    _require_existing_path("base_model_dir", args.base_model_dir)
    _require_existing_path("stage1_ckpt_dir", args.stage1_ckpt_dir)
    _require_existing_path("dataset_dir", args.dataset_dir)
    _require_existing_path("lingbot_code_dir", args.lingbot_code_dir)
    _require_existing_path("teacher_repo_dir", args.teacher_repo_dir)
    if args.teacher_checkpoint_path:
        _require_existing_path("teacher_checkpoint_path", args.teacher_checkpoint_path)
    else:
        _require_existing_path("teacher_checkpoint_dir", args.teacher_checkpoint_dir)

    configure_logging(Path(args.output_root) / "logs" / f"smoke_restore_{args.experiment_name}_{args.model_type}.log")
    set_seed(args.seed)

    runner = TRDTrainingRunner(args)
    ok, errors = verify_stage1_checkpoint(args.stage1_ckpt_dir)
    if not ok:
        raise SystemExit("\n".join(errors))

    runner.teacher_checkpoint_path = _resolve_teacher_checkpoint(args.teacher_checkpoint_dir, args.teacher_checkpoint_path)
    runner._build_train_and_val_loaders()
    runner._log_train_plan()
    runner._initialize_training_runtime(checkpoint_dir=args.stage1_ckpt_dir, resume_state=None)

    candidate_path = save_dual_bundle_checkpoint(
        accelerator=runner.accelerator,
        model_bundle=runner.model_bundle,
        args=runner.args,
        tag="smoke_restore_candidate",
        extra_training_state=runner._training_state_payload_factory(
            tag="smoke_restore",
            include_optimizer=True,
            include_scheduler=True,
        ),
        training_state_filename="resume_state.pt",
    )
    runner._release_training_runtime()
    runner._restore_training_runtime(candidate_path)
    runner._release_training_runtime()
    runner.accelerator.end_training()
    if runner.accelerator.is_main_process:
        print(f"[SMOKE OK] restored runtime successfully from {candidate_path}")


if __name__ == "__main__":
    main()
