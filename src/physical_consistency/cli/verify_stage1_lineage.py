"""CLI to verify Stage-1 parent checkpoint contract."""

from __future__ import annotations

import argparse
import sys

from physical_consistency.common.defaults import CONFIG_DIR
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.lineage.contract import verify_stage1_checkpoint


def main() -> None:
    """Verify the default or provided Stage-1 checkpoint path."""
    parser = argparse.ArgumentParser(description="Verify Stage-1 lineage contract.")
    parser.add_argument(
        "--env_file",
        type=str,
        default=str(CONFIG_DIR / "path_config_cluster.env"),
    )
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    args = parser.parse_args()

    configure_logging()
    cfg = resolve_path_config(args, env_file=args.env_file)
    ok, errors = verify_stage1_checkpoint(cfg.stage1_ckpt_dir)
    if ok:
        print(f"[OK] Stage-1 checkpoint verified: {cfg.stage1_ckpt_dir}")
        return
    for error in errors:
        print(f"[ERROR] {error}")
    sys.exit(1)


if __name__ == "__main__":
    main()
