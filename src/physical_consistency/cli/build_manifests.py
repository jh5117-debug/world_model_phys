"""CLI for building fixed validation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from physical_consistency.common.defaults import CONFIG_DIR, DATA_DIR
from physical_consistency.common.logging_utils import configure_logging
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.datasets.manifest_builder import build_fixed_manifest


def main() -> None:
    """Build the workflow's fixed `val50` and `val200` subsets."""
    parser = argparse.ArgumentParser(description="Build fixed validation manifests.")
    parser.add_argument(
        "--env_file",
        type=str,
        default=str(CONFIG_DIR / "path_config_cluster.env"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "manifests"),
    )
    args = parser.parse_args()

    configure_logging()
    path_cfg = resolve_path_config(args, env_file=args.env_file)
    metadata_csv = Path(path_cfg.dataset_dir) / "metadata_val.csv"
    output_dir = Path(args.output_dir)

    build_fixed_manifest(
        metadata_csv,
        output_dir / "csgo_phys_val50.csv",
        sample_count=50,
        seed=args.seed,
    )
    build_fixed_manifest(
        metadata_csv,
        output_dir / "csgo_phys_val200.csv",
        sample_count=200,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
