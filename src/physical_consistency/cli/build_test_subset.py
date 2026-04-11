"""CLI for creating a smaller CSGO test split from metadata_val.csv."""

from __future__ import annotations

import argparse

from physical_consistency.datasets.test_subset import build_csgo_test_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata_test.csv and test/clips from metadata_val.csv.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_split", type=str, default="val")
    parser.add_argument("--target_split", type=str, default="test")
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_csgo_test_subset(
        args.dataset_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        source_split=args.source_split,
        target_split=args.target_split,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )
    print(f"Built {result.target_split} split: {result.selected_rows}/{result.total_rows}")
    print(f"Metadata: {result.output_metadata_path}")
    print(f"Clips: {result.output_clips_dir}")


if __name__ == "__main__":
    main()
