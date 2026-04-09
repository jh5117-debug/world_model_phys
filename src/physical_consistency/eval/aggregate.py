"""Aggregate experiment-level summaries into a single report."""

from __future__ import annotations

import argparse
from pathlib import Path

from physical_consistency.common.io import read_json, write_json


def compare_experiments(eval_root: str | Path) -> dict:
    """Collect `summary.json` files under one eval root."""
    root = Path(eval_root)
    output: dict[str, dict] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        rel = summary_path.relative_to(root)
        output[str(rel.parent)] = read_json(summary_path)
    return output


def main() -> None:
    """CLI main."""
    parser = argparse.ArgumentParser(description="Aggregate experiment summaries.")
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    payload = compare_experiments(args.eval_root)
    write_json(args.output_json, payload)


if __name__ == "__main__":
    main()
