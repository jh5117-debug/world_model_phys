#!/usr/bin/env python3
"""Split one multi-rank Phy-CSGO dataset into rank-specific hard_top datasets."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def materialize_clip(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def parse_ranks(rank_text: str) -> list[int]:
    ranks = []
    for item in rank_text.split(","):
        item = item.strip()
        if not item:
            continue
        rank = int(item)
        if rank < 1:
            raise ValueError("Ranks are 1-based: use 1,2,3...")
        ranks.append(rank - 1)
    return sorted(set(ranks))


def split_dataset(dataset_dir: Path, output_root: Path, target_ranks: list[int], prefix: str, mode: str) -> None:
    metadata_fields = {}
    metadata_rows = {}
    audit_fields = {}
    audit_rows = {}

    for split in ("train", "val"):
        metadata_fields[split], metadata_rows[split] = read_csv_rows(dataset_dir / f"metadata_{split}.csv")
        audit_fields[split], audit_rows[split] = read_csv_rows(dataset_dir / f"metadata_{split}_motion_audit.csv")

    available_ranks = sorted(
        {
            int(row["selection_rank"])
            for split in ("train", "val")
            for row in audit_rows[split]
            if str(row.get("selection_rank", "")).strip() != ""
        }
    )
    if not available_ranks:
        raise SystemExit("No selection_rank values found in audit CSVs")

    if not target_ranks:
        target_ranks = available_ranks

    for rank in target_ranks:
        rank_name = f"{prefix}{rank + 1}"
        target_dir = output_root / rank_name
        print(f"[build] {rank_name} from selection_rank={rank}")

        for split in ("train", "val"):
            meta_by_clip = {row["clip_path"]: row for row in metadata_rows[split]}
            selected_audit_rows = [row for row in audit_rows[split] if int(row["selection_rank"]) == rank]
            selected_clip_paths = {row["clip_path"] for row in selected_audit_rows}
            selected_meta_rows = []
            rewritten_audit_rows = []

            for clip_path in sorted(selected_clip_paths):
                meta = meta_by_clip.get(clip_path)
                if meta is None:
                    continue
                src_clip_dir = dataset_dir / clip_path
                clip_name = Path(clip_path).name
                new_clip_path = f"{split}/clips/{clip_name}"
                dst_clip_dir = target_dir / new_clip_path
                materialize_clip(src_clip_dir, dst_clip_dir, mode)

                new_meta = dict(meta)
                new_meta["clip_path"] = new_clip_path
                if "video" in new_meta:
                    new_meta["video"] = os.path.join(new_clip_path, "video.mp4")
                selected_meta_rows.append(new_meta)

            for audit in selected_audit_rows:
                clip_name = Path(audit["clip_path"]).name
                new_audit = dict(audit)
                new_audit["clip_path"] = f"{split}/clips/{clip_name}"
                rewritten_audit_rows.append(new_audit)

            write_csv_rows(target_dir / f"metadata_{split}.csv", metadata_fields[split], selected_meta_rows)
            write_csv_rows(
                target_dir / f"metadata_{split}_motion_audit.csv",
                audit_fields[split],
                rewritten_audit_rows,
            )
            print(f"  {split}: {len(selected_meta_rows)} clips")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split ranked Phy-CSGO pool into hard_top{rank} datasets.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Input dataset generated with top_k_per_stream > 1")
    parser.add_argument("--output_root", type=str, required=True, help="Parent directory for rank-specific outputs")
    parser.add_argument("--ranks", type=str, default="", help="1-based ranks to export, e.g. 2,3,4. Default: all")
    parser.add_argument("--prefix", type=str, default="Phy_CSGO_hard_top")
    parser.add_argument("--mode", type=str, choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    target_ranks = parse_ranks(args.ranks) if args.ranks else []
    split_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_root=Path(args.output_root),
        target_ranks=target_ranks,
        prefix=args.prefix,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
