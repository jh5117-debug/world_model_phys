#!/usr/bin/env python3
"""Attach a small validation split to an existing dataset without touching the source dataset."""

from __future__ import annotations

import argparse
import csv
import os
import random
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a val split to a target dataset from a source dataset.")
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=("symlink", "copy"), default="symlink")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=("random", "top_combat"),
        default="random",
        help="random is more representative; top_combat makes val deliberately harder.",
    )
    args = parser.parse_args()

    source_dataset = Path(args.source_dataset)
    target_dataset = Path(args.target_dataset)
    target_val_dir = target_dataset / "val" / "clips"
    target_val_dir.mkdir(parents=True, exist_ok=True)

    meta_fields, meta_rows = read_csv_rows(source_dataset / "metadata_train.csv")
    audit_fields, audit_rows = read_csv_rows(source_dataset / "metadata_train_motion_audit.csv")
    if not meta_rows:
        raise SystemExit(f"No metadata_train.csv rows found in {source_dataset}")

    audit_by_clip = {row["clip_path"]: row for row in audit_rows}
    available = [row for row in meta_rows if (source_dataset / row["clip_path"]).exists()]
    count = min(args.count, len(available))

    if args.strategy == "top_combat" and audit_rows:
        ranked = sorted(
            available,
            key=lambda row: float(audit_by_clip.get(row["clip_path"], {}).get("combat_score", 0.0)),
            reverse=True,
        )
        selected = ranked[:count]
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(available, count)

    new_meta_rows = []
    new_audit_rows = []
    for row in selected:
        clip_path = row["clip_path"]
        clip_name = Path(clip_path).name
        src_clip_dir = source_dataset / clip_path
        dst_clip_dir = target_val_dir / clip_name
        materialize_clip(src_clip_dir, dst_clip_dir, args.mode)

        new_row = dict(row)
        new_row["clip_path"] = f"val/clips/{clip_name}"
        if "video" in new_row:
            new_row["video"] = os.path.join("val", "clips", clip_name, "video.mp4")
        new_meta_rows.append(new_row)

        audit_row = audit_by_clip.get(clip_path)
        if audit_row is not None:
            new_audit = dict(audit_row)
            new_audit["clip_path"] = f"val/clips/{clip_name}"
            new_audit_rows.append(new_audit)

    write_csv_rows(target_dataset / "metadata_val.csv", meta_fields, new_meta_rows)
    if audit_fields:
        write_csv_rows(target_dataset / "metadata_val_motion_audit.csv", audit_fields, new_audit_rows)

    print(f"attached val clips: {len(new_meta_rows)}")
    print(f"target dataset:     {target_dataset}")


if __name__ == "__main__":
    main()
