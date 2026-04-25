#!/usr/bin/env python3
"""Merge multiple hard-top Phy-CSGO datasets into one weighted training dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    repeat: int


def parse_dataset_specs(text: str) -> list[DatasetSpec]:
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            name, path, repeat = item.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                "Dataset specs must look like name:/abs/path:repeat"
            ) from exc
        specs.append(DatasetSpec(name=name, path=Path(path), repeat=int(repeat)))
    if not specs:
        raise ValueError("No dataset specs provided")
    return specs


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


def augment_row(row: dict[str, str], *, split: str, dataset_name: str, clip_name: str, repeat: int) -> dict[str, str]:
    new_row = dict(row)
    new_clip_path = f"{split}/clips/{dataset_name}__{clip_name}"
    new_row["clip_path"] = new_clip_path
    if "video" in new_row:
        new_row["video"] = os.path.join(new_clip_path, "video.mp4")
    new_row["source_dataset"] = dataset_name
    new_row["sampling_repeat"] = str(repeat)
    return new_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a weighted merged Phy-CSGO dataset from hard-top subsets.")
    parser.add_argument(
        "--dataset_specs",
        type=str,
        required=True,
        help="Comma-separated name:/abs/path:repeat specs, e.g. top1:/data/Phy_CSGO_hard_top1:4,top2:/data/Phy_CSGO_hard_top2:2",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=("symlink", "copy"), default="symlink")
    parser.add_argument(
        "--val_strategy",
        type=str,
        choices=("first", "union"),
        default="first",
        help="How to build metadata_val.csv. 'first' keeps only the first dataset's val split once.",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="",
        help="Optional dataset name to use for val when val_strategy=first. Default: first spec.",
    )
    args = parser.parse_args()

    specs = parse_dataset_specs(args.dataset_specs)
    output_dir = Path(args.output_dir)
    (output_dir / "train" / "clips").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "clips").mkdir(parents=True, exist_ok=True)

    chosen_val_dataset = args.val_dataset or specs[0].name
    merged_meta = {"train": [], "val": []}
    merged_audit = {"train": [], "val": []}
    meta_fieldnames = {"train": [], "val": []}
    audit_fieldnames = {"train": [], "val": []}
    manifest = {"datasets": []}

    for spec in specs:
        manifest["datasets"].append({
            "name": spec.name,
            "path": str(spec.path),
            "repeat": spec.repeat,
        })
        for split in ("train", "val"):
            meta_fields, meta_rows = read_csv_rows(spec.path / f"metadata_{split}.csv")
            audit_fields, audit_rows = read_csv_rows(spec.path / f"metadata_{split}_motion_audit.csv")
            meta_by_clip = {row["clip_path"]: row for row in meta_rows}
            audit_by_clip = {row["clip_path"]: row for row in audit_rows}

            if not meta_fieldnames[split]:
                meta_fieldnames[split] = list(meta_fields)
                if "source_dataset" not in meta_fieldnames[split]:
                    meta_fieldnames[split].extend(["source_dataset", "sampling_repeat"])
            if not audit_fieldnames[split]:
                audit_fieldnames[split] = list(audit_fields)
                if "source_dataset" not in audit_fieldnames[split]:
                    audit_fieldnames[split].extend(["source_dataset", "sampling_repeat"])

            row_repeat = spec.repeat if split == "train" else 1
            if split == "val" and args.val_strategy == "first" and spec.name != chosen_val_dataset:
                continue

            for clip_path in sorted(meta_by_clip):
                clip_name = Path(clip_path).name
                src_clip_dir = spec.path / clip_path
                dst_clip_dir = output_dir / split / "clips" / f"{spec.name}__{clip_name}"
                materialize_clip(src_clip_dir, dst_clip_dir, args.mode)

                meta_row = augment_row(meta_by_clip[clip_path], split=split, dataset_name=spec.name, clip_name=clip_name, repeat=row_repeat)
                audit_row = augment_row(audit_by_clip.get(clip_path, {"clip_path": clip_path}), split=split, dataset_name=spec.name, clip_name=clip_name, repeat=row_repeat)

                repeats_to_add = row_repeat
                if split == "val":
                    repeats_to_add = 1
                for _ in range(repeats_to_add):
                    merged_meta[split].append(dict(meta_row))
                    if audit_fields:
                        merged_audit[split].append(dict(audit_row))

    write_csv_rows(output_dir / "metadata_train.csv", meta_fieldnames["train"], merged_meta["train"])
    write_csv_rows(output_dir / "metadata_val.csv", meta_fieldnames["val"], merged_meta["val"])
    if audit_fieldnames["train"]:
        write_csv_rows(output_dir / "metadata_train_motion_audit.csv", audit_fieldnames["train"], merged_audit["train"])
    if audit_fieldnames["val"]:
        write_csv_rows(output_dir / "metadata_val_motion_audit.csv", audit_fieldnames["val"], merged_audit["val"])

    manifest["output_dir"] = str(output_dir)
    manifest["val_strategy"] = args.val_strategy
    manifest["val_dataset"] = chosen_val_dataset
    manifest["train_rows"] = len(merged_meta["train"])
    manifest["val_rows"] = len(merged_meta["val"])
    with open(output_dir / "weighted_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"train rows: {len(merged_meta['train'])}")
    print(f"val rows:   {len(merged_meta['val'])}")
    print(f"output:     {output_dir}")


if __name__ == "__main__":
    main()
