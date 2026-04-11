"""Fixed validation subset builders and dataset view materialization."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from physical_consistency.common.io import ensure_dir, read_csv_rows, write_csv_rows


@dataclass(slots=True)
class ManifestBuildResult:
    """Summary of a built manifest."""

    output_path: Path
    total_rows: int
    selected_rows: int
    source_csv: Path


def build_fixed_manifest(
    metadata_csv: str | Path,
    output_csv: str | Path,
    *,
    sample_count: int,
    seed: int,
) -> ManifestBuildResult:
    """Create a deterministic subset CSV from a metadata file."""
    rows = read_csv_rows(metadata_csv)
    if sample_count > len(rows):
        sample_count = len(rows)

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = [rows[idx] for idx in sorted(indices[:sample_count])]
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_rows(output_csv, selected, fieldnames)
    return ManifestBuildResult(
        output_path=Path(output_csv),
        total_rows=len(rows),
        selected_rows=len(selected),
        source_csv=Path(metadata_csv),
    )


def materialize_dataset_view(
    base_dataset_dir: str | Path,
    manifest_csv: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a small dataset view directory compatible with eval_batch.py."""
    base = Path(base_dataset_dir)
    out_dir = ensure_dir(output_dir)

    for split_name in ["train", "val", "test"]:
        source = base / split_name
        target = out_dir / split_name
        if not target.exists() and source.exists():
            target.symlink_to(source, target_is_directory=True)

    selected_rows = read_csv_rows(manifest_csv)
    if not selected_rows:
        raise ValueError(f"Manifest is empty: {manifest_csv}")

    source_train = base / "metadata_train.csv"
    write_csv_rows(
        out_dir / "metadata_val.csv",
        selected_rows,
        list(selected_rows[0].keys()),
    )
    if source_train.exists():
        # Keep train metadata for any downstream script that expects it.
        train_rows = read_csv_rows(source_train)
        if train_rows:
            write_csv_rows(
                out_dir / "metadata_train.csv",
                train_rows,
                list(train_rows[0].keys()),
            )
    return out_dir


def hash_manifest(path: str | Path) -> str:
    """Hash a manifest file for reproducibility."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16]
