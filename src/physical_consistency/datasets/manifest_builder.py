"""Fixed validation subset builders and dataset view materialization."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
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

    selected_rows = read_csv_rows(manifest_csv)
    if not selected_rows:
        raise ValueError(f"Manifest is empty: {manifest_csv}")

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_split[_extract_split_name(row)].append(row)

    for split_name in ["train", "val", "test"]:
        source = base / split_name
        target = out_dir / split_name
        if split_name in rows_by_split:
            _materialize_selected_split(
                base=base,
                source_split_dir=source,
                split_name=split_name,
                rows=rows_by_split[split_name],
                out_dir=out_dir,
            )
            continue
        if not target.exists() and source.exists():
            target.symlink_to(source, target_is_directory=True)

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


def _extract_split_name(row: dict[str, str]) -> str:
    clip_path = Path(str(row.get("clip_path", "")))
    if len(clip_path.parts) >= 3 and clip_path.parts[1] == "clips":
        return clip_path.parts[0]
    return "val"


def _clip_dir_has_media(clip_dir: Path) -> bool:
    return (clip_dir / "image.jpg").exists() or (clip_dir / "video.mp4").exists()


def _resolve_source_clip_dir(base: Path, row: dict[str, str]) -> Path | None:
    clip_path = str(row.get("clip_path", "")).strip()
    if clip_path:
        direct = base / clip_path
        if direct.exists() and _clip_dir_has_media(direct):
            return direct

    video_path = str(row.get("video", "")).strip()
    if video_path:
        video_candidate = Path(video_path)
        if not video_candidate.is_absolute():
            video_candidate = base / video_candidate
        if video_candidate.exists():
            parent = video_candidate.parent
            if parent.exists() and _clip_dir_has_media(parent):
                return parent

    clip_name = Path(clip_path).name
    if not clip_name:
        return None
    for split_name in ["train", "val", "test"]:
        candidate = base / split_name / "clips" / clip_name
        if candidate.exists() and _clip_dir_has_media(candidate):
            return candidate
    for candidate in sorted(base.glob(f"*/clips/{clip_name}")):
        if candidate.exists() and _clip_dir_has_media(candidate):
            return candidate
    return None


def _materialize_selected_split(
    *,
    base: Path,
    source_split_dir: Path,
    split_name: str,
    rows: list[dict[str, str]],
    out_dir: Path,
) -> None:
    target_split_dir = out_dir / split_name
    direct_rows_ok = bool(source_split_dir.exists())
    if direct_rows_ok:
        for row in rows:
            clip_path = str(row.get("clip_path", "")).strip()
            if not clip_path:
                direct_rows_ok = False
                break
            direct_candidate = base / clip_path
            if not (direct_candidate.exists() and _clip_dir_has_media(direct_candidate)):
                direct_rows_ok = False
                break
    if direct_rows_ok:
        if not target_split_dir.exists():
            target_split_dir.symlink_to(source_split_dir, target_is_directory=True)
        return

    if target_split_dir.exists():
        if target_split_dir.is_symlink() or target_split_dir.is_file():
            target_split_dir.unlink()
        else:
            shutil.rmtree(target_split_dir)
    ensure_dir(target_split_dir / "clips")

    for row in rows:
        clip_path = str(row.get("clip_path", "")).strip()
        if not clip_path:
            raise ValueError(f"Manifest row missing clip_path: {row}")
        target_clip_dir = out_dir / clip_path
        if target_clip_dir.exists():
            continue
        source_clip_dir = _resolve_source_clip_dir(base, row)
        if source_clip_dir is None:
            raise FileNotFoundError(
                f"Could not resolve source clip directory for manifest row clip_path={clip_path!r}"
            )
        ensure_dir(target_clip_dir.parent)
        target_clip_dir.symlink_to(source_clip_dir, target_is_directory=True)
