"""Helpers for preparing official VideoPhy-2 benchmark assets."""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import urlopen

from physical_consistency.common.io import read_csv_rows, write_csv_rows


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _is_truthy(value: str | None) -> bool:
    return _normalize_text(value).lower() in {"1", "true", "yes", "y"}


@dataclass(slots=True)
class PreparedPromptManifests:
    """Prompt manifest paths and counts."""

    original_manifest: Path
    original_hard_manifest: Path
    upsampled_manifest: Path
    upsampled_hard_manifest: Path
    original_count: int
    original_hard_count: int
    upsampled_count: int
    upsampled_hard_count: int


@dataclass(slots=True)
class PreparedOfficialVideoSubset:
    """Downloaded official VideoPhy-2 video subset and manifest."""

    manifest_path: Path
    video_root: Path
    count: int
    prompt_mode: str
    hard_only: bool
    model_names: list[str]


def build_prompt_manifest_rows(
    rows: list[dict[str, str]],
    *,
    prompt_mode: str,
) -> list[dict[str, str]]:
    """Deduplicate the official VideoPhy-2 table into one-row-per-prompt manifests."""
    if prompt_mode not in {"original", "upsampled"}:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    by_prompt: dict[str, dict[str, str]] = {}
    ordered_keys: list[str] = []

    for row in rows:
        caption = _normalize_text(row.get("caption"))
        upsampled_caption = _normalize_text(row.get("upsampled_caption"))
        prompt = caption
        if prompt_mode == "upsampled":
            prompt = upsampled_caption or caption
        if not prompt:
            continue

        existing = by_prompt.get(prompt)
        if existing is None:
            existing = {
                "sample_id": "",
                "prompt": prompt,
                "caption": caption,
                "upsampled_caption": upsampled_caption,
                "action": _normalize_text(row.get("action")),
                "category": _normalize_text(row.get("category")),
                "is_hard": "1" if _is_truthy(row.get("is_hard")) else "0",
                "source_mode": prompt_mode,
            }
            by_prompt[prompt] = existing
            ordered_keys.append(prompt)
            continue

        if existing["is_hard"] != "1" and _is_truthy(row.get("is_hard")):
            existing["is_hard"] = "1"
        for key, value in (
            ("caption", caption),
            ("upsampled_caption", upsampled_caption),
            ("action", _normalize_text(row.get("action"))),
            ("category", _normalize_text(row.get("category"))),
        ):
            if not existing[key] and value:
                existing[key] = value

    output_rows: list[dict[str, str]] = []
    for idx, key in enumerate(ordered_keys):
        row = dict(by_prompt[key])
        row["sample_id"] = f"videophy2_{prompt_mode}_{idx:04d}"
        output_rows.append(row)
    return output_rows


def _slugify(value: str | None) -> str:
    text = _normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _prompt_from_row(row: dict[str, str], *, prompt_mode: str) -> str:
    if prompt_mode not in {"original", "upsampled"}:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    caption = _normalize_text(row.get("caption"))
    upsampled_caption = _normalize_text(row.get("upsampled_caption"))
    if prompt_mode == "upsampled":
        return upsampled_caption or caption
    return caption


def _relative_video_path(row: dict[str, str], *, ordinal: int) -> str:
    model_name = _slugify(row.get("model_name"))
    video_url = _normalize_text(row.get("video_url"))
    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    stem = _slugify(Path(parsed.path).stem)[:80]
    return f"{model_name}/{ordinal:04d}_{stem}{suffix}"


def build_official_video_manifest_rows(
    rows: list[dict[str, str]],
    *,
    prompt_mode: str,
    hard_only: bool = False,
    model_names: list[str] | None = None,
    limit: int = 0,
) -> list[dict[str, str]]:
    """Build a local-manifest view over official test videos."""
    wanted_models = {_normalize_text(name).lower() for name in (model_names or []) if _normalize_text(name)}

    output_rows: list[dict[str, str]] = []
    for row in rows:
        if hard_only and not _is_truthy(row.get("is_hard")):
            continue

        model_name = _normalize_text(row.get("model_name"))
        if wanted_models and model_name.lower() not in wanted_models:
            continue

        prompt = _prompt_from_row(row, prompt_mode=prompt_mode)
        video_url = _normalize_text(row.get("video_url"))
        if not prompt or not video_url:
            continue

        ordinal = len(output_rows)
        output_rows.append(
            {
                "sample_id": f"videophy2_official_{prompt_mode}_{ordinal:04d}",
                "prompt": prompt,
                "caption": _normalize_text(row.get("caption")),
                "upsampled_caption": _normalize_text(row.get("upsampled_caption")),
                "action": _normalize_text(row.get("action")),
                "category": _normalize_text(row.get("category")),
                "is_hard": "1" if _is_truthy(row.get("is_hard")) else "0",
                "model_name": model_name,
                "video_url": video_url,
                "videopath": _relative_video_path(row, ordinal=ordinal),
                "source_mode": "official_test_video",
                "prompt_mode": prompt_mode,
                "sa_human": _normalize_text(row.get("sa")),
                "pc_human": _normalize_text(row.get("pc")),
                "joint_human": _normalize_text(row.get("joint")),
                "physics_rules_followed": _normalize_text(row.get("physics_rules_followed")),
                "physics_rules_unfollowed": _normalize_text(row.get("physics_rules_unfollowed")),
                "physics_rules_cannot_be_determined": _normalize_text(row.get("physics_rules_cannot_be_determined")),
                "human_violated_rules": _normalize_text(row.get("human_violated_rules")),
                "metadata_rules": _normalize_text(row.get("metadata_rules")),
            }
        )
        if limit > 0 and len(output_rows) >= limit:
            break
    return output_rows


def download_official_video_subset(
    rows: list[dict[str, str]],
    *,
    video_root: str | Path,
    overwrite: bool = False,
    timeout_sec: float = 120.0,
    downloader: Callable[[str, Path, float], None] | None = None,
) -> None:
    """Download official VideoPhy-2 videos referenced by the manifest rows."""
    root = Path(video_root)
    root.mkdir(parents=True, exist_ok=True)

    def _default_downloader(url: str, dest_path: Path, timeout: float) -> None:
        with urlopen(url, timeout=timeout) as response, dest_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    fetch = downloader or _default_downloader
    for row in rows:
        url = _normalize_text(row.get("video_url"))
        rel_path = _normalize_text(row.get("videopath"))
        if not url or not rel_path:
            continue
        dest_path = root / rel_path
        if dest_path.exists() and dest_path.stat().st_size > 0 and not overwrite:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        fetch(url, dest_path, timeout_sec)


def write_official_video_subset_manifest(
    *,
    official_test_csv: str | Path,
    manifest_path: str | Path,
    video_root: str | Path,
    prompt_mode: str,
    hard_only: bool = False,
    model_names: list[str] | None = None,
    limit: int = 0,
    download_videos: bool = False,
    overwrite_videos: bool = False,
    timeout_sec: float = 120.0,
) -> PreparedOfficialVideoSubset:
    """Write a local manifest over official test videos and optionally download them."""
    rows = build_official_video_manifest_rows(
        read_csv_rows(official_test_csv),
        prompt_mode=prompt_mode,
        hard_only=hard_only,
        model_names=model_names,
        limit=limit,
    )

    fieldnames = [
        "sample_id",
        "prompt",
        "caption",
        "upsampled_caption",
        "action",
        "category",
        "is_hard",
        "model_name",
        "video_url",
        "videopath",
        "source_mode",
        "prompt_mode",
        "sa_human",
        "pc_human",
        "joint_human",
        "physics_rules_followed",
        "physics_rules_unfollowed",
        "physics_rules_cannot_be_determined",
        "human_violated_rules",
        "metadata_rules",
    ]
    write_csv_rows(manifest_path, rows, fieldnames)

    if download_videos:
        download_official_video_subset(
            rows,
            video_root=video_root,
            overwrite=overwrite_videos,
            timeout_sec=timeout_sec,
        )

    return PreparedOfficialVideoSubset(
        manifest_path=Path(manifest_path),
        video_root=Path(video_root),
        count=len(rows),
        prompt_mode=prompt_mode,
        hard_only=hard_only,
        model_names=[name for name in (model_names or []) if _normalize_text(name)],
    )


def merge_rows_with_test_metadata(
    rows: list[dict[str, str]],
    *,
    metadata_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Fill prompt rows with metadata from the official test table keyed by caption."""
    metadata_by_caption: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        caption = _normalize_text(row.get("caption"))
        if caption and caption not in metadata_by_caption:
            metadata_by_caption[caption] = row

    merged: list[dict[str, str]] = []
    for row in rows:
        caption = _normalize_text(row.get("caption"))
        meta = metadata_by_caption.get(caption, {})
        merged.append(
            {
                **meta,
                **row,
                "caption": caption,
                "upsampled_caption": _normalize_text(row.get("upsampled_caption"))
                or _normalize_text(meta.get("upsampled_caption")),
                "action": _normalize_text(meta.get("action")),
                "category": _normalize_text(meta.get("category")),
                "is_hard": "1" if _is_truthy(meta.get("is_hard")) else "0",
            }
        )
    return merged


def write_official_prompt_manifests(
    *,
    official_test_csv: str | Path,
    manifest_dir: str | Path,
    official_upsampled_test_csv: str | Path | None = None,
) -> PreparedPromptManifests:
    """Write original-prompt and upsampled-prompt benchmark manifests."""
    rows = read_csv_rows(official_test_csv)
    manifest_root = Path(manifest_dir)
    manifest_root.mkdir(parents=True, exist_ok=True)

    original_rows = build_prompt_manifest_rows(rows, prompt_mode="original")
    original_hard_rows = [row for row in original_rows if _is_truthy(row.get("is_hard"))]
    upsampled_source_rows = rows
    if official_upsampled_test_csv:
        upsampled_source_rows = merge_rows_with_test_metadata(
            read_csv_rows(official_upsampled_test_csv),
            metadata_rows=rows,
        )
    upsampled_rows = build_prompt_manifest_rows(upsampled_source_rows, prompt_mode="upsampled")
    upsampled_hard_rows = [row for row in upsampled_rows if _is_truthy(row.get("is_hard"))]

    fieldnames = [
        "sample_id",
        "prompt",
        "caption",
        "upsampled_caption",
        "action",
        "category",
        "is_hard",
        "source_mode",
    ]
    original_manifest = manifest_root / "videophy2_test_original.csv"
    original_hard_manifest = manifest_root / "videophy2_test_original_hard.csv"
    upsampled_manifest = manifest_root / "videophy2_test_upsampled.csv"
    upsampled_hard_manifest = manifest_root / "videophy2_test_upsampled_hard.csv"

    write_csv_rows(original_manifest, original_rows, fieldnames)
    write_csv_rows(original_hard_manifest, original_hard_rows, fieldnames)
    write_csv_rows(upsampled_manifest, upsampled_rows, fieldnames)
    write_csv_rows(upsampled_hard_manifest, upsampled_hard_rows, fieldnames)

    return PreparedPromptManifests(
        original_manifest=original_manifest,
        original_hard_manifest=original_hard_manifest,
        upsampled_manifest=upsampled_manifest,
        upsampled_hard_manifest=upsampled_hard_manifest,
        original_count=len(original_rows),
        original_hard_count=len(original_hard_rows),
        upsampled_count=len(upsampled_rows),
        upsampled_hard_count=len(upsampled_hard_rows),
    )


def read_csv_header(csv_path: str | Path) -> list[str]:
    """Return a CSV header without loading the full table."""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])
