"""Helpers for preparing official VideoPhy-2 benchmark assets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

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
