"""Small file I/O helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .defaults import PROJECT_ROOT


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if missing and return it as a Path."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write YAML with stable ordering."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with indentation."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows preserving header order."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write rows to a CSV file."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_project_path(path: str | Path | None) -> str:
    """Resolve project-relative paths while preserving absolute inputs."""
    raw = str(path or "").strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).absolute())
