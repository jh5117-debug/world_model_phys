"""Build a smaller CSGO test split from metadata_val.csv with diverse coverage."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from physical_consistency.common.io import ensure_dir, read_csv_rows, write_csv_rows


@dataclass(slots=True)
class TestSubsetBuildResult:
    """Summary of a generated test subset."""

    output_metadata_path: Path
    output_clips_dir: Path
    total_rows: int
    selected_rows: int
    source_metadata_path: Path
    target_split: str


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "::".join((str(base_seed), *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _extract_team(row: dict[str, str]) -> str:
    clip_path = str(row.get("clip_path", ""))
    marker = "team_"
    if marker not in clip_path:
        return "unknown"
    tail = clip_path.split(marker, 1)[1]
    return tail.split("_", 1)[0]


def _extract_player(row: dict[str, str]) -> str:
    clip_path = str(row.get("clip_path", ""))
    marker = "player_"
    if marker not in clip_path:
        return "unknown"
    tail = clip_path.split(marker, 1)[1]
    return tail.split("_", 1)[0]


def _extract_weapon(row: dict[str, str]) -> str:
    prompt = str(row.get("prompt", ""))
    marker = "holding a "
    if marker not in prompt:
        return "UNKNOWN"
    tail = prompt.split(marker, 1)[1]
    return tail.split(".", 1)[0].strip() or "UNKNOWN"


def _allocate_quotas(group_sizes: dict[str, int], *, sample_count: int) -> dict[str, int]:
    nonempty = {key: size for key, size in group_sizes.items() if size > 0}
    if not nonempty or sample_count <= 0:
        return {key: 0 for key in group_sizes}

    sample_count = min(sample_count, sum(nonempty.values()))
    quotas = {key: 0 for key in group_sizes}

    guaranteed = min(sample_count, len(nonempty))
    ordered_keys = sorted(nonempty)
    for key in ordered_keys[:guaranteed]:
        quotas[key] = 1
    remaining = sample_count - guaranteed
    if remaining <= 0:
        return quotas

    residual_capacity = {key: nonempty[key] - quotas[key] for key in nonempty}
    total_capacity = sum(residual_capacity.values())
    if total_capacity <= 0:
        return quotas

    fractional_parts: list[tuple[float, str]] = []
    for key, capacity in residual_capacity.items():
        exact = remaining * capacity / total_capacity
        base = min(capacity, int(exact))
        quotas[key] += base
        fractional_parts.append((exact - base, key))

    assigned = sum(quotas.values())
    leftovers = sample_count - assigned
    for _, key in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
        if leftovers <= 0:
            break
        if quotas[key] < nonempty[key]:
            quotas[key] += 1
            leftovers -= 1

    return quotas


def _diverse_sample_rows(rows: list[dict[str, str]], *, sample_count: int, seed: int) -> list[dict[str, str]]:
    if sample_count <= 0:
        return []
    if sample_count >= len(rows):
        return list(rows)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(_extract_team(row), _extract_player(row))].append(row)

    queues: list[list[dict[str, str]]] = []
    for key in sorted(grouped):
        queue = list(grouped[key])
        random.Random(_stable_seed(seed, key[0], key[1])).shuffle(queue)
        queues.append(queue)

    selected: list[dict[str, str]] = []
    while len(selected) < sample_count and any(queues):
        for queue in queues:
            if not queue:
                continue
            selected.append(queue.pop(0))
            if len(selected) >= sample_count:
                break
    return selected


def select_diverse_test_rows(
    rows: list[dict[str, str]],
    *,
    sample_count: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select a diverse subset while roughly preserving weapon frequencies."""
    if sample_count >= len(rows):
        return list(rows)

    weapon_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    order_index = {id(row): idx for idx, row in enumerate(rows)}
    for row in rows:
        weapon_groups[_extract_weapon(row)].append(row)

    quotas = _allocate_quotas(
        {weapon: len(group_rows) for weapon, group_rows in weapon_groups.items()},
        sample_count=sample_count,
    )

    selected: list[dict[str, str]] = []
    for weapon in sorted(weapon_groups):
        group_rows = weapon_groups[weapon]
        quota = min(quotas.get(weapon, 0), len(group_rows))
        picked = _diverse_sample_rows(
            group_rows,
            sample_count=quota,
            seed=_stable_seed(seed, weapon),
        )
        selected.extend(picked)

    selected = sorted(selected, key=lambda row: order_index[id(row)])
    return selected[:sample_count]


def build_csgo_test_subset(
    dataset_dir: str | Path,
    *,
    sample_count: int = 80,
    seed: int = 0,
    source_split: str = "val",
    target_split: str = "test",
    link_mode: str = "symlink",
    overwrite: bool = True,
) -> TestSubsetBuildResult:
    """Create metadata_test.csv plus test/clips symlinks from metadata_val.csv."""
    dataset_root = Path(dataset_dir)
    source_metadata_path = dataset_root / f"metadata_{source_split}.csv"
    output_metadata_path = dataset_root / f"metadata_{target_split}.csv"
    output_split_root = dataset_root / target_split
    output_clips_dir = output_split_root / "clips"

    rows = read_csv_rows(source_metadata_path)
    if not rows:
        raise ValueError(f"No rows found in {source_metadata_path}")

    selected = select_diverse_test_rows(rows, sample_count=min(sample_count, len(rows)), seed=seed)

    if overwrite and output_split_root.exists():
        shutil.rmtree(output_split_root)
    ensure_dir(output_clips_dir)

    rewritten_rows: list[dict[str, str]] = []
    for row in selected:
        clip_name = Path(str(row["clip_path"])).name
        src_clip_dir = dataset_root / str(row["clip_path"])
        dst_clip_dir = output_clips_dir / clip_name
        if dst_clip_dir.exists() or dst_clip_dir.is_symlink():
            if dst_clip_dir.is_dir() and not dst_clip_dir.is_symlink():
                shutil.rmtree(dst_clip_dir)
            else:
                dst_clip_dir.unlink()
        if link_mode == "copy":
            shutil.copytree(src_clip_dir, dst_clip_dir)
        else:
            dst_clip_dir.symlink_to(src_clip_dir, target_is_directory=True)

        new_row = dict(row)
        new_row["clip_path"] = f"{target_split}/clips/{clip_name}"
        if "video" in new_row:
            new_row["video"] = f"{target_split}/clips/{clip_name}/video.mp4"
        rewritten_rows.append(new_row)

    fieldnames = list(rewritten_rows[0].keys())
    write_csv_rows(output_metadata_path, rewritten_rows, fieldnames)
    return TestSubsetBuildResult(
        output_metadata_path=output_metadata_path,
        output_clips_dir=output_clips_dir,
        total_rows=len(rows),
        selected_rows=len(rewritten_rows),
        source_metadata_path=source_metadata_path,
        target_split=target_split,
    )
