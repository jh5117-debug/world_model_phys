"""Physics-IQ-style paired video evaluation.

This module adapts the public Physics-IQ metric recipe to our paired-video
setting where each generated clip is compared against its matched real CSGO
reference clip. It is intentionally labeled "Physics-IQ-style" because it does
not use the official Physics-IQ benchmark scenarios, takes, and masks.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physical_consistency.common.defaults import DEFAULT_SEED_LIST
from physical_consistency.common.io import (
    ensure_dir,
    read_csv_rows,
    read_json,
    read_yaml,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from physical_consistency.common.path_config import resolve_path_config


@dataclass(slots=True)
class PhysicsIQConfig:
    """Config for paired Physics-IQ-style evaluation."""

    seed_list: list[int]
    output_root: str
    compare_seconds: float = 5.0
    sample_frames: int = 40
    resize_divisor: int = 4
    mask_threshold: int = 10
    max_samples: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhysicsIQConfig":
        payload = read_yaml(path)
        payload.setdefault("seed_list", DEFAULT_SEED_LIST)
        if "output_root" in payload:
            payload["output_root"] = resolve_project_path(payload["output_root"])
        return cls(**payload)


def resolve_video_path(
    row: dict[str, str],
    *,
    source_root: str | Path,
    source_mode: str,
    video_filename: str = "video.mp4",
    video_suffix: str = "_gen.mp4",
) -> Path:
    """Resolve a manifest row to a concrete video path."""
    root = Path(source_root)
    clip_path = row.get("clip_path", "")
    clip_name = Path(clip_path).name

    if source_mode == "generated":
        return root / f"{clip_name}{video_suffix}"
    if source_mode == "dataset_clip":
        return root / clip_path / video_filename
    if source_mode == "manifest_video":
        return root / row.get("video", "")
    if source_mode == "manifest_videopath":
        return Path(row.get("videopath", ""))
    raise ValueError(f"Unsupported Physics-IQ source mode: {source_mode}")


def build_physics_iq_input_csv(
    *,
    manifest_csv: str | Path,
    reference_source_root: str | Path,
    candidate_source_root: str | Path,
    output_csv: str | Path,
    reference_source_mode: str = "dataset_clip",
    candidate_source_mode: str = "dataset_clip",
    video_filename: str = "video.mp4",
    video_suffix: str = "_gen.mp4",
    max_samples: int = 0,
) -> Path:
    """Build paired reference/candidate CSV expected by our evaluator."""
    rows = read_csv_rows(manifest_csv)
    output_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        reference_path = resolve_video_path(
            row,
            source_root=reference_source_root,
            source_mode=reference_source_mode,
            video_filename=video_filename,
            video_suffix=video_suffix,
        )
        candidate_path = resolve_video_path(
            row,
            source_root=candidate_source_root,
            source_mode=candidate_source_mode,
            video_filename=video_filename,
            video_suffix=video_suffix,
        )
        if not reference_path.exists() or not candidate_path.exists():
            continue
        output_rows.append(
            {
                "sample_id": row.get("clip_path", Path(candidate_path).stem) or f"sample_{idx:05d}",
                "clip_path": row.get("clip_path", ""),
                "prompt": row.get("prompt", ""),
                "reference_videopath": str(reference_path),
                "candidate_videopath": str(candidate_path),
            }
        )
        if max_samples > 0 and len(output_rows) >= max_samples:
            break

    write_csv_rows(
        output_csv,
        output_rows,
        ["sample_id", "clip_path", "prompt", "reference_videopath", "candidate_videopath"],
    )
    return Path(output_csv)


def _load_video_frames(path: str | Path, *, compare_seconds: float) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 1.0
    max_frames = max(1, int(round(compare_seconds * fps)))
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {path}")
    return frames, fps


def _resize_frames(frames: list[np.ndarray], target_size: tuple[int, int]) -> list[np.ndarray]:
    return [cv2.resize(frame, target_size) for frame in frames]


def _sample_sequence(items: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    if len(items) <= target_count:
        return items
    indices = np.linspace(0, len(items) - 1, target_count).round().astype(int)
    return [items[int(idx)] for idx in indices]


def _generate_motion_masks(
    frames: list[np.ndarray],
    *,
    threshold: int,
) -> list[np.ndarray]:
    """Generate binary motion masks following the official Physics-IQ recipe."""
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    gray0 = cv2.GaussianBlur(gray0, (5, 5), 0)
    avg_frame = gray0.astype(np.float32)
    masks: list[np.ndarray] = [np.zeros_like(gray0, dtype=np.uint8)]
    kernel = np.ones((5, 5), np.uint8)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.accumulateWeighted(gray, avg_frame, 0.3)
        avg_gray = cv2.convertScaleAbs(avg_frame)
        diff = cv2.absdiff(gray, avg_gray)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        masks.append((binary > 127).astype(np.uint8))
    return masks


def _spatiotemporal_iou_per_frame(mask1: list[np.ndarray], mask2: list[np.ndarray]) -> list[float]:
    values: list[float] = []
    for a, b in zip(mask1, mask2):
        intersection = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        values.append(1.0 if union == 0 else intersection / union)
    return values


def _collapse_spatial_mask(mask_frames: list[np.ndarray]) -> np.ndarray:
    spatial_mask = np.max(mask_frames, axis=0)
    return (spatial_mask > 0).astype(np.uint8)


def _weighted_spatial_mask(mask_frames: list[np.ndarray]) -> np.ndarray:
    return np.sum(mask_frames, axis=0, dtype=np.float32) / max(len(mask_frames), 1)


def _weighted_spatial_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.minimum(mask1, mask2)
    union = np.maximum(mask1, mask2)
    valid = union > 0
    if not np.any(valid):
        return 1.0
    return float(np.sum(intersection[valid]) / np.sum(union[valid]))


def _mse_per_frame(video1: list[np.ndarray], video2: list[np.ndarray]) -> list[float]:
    values: list[float] = []
    for a, b in zip(video1, video2):
        values.append(float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))
    return values


def evaluate_video_pair(
    *,
    reference_videopath: str | Path,
    candidate_videopath: str | Path,
    compare_seconds: float,
    sample_frames: int,
    resize_divisor: int,
    mask_threshold: int,
) -> dict[str, Any]:
    """Evaluate one candidate video against its paired real reference."""
    reference_frames, _ = _load_video_frames(reference_videopath, compare_seconds=compare_seconds)
    candidate_frames, _ = _load_video_frames(candidate_videopath, compare_seconds=compare_seconds)

    ref_h, ref_w = reference_frames[0].shape[:2]
    target_size = (
        max(2, ref_w // max(resize_divisor, 1)),
        max(2, ref_h // max(resize_divisor, 1)),
    )

    reference_frames = _resize_frames(reference_frames, target_size)
    candidate_frames = _resize_frames(candidate_frames, target_size)
    reference_masks = _generate_motion_masks(reference_frames, threshold=mask_threshold)
    candidate_masks = _generate_motion_masks(candidate_frames, threshold=mask_threshold)

    target_count = min(len(reference_frames), len(candidate_frames), sample_frames)
    if target_count < 2:
        raise RuntimeError(
            f"Not enough comparable frames between {reference_videopath} and {candidate_videopath}"
        )

    reference_frames = _sample_sequence(reference_frames, target_count)
    candidate_frames = _sample_sequence(candidate_frames, target_count)
    reference_masks = _sample_sequence(reference_masks, target_count)
    candidate_masks = _sample_sequence(candidate_masks, target_count)

    reference_frames_norm = [frame.astype(np.float32) / 255.0 for frame in reference_frames]
    candidate_frames_norm = [frame.astype(np.float32) / 255.0 for frame in candidate_frames]

    mse = _mse_per_frame(reference_frames_norm, candidate_frames_norm)
    spatiotemporal_iou = _spatiotemporal_iou_per_frame(reference_masks, candidate_masks)

    reference_spatial = _collapse_spatial_mask(reference_masks)
    candidate_spatial = _collapse_spatial_mask(candidate_masks)
    spatial_iou = _spatiotemporal_iou_per_frame([reference_spatial], [candidate_spatial])[0]

    reference_weighted = _weighted_spatial_mask(reference_masks)
    candidate_weighted = _weighted_spatial_mask(candidate_masks)
    weighted_spatial_iou = _weighted_spatial_iou(reference_weighted, candidate_weighted)

    # Internal paired adaptation: use a perfect-variance identity baseline.
    score = (
        (
            float(np.mean(spatiotemporal_iou))
            + float(spatial_iou)
            + float(weighted_spatial_iou)
        )
        / 3.0
        - float(np.mean(mse))
    )
    score = round(float(np.clip(score * 100.0, 0.0, 100.0)), 2)

    return {
        "compare_frame_count": int(target_count),
        "mse_mean": round(float(np.mean(mse)), 6),
        "spatiotemporal_iou_mean": round(float(np.mean(spatiotemporal_iou)), 6),
        "spatial_iou": round(float(spatial_iou), 6),
        "weighted_spatial_iou": round(float(weighted_spatial_iou), 6),
        "physics_iq_style_score": score,
    }


def run_physics_iq_for_seed(
    *,
    manifest_csv: str | Path,
    reference_source_root: str | Path,
    candidate_source_root: str | Path,
    output_dir: str | Path,
    seed: int,
    compare_seconds: float,
    sample_frames: int,
    resize_divisor: int,
    mask_threshold: int,
    reference_source_mode: str = "dataset_clip",
    candidate_source_mode: str = "dataset_clip",
    video_filename: str = "video.mp4",
    video_suffix: str = "_gen.mp4",
    max_samples: int = 0,
) -> Path:
    """Run Physics-IQ-style scoring for one manifest/seed."""
    output_root = ensure_dir(output_dir)
    inputs_dir = ensure_dir(output_root / "inputs")
    input_csv = build_physics_iq_input_csv(
        manifest_csv=manifest_csv,
        reference_source_root=reference_source_root,
        candidate_source_root=candidate_source_root,
        output_csv=inputs_dir / f"seed_{seed}_pairs.csv",
        reference_source_mode=reference_source_mode,
        candidate_source_mode=candidate_source_mode,
        video_filename=video_filename,
        video_suffix=video_suffix,
        max_samples=max_samples,
    )

    rows = read_csv_rows(input_csv)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result = evaluate_video_pair(
            reference_videopath=row["reference_videopath"],
            candidate_videopath=row["candidate_videopath"],
            compare_seconds=compare_seconds,
            sample_frames=sample_frames,
            resize_divisor=resize_divisor,
            mask_threshold=mask_threshold,
        )
        output_rows.append({**row, **result})

    output_csv = output_root / "output_pairs.csv"
    fieldnames = [
        "sample_id",
        "clip_path",
        "prompt",
        "reference_videopath",
        "candidate_videopath",
        "compare_frame_count",
        "mse_mean",
        "spatiotemporal_iou_mean",
        "spatial_iou",
        "weighted_spatial_iou",
        "physics_iq_style_score",
    ]
    write_csv_rows(output_csv, output_rows, fieldnames)
    write_json(output_root / "summary.json", summarize_physics_iq_outputs(output_csv))
    return output_csv


def summarize_physics_iq_outputs(path: str | Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {"count": 0, "means": {}}

    numeric_keys = [
        "compare_frame_count",
        "mse_mean",
        "spatiotemporal_iou_mean",
        "spatial_iou",
        "weighted_spatial_iou",
        "physics_iq_style_score",
    ]
    aggregate: dict[str, Any] = {"count": len(rows), "means": {}}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        if values:
            aggregate["means"][key] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    return aggregate


def write_physics_iq_summary(output_dir: str | Path) -> Path:
    """Aggregate all seed directories under one experiment root."""
    root = Path(output_dir)
    seed_summaries = []
    for seed_dir in sorted(root.glob("seed_*")):
        pairs_csv = seed_dir / "output_pairs.csv"
        if not pairs_csv.exists():
            continue
        summary = summarize_physics_iq_outputs(pairs_csv)
        summary["seed"] = int(seed_dir.name.split("_")[-1])
        seed_summaries.append(summary)

    aggregate: dict[str, Any] = {"seeds": seed_summaries, "means": {}}
    metric_names = set()
    for seed_summary in seed_summaries:
        metric_names.update(seed_summary.get("means", {}).keys())

    for metric_name in sorted(metric_names):
        values = [
            float(seed_summary["means"][metric_name]["mean"])
            for seed_summary in seed_summaries
            if metric_name in seed_summary.get("means", {})
        ]
        if values:
            aggregate["means"][metric_name] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }

    summary_path = root / "summary.json"
    write_json(summary_path, aggregate)
    return summary_path


def main() -> None:
    """CLI main."""
    parser = argparse.ArgumentParser(description="Run Physics-IQ-style paired evaluation.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default="")
    parser.add_argument(
        "--reference_source_mode",
        type=str,
        default="dataset_clip",
        choices=["generated", "dataset_clip", "manifest_video", "manifest_videopath"],
    )
    parser.add_argument(
        "--candidate_source_mode",
        type=str,
        default="dataset_clip",
        choices=["generated", "dataset_clip", "manifest_video", "manifest_videopath"],
    )
    parser.add_argument("--reference_source_root", type=str, default="")
    parser.add_argument("--candidate_source_root", type=str, default="")
    parser.add_argument("--video_filename", type=str, default="video.mp4")
    parser.add_argument("--video_suffix", type=str, default="_gen.mp4")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--output_root", type=str, default="")
    args = parser.parse_args()

    cfg = PhysicsIQConfig.from_yaml(args.config)
    if args.seed >= 0:
        cfg.seed_list = [args.seed]
    path_cfg = resolve_path_config(args, env_file=args.env_file or None)
    output_root = (
        resolve_project_path(args.output_root) if args.output_root else cfg.output_root or path_cfg.output_root
    )
    manifest_csv = (
        resolve_project_path(args.manifest_csv)
        if args.manifest_csv
        else str(Path(path_cfg.dataset_dir) / "metadata_val.csv")
    )
    reference_source_root = (
        resolve_project_path(args.reference_source_root)
        if args.reference_source_root
        else path_cfg.dataset_dir
    )
    candidate_source_root = (
        resolve_project_path(args.candidate_source_root)
        if args.candidate_source_root
        else path_cfg.dataset_dir
    )

    base_output = Path(output_root) / "runs" / "eval" / "physics_iq" / args.experiment_name
    ensure_dir(base_output)

    if args.summary_only:
        summary_path = write_physics_iq_summary(base_output)
        print(json.dumps(read_json(summary_path), indent=2))
        return

    for seed in cfg.seed_list:
        run_physics_iq_for_seed(
            manifest_csv=manifest_csv,
            reference_source_root=reference_source_root,
            candidate_source_root=candidate_source_root,
            output_dir=base_output / f"seed_{seed}",
            seed=seed,
            compare_seconds=cfg.compare_seconds,
            sample_frames=cfg.sample_frames,
            resize_divisor=cfg.resize_divisor,
            mask_threshold=cfg.mask_threshold,
            reference_source_mode=args.reference_source_mode,
            candidate_source_mode=args.candidate_source_mode,
            video_filename=args.video_filename,
            video_suffix=args.video_suffix,
            max_samples=cfg.max_samples,
        )

    summary_path = write_physics_iq_summary(base_output)
    print(json.dumps(read_json(summary_path), indent=2))


if __name__ == "__main__":
    main()
