"""Single-GPU sharded full-val LingBot generation with rolling summaries."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from physical_consistency.common.io import (
    ensure_dir,
    read_csv_rows,
    read_json,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.summary_tables import format_lingbot_progress_summary
from physical_consistency.datasets.manifest_builder import materialize_dataset_view
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.eval.physics_iq import PhysicsIQConfig, evaluate_video_pair


@dataclass(slots=True)
class FullvalConfig:
    """Configuration for the sharded full-val pipeline."""

    manifest_path: str
    dataset_dir: str
    output_root: str
    base_model_dir: str
    stage1_ckpt_dir: str
    val_inf_root: str
    physics_config: str
    gpu_list: list[str]
    seed: int = 0
    frame_num: int = 81
    sample_steps: int = 70
    guide_scale: float = 5.0
    height: int = 480
    width: int = 832
    control_type: str = "act"
    models: str = "both"
    video_filename: str = "video.mp4"
    video_suffix: str = "_gen.mp4"
    report_every: int = 10
    poll_seconds: int = 15


@dataclass(slots=True)
class ModelSpec:
    """One LingBot model variant to run."""

    model_label: str
    subdir_name: str
    base_model_dir: str
    ft_ckpt_dir: str = ""


@dataclass(slots=True)
class WorkerProc:
    """A running single-GPU LingBot worker."""

    gpu_id: str
    shard_index: int
    shard_dir: Path
    output_dir: Path
    log_path: Path
    process: subprocess.Popen[str]
    log_handle: TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sharded single-GPU LingBot base/stage1 generation with rolling Physics-IQ + PSNR summaries."
    )
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    parser.add_argument("--val_inf_root", type=str, default="")
    parser.add_argument("--physics_config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=70)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--control_type", type=str, default="act", choices=["act", "cam"])
    parser.add_argument("--models", type=str, default="both", choices=["both", "base", "stage1"])
    parser.add_argument("--video_filename", type=str, default="video.mp4")
    parser.add_argument("--video_suffix", type=str, default="_gen.mp4")
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--poll_seconds", type=int, default=15)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=0)
    return parser.parse_args()


def _resolve_gpu_list(args: argparse.Namespace) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        gpu_list = [item.strip() for item in visible_devices.split(",") if item.strip()]
    else:
        count = args.num_gpus if args.num_gpus > 0 else 8
        gpu_list = [str(idx) for idx in range(count)]
    if args.num_gpus > 0:
        gpu_list = gpu_list[: args.num_gpus]
    if not gpu_list:
        raise ValueError("No GPUs resolved for LingBot full-val pipeline.")
    return gpu_list


def _build_config(args: argparse.Namespace) -> FullvalConfig:
    path_cfg = resolve_path_config(args, env_file=args.env_file or None)
    dataset_dir = resolve_project_path(args.dataset_dir) if args.dataset_dir else path_cfg.dataset_dir
    manifest_path = (
        resolve_project_path(args.manifest_path)
        if args.manifest_path
        else str(Path(dataset_dir) / "metadata_val.csv")
    )
    output_root = resolve_project_path(args.output_root) if args.output_root else path_cfg.output_root
    base_model_dir = (
        resolve_project_path(args.base_model_dir) if args.base_model_dir else path_cfg.base_model_dir
    )
    stage1_ckpt_dir = (
        resolve_project_path(args.stage1_ckpt_dir)
        if args.stage1_ckpt_dir
        else path_cfg.stage1_ckpt_dir
    )
    val_inf_root = (
        resolve_project_path(args.val_inf_root)
        if args.val_inf_root
        else str(Path(dataset_dir) / "val_inf_result")
    )
    physics_config = (
        resolve_project_path(args.physics_config)
        if args.physics_config
        else str(Path(output_root) / "configs" / "physics_iq_dataset_eval.yaml")
    )
    return FullvalConfig(
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        output_root=output_root,
        base_model_dir=base_model_dir,
        stage1_ckpt_dir=stage1_ckpt_dir,
        val_inf_root=val_inf_root,
        physics_config=physics_config,
        gpu_list=_resolve_gpu_list(args),
        seed=args.seed,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        height=args.height,
        width=args.width,
        control_type=args.control_type,
        models=args.models,
        video_filename=args.video_filename,
        video_suffix=args.video_suffix,
        report_every=max(1, args.report_every),
        poll_seconds=max(1, args.poll_seconds),
    )


def shard_rows(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    """Round-robin shard rows across GPUs for better load balancing."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shards = [[] for _ in range(shard_count)]
    for idx, row in enumerate(rows):
        shards[idx % shard_count].append(row)
    return shards


def _clip_name(row: dict[str, str]) -> str:
    return Path(row["clip_path"]).name


def _safe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _compute_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_progress_row(
    *,
    model_label: str,
    processed_count: int,
    total_count: int,
    metrics_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one rolling summary row for terminal output."""
    mean_score = _compute_mean(physics_rows, "physics_iq_style_score")
    mean_psnr = _compute_mean(metrics_rows, "psnr")
    return {
        "Model": model_label,
        "Processed": processed_count,
        "Total": total_count,
        "Mean Physics-IQ Score": mean_score if mean_score is not None else "",
        "Mean PSNR": mean_psnr if mean_psnr is not None else "",
    }


def _read_video_frames(video_path: str | Path, *, max_frames: int, height: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return np.stack(frames)


def compute_video_psnr(
    *,
    reference_videopath: str | Path,
    candidate_videopath: str | Path,
    frame_num: int,
    height: int,
    width: int,
) -> float:
    """Compute PSNR using the same resizing/frame-alignment logic as eval_batch.py."""
    gt_frames = _read_video_frames(reference_videopath, max_frames=frame_num, height=height, width=width)
    gen_frames = _read_video_frames(candidate_videopath, max_frames=frame_num, height=height, width=width)
    min_frames = min(len(gt_frames), len(gen_frames))
    gt_frames = gt_frames[:min_frames]
    gen_frames = gen_frames[:min_frames]
    mse = float(np.mean((gen_frames.astype(np.float64) - gt_frames.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0**2 / mse)


def summarize_physics_iq_outputs_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize already-loaded Physics-IQ per-pair rows."""
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
        values = [_safe_float(row.get(key)) for row in rows]
        filtered = [value for value in values if value is not None]
        if filtered:
            aggregate["means"][key] = {
                "mean": sum(filtered) / len(filtered),
                "count": len(filtered),
            }
    return aggregate


def _build_physics_rollup_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    if not rows:
        return {"seeds": [], "means": {}}
    summary = summarize_physics_iq_outputs_from_rows(rows)
    return {
        "seeds": [{"seed": seed, "count": summary["count"], "means": summary["means"]}],
        "means": summary["means"],
    }


def _build_eval_report(
    rows: list[dict[str, Any]],
    *,
    manifest_count: int,
    cfg: FullvalConfig,
    model: ModelSpec,
) -> dict[str, Any]:
    valid = [row for row in rows if _safe_float(row.get("psnr")) is not None]
    report: dict[str, Any] = {
        "config": {
            "ckpt_dir": model.base_model_dir,
            "ft_ckpt_dir": model.ft_ckpt_dir,
            "split": "val",
            "num_clips": manifest_count,
            "num_evaluated": len(valid),
            "sampling_steps": cfg.sample_steps,
            "guide_scale": cfg.guide_scale,
            "frame_num": cfg.frame_num,
            "resolution": f"{cfg.height}x{cfg.width}",
        },
        "aggregate_metrics": {},
        "per_clip": rows,
    }
    values = [_safe_float(row.get("psnr")) for row in valid]
    filtered = [value for value in values if value is not None]
    if filtered:
        mean_value = sum(filtered) / len(filtered)
        report["aggregate_metrics"]["psnr"] = {
            "mean": round(mean_value, 4),
            "std": round((sum((value - mean_value) ** 2 for value in filtered) / len(filtered)) ** 0.5, 4),
            "min": round(min(filtered), 4),
            "max": round(max(filtered), 4),
        }
    return report


def _sorted_by_manifest_order(
    rows: list[dict[str, Any]],
    *,
    index_by_clip: dict[str, int],
    key_name: str,
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: index_by_clip.get(str(row.get(key_name, "")), 10**9))


def _write_model_rollups(
    *,
    model_root: Path,
    cfg: FullvalConfig,
    model: ModelSpec,
    manifest_count: int,
    metrics_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
    index_by_clip: dict[str, int],
) -> None:
    sorted_metrics = _sorted_by_manifest_order(metrics_rows, index_by_clip=index_by_clip, key_name="clip_name")
    sorted_physics = _sorted_by_manifest_order(
        physics_rows,
        index_by_clip=index_by_clip,
        key_name="clip_name",
    )
    if sorted_metrics:
        write_csv_rows(
            model_root / "metrics.csv",
            sorted_metrics,
            [
                "clip_name",
                "clip_path",
                "prompt",
                "reference_videopath",
                "candidate_videopath",
                "psnr",
            ],
        )
    write_json(
        model_root / "eval_report.json",
        _build_eval_report(rows=sorted_metrics, manifest_count=manifest_count, cfg=cfg, model=model),
    )
    write_csv_rows(
        model_root / "physics_iq_output_pairs.csv",
        sorted_physics,
        [
            "clip_name",
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
        ],
    )
    write_json(
        model_root / "physics_iq_summary.json",
        _build_physics_rollup_summary(sorted_physics, seed=cfg.seed),
    )


def _load_existing_rows(
    model_root: Path,
    *,
    index_by_clip: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    metrics_rows = read_csv_rows(model_root / "metrics.csv") if (model_root / "metrics.csv").exists() else []
    physics_rows = (
        read_csv_rows(model_root / "physics_iq_output_pairs.csv")
        if (model_root / "physics_iq_output_pairs.csv").exists()
        else []
    )
    physics_by_clip = {str(row.get("clip_name", "")): row for row in physics_rows}
    filtered_metrics = []
    filtered_physics = []
    processed = set()
    for row in _sorted_by_manifest_order(metrics_rows, index_by_clip=index_by_clip, key_name="clip_name"):
        clip_name = str(row.get("clip_name", ""))
        if clip_name and clip_name in physics_by_clip:
            filtered_metrics.append(row)
            filtered_physics.append(physics_by_clip[clip_name])
            processed.add(clip_name)
    return filtered_metrics, filtered_physics, processed


def _build_models(cfg: FullvalConfig) -> list[ModelSpec]:
    model_specs: list[ModelSpec] = []
    if cfg.models in {"both", "base"}:
        model_specs.append(
            ModelSpec(
                model_label="LingBot-base",
                subdir_name="lingbotbase",
                base_model_dir=cfg.base_model_dir,
            )
        )
    if cfg.models in {"both", "stage1"}:
        model_specs.append(
            ModelSpec(
                model_label="LingBot-Stage1",
                subdir_name="lingbotstage1",
                base_model_dir=cfg.base_model_dir,
                ft_ckpt_dir=cfg.stage1_ckpt_dir,
            )
        )
    return model_specs


def _require_existing_path(label: str, value: str) -> None:
    path = Path(value)
    if not value or not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _materialize_worker_view(
    *,
    cfg: FullvalConfig,
    model: ModelSpec,
    shard_rows_payload: list[dict[str, str]],
    shard_index: int,
    cache_root: Path,
    manifests_root: Path,
) -> tuple[Path, Path]:
    shard_name = f"gpu_{shard_index}"
    manifest_path = manifests_root / f"{shard_name}.csv"
    if shard_rows_payload:
        write_csv_rows(manifest_path, shard_rows_payload, list(shard_rows_payload[0].keys()))
    else:
        write_csv_rows(manifest_path, [], ["prompt", "video", "clip_path", "map", "episode_id", "stem", "num_frames"])
    view_dir = cache_root / shard_name
    if view_dir.exists():
        shutil.rmtree(view_dir)
    if shard_rows_payload:
        materialize_dataset_view(cfg.dataset_dir, manifest_path, view_dir)
    return manifest_path, view_dir


def _spawn_workers(
    *,
    cfg: FullvalConfig,
    model: ModelSpec,
    path_cfg: Any,
    shard_payloads: list[list[dict[str, str]]],
) -> list[WorkerProc]:
    workers: list[WorkerProc] = []
    model_root = Path(cfg.val_inf_root) / model.subdir_name
    workers_root = ensure_dir(model_root / "workers")
    cache_root = ensure_dir(Path(cfg.output_root) / "cache" / "fullval_shards" / model.subdir_name)
    manifests_root = ensure_dir(model_root / "worker_manifests")
    logs_root = ensure_dir(Path(cfg.output_root) / "logs" / "fullval")

    effective_ft_ckpt_dir = ""
    if model.ft_ckpt_dir:
        effective_ft_ckpt_dir = str(
            materialize_eval_checkpoint_bundle(
                ft_ckpt_dir=model.ft_ckpt_dir,
                output_root=cfg.output_root,
                experiment_name=f"{model.subdir_name}_fullval_single_gpu",
                stage1_ckpt_dir=cfg.stage1_ckpt_dir,
                allow_stage1_fallback=False,
            )
        )

    for shard_index, gpu_id in enumerate(cfg.gpu_list):
        shard_rows_payload = shard_payloads[shard_index] if shard_index < len(shard_payloads) else []
        if not shard_rows_payload:
            continue
        worker_dir = ensure_dir(workers_root / f"gpu_{gpu_id}")
        if worker_dir.exists():
            for child in worker_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        ensure_dir(worker_dir / "videos")
        _, view_dir = _materialize_worker_view(
            cfg=cfg,
            model=model,
            shard_rows_payload=shard_rows_payload,
            shard_index=shard_index,
            cache_root=cache_root,
            manifests_root=manifests_root,
        )
        log_path = logs_root / f"{model.subdir_name}_gpu_{gpu_id}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "eval_batch.py",
            "--ckpt_dir",
            model.base_model_dir,
            "--lingbot_code_dir",
            path_cfg.lingbot_code_dir,
            "--dataset_dir",
            str(view_dir),
            "--output_dir",
            str(worker_dir),
            "--split",
            "val",
            "--max_samples",
            "0",
            "--skip_metrics",
            "--sample_steps",
            str(cfg.sample_steps),
            "--guide_scale",
            str(cfg.guide_scale),
            "--frame_num",
            str(cfg.frame_num),
            "--height",
            str(cfg.height),
            "--width",
            str(cfg.width),
            "--seed",
            str(cfg.seed),
            "--control_type",
            cfg.control_type,
            "--ulysses_size",
            "1",
        ]
        if effective_ft_ckpt_dir:
            command.extend(["--ft_ckpt_dir", effective_ft_ckpt_dir])
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(
            command,
            cwd=path_cfg.finetune_code_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append(
            WorkerProc(
                gpu_id=gpu_id,
                shard_index=shard_index,
                shard_dir=view_dir,
                output_dir=worker_dir,
                log_path=log_path,
                process=process,
                log_handle=log_handle,
            )
        )
    return workers


def _reference_video_path(cfg: FullvalConfig, row: dict[str, str]) -> Path:
    return Path(cfg.dataset_dir) / row["clip_path"] / cfg.video_filename


def _candidate_video_name(cfg: FullvalConfig, clip_name: str) -> str:
    return f"{clip_name}{cfg.video_suffix}"


def _maybe_process_video(
    *,
    cfg: FullvalConfig,
    physics_cfg: PhysicsIQConfig,
    row: dict[str, str],
    worker_video_path: Path,
    common_video_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = _reference_video_path(cfg, row)
    psnr = compute_video_psnr(
        reference_videopath=reference_path,
        candidate_videopath=worker_video_path,
        frame_num=cfg.frame_num,
        height=cfg.height,
        width=cfg.width,
    )
    physics_result = evaluate_video_pair(
        reference_videopath=reference_path,
        candidate_videopath=worker_video_path,
        compare_seconds=physics_cfg.compare_seconds,
        sample_frames=physics_cfg.sample_frames,
        resize_divisor=physics_cfg.resize_divisor,
        mask_threshold=physics_cfg.mask_threshold,
    )
    ensure_dir(common_video_path.parent)
    shutil.copy2(worker_video_path, common_video_path)
    clip_name = _clip_name(row)
    metrics_row = {
        "clip_name": clip_name,
        "clip_path": row["clip_path"],
        "prompt": row.get("prompt", ""),
        "reference_videopath": str(reference_path),
        "candidate_videopath": str(common_video_path),
        "psnr": round(psnr, 4),
    }
    physics_row = {
        "clip_name": clip_name,
        "sample_id": row["clip_path"],
        "clip_path": row["clip_path"],
        "prompt": row.get("prompt", ""),
        "reference_videopath": str(reference_path),
        "candidate_videopath": str(common_video_path),
        **physics_result,
    }
    return metrics_row, physics_row


def _scan_new_outputs(
    *,
    cfg: FullvalConfig,
    physics_cfg: PhysicsIQConfig,
    model_root: Path,
    workers: list[WorkerProc],
    row_by_clip: dict[str, dict[str, str]],
    processed_clips: set[str],
    metrics_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
) -> int:
    new_count = 0
    common_videos_dir = ensure_dir(model_root / "videos")
    for worker in workers:
        worker_videos_dir = worker.output_dir / "videos"
        for video_path in sorted(worker_videos_dir.glob(f"*{cfg.video_suffix}")):
            clip_name = video_path.name[: -len(cfg.video_suffix)]
            if clip_name in processed_clips:
                continue
            row = row_by_clip.get(clip_name)
            if row is None:
                continue
            common_video_path = common_videos_dir / video_path.name
            try:
                metrics_row, physics_row = _maybe_process_video(
                    cfg=cfg,
                    physics_cfg=physics_cfg,
                    row=row,
                    worker_video_path=video_path,
                    common_video_path=common_video_path,
                )
            except Exception:
                continue
            metrics_rows.append(metrics_row)
            physics_rows.append(physics_row)
            processed_clips.add(clip_name)
            new_count += 1
    return new_count


def run_model_fullval(
    *,
    cfg: FullvalConfig,
    physics_cfg: PhysicsIQConfig,
    path_cfg: Any,
    model: ModelSpec,
    manifest_rows: list[dict[str, str]],
) -> dict[str, Any]:
    model_root = ensure_dir(Path(cfg.val_inf_root) / model.subdir_name)
    ensure_dir(model_root / "videos")
    write_csv_rows(model_root / "run_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    index_by_clip = {_clip_name(row): idx for idx, row in enumerate(manifest_rows)}
    row_by_clip = {_clip_name(row): row for row in manifest_rows}
    metrics_rows, physics_rows, processed_clips = _load_existing_rows(model_root, index_by_clip=index_by_clip)
    if processed_clips:
        _write_model_rollups(
            model_root=model_root,
            cfg=cfg,
            model=model,
            manifest_count=len(manifest_rows),
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
            index_by_clip=index_by_clip,
        )
        print(
            format_lingbot_progress_summary(
                [
                    build_progress_row(
                        model_label=model.model_label,
                        processed_count=len(processed_clips),
                        total_count=len(manifest_rows),
                        metrics_rows=metrics_rows,
                        physics_rows=physics_rows,
                    )
                ],
                title=f"LingBot Rolling Summary: {model.model_label} (resume)",
            )
        )

    remaining_rows = [row for row in manifest_rows if _clip_name(row) not in processed_clips]
    if not remaining_rows:
        return build_progress_row(
            model_label=model.model_label,
            processed_count=len(processed_clips),
            total_count=len(manifest_rows),
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
        )

    shard_payloads = shard_rows(remaining_rows, len(cfg.gpu_list))
    workers = _spawn_workers(
        cfg=cfg,
        model=model,
        path_cfg=path_cfg,
        shard_payloads=shard_payloads,
    )
    next_report_threshold = ((len(processed_clips) // cfg.report_every) + 1) * cfg.report_every
    failed_workers: list[tuple[str, int]] = []

    try:
        while True:
            _scan_new_outputs(
                cfg=cfg,
                physics_cfg=physics_cfg,
                model_root=model_root,
                workers=workers,
                row_by_clip=row_by_clip,
                processed_clips=processed_clips,
                metrics_rows=metrics_rows,
                physics_rows=physics_rows,
            )
            _write_model_rollups(
                model_root=model_root,
                cfg=cfg,
                model=model,
                manifest_count=len(manifest_rows),
                metrics_rows=metrics_rows,
                physics_rows=physics_rows,
                index_by_clip=index_by_clip,
            )

            while len(processed_clips) >= next_report_threshold:
                print(
                    format_lingbot_progress_summary(
                        [
                            build_progress_row(
                                model_label=model.model_label,
                                processed_count=len(processed_clips),
                                total_count=len(manifest_rows),
                                metrics_rows=metrics_rows,
                                physics_rows=physics_rows,
                            )
                        ],
                        title=f"LingBot Rolling Summary: {model.model_label} ({len(processed_clips)}/{len(manifest_rows)})",
                    )
                )
                next_report_threshold += cfg.report_every

            all_done = True
            for worker in workers:
                return_code = worker.process.poll()
                if return_code is None:
                    all_done = False
                elif return_code != 0 and (worker.gpu_id, return_code) not in failed_workers:
                    failed_workers.append((worker.gpu_id, return_code))

            if all_done:
                break
            time.sleep(cfg.poll_seconds)

        _scan_new_outputs(
            cfg=cfg,
            physics_cfg=physics_cfg,
            model_root=model_root,
            workers=workers,
            row_by_clip=row_by_clip,
            processed_clips=processed_clips,
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
        )
        _write_model_rollups(
            model_root=model_root,
            cfg=cfg,
            model=model,
            manifest_count=len(manifest_rows),
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
            index_by_clip=index_by_clip,
        )
    finally:
        for worker in workers:
            worker.log_handle.close()

    if failed_workers:
        details = ", ".join(f"gpu={gpu} rc={rc}" for gpu, rc in failed_workers)
        raise RuntimeError(f"LingBot workers failed for {model.model_label}: {details}")

    print(
        format_lingbot_progress_summary(
            [
                build_progress_row(
                    model_label=model.model_label,
                    processed_count=len(processed_clips),
                    total_count=len(manifest_rows),
                    metrics_rows=metrics_rows,
                    physics_rows=physics_rows,
                )
            ],
            title=f"LingBot Rolling Summary: {model.model_label} (final)",
        )
    )
    return build_progress_row(
        model_label=model.model_label,
        processed_count=len(processed_clips),
        total_count=len(manifest_rows),
        metrics_rows=metrics_rows,
        physics_rows=physics_rows,
    )


def main() -> None:
    args = parse_args()
    cfg = _build_config(args)
    path_cfg = resolve_path_config(args, env_file=args.env_file or None)

    _require_existing_path("manifest_path", cfg.manifest_path)
    _require_existing_path("dataset_dir", cfg.dataset_dir)
    _require_existing_path("base_model_dir", cfg.base_model_dir)
    _require_existing_path("lingbot_code_dir", path_cfg.lingbot_code_dir)
    _require_existing_path("finetune_code_dir", path_cfg.finetune_code_dir)
    _require_existing_path("physics_config", cfg.physics_config)
    if cfg.models in {"both", "stage1"}:
        _require_existing_path("stage1_ckpt_dir", cfg.stage1_ckpt_dir)

    manifest_rows = read_csv_rows(cfg.manifest_path)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {cfg.manifest_path}")
    ensure_dir(cfg.val_inf_root)

    physics_cfg = PhysicsIQConfig.from_yaml(cfg.physics_config)
    final_rows: list[dict[str, Any]] = []
    for model in _build_models(cfg):
        final_rows.append(
            run_model_fullval(
                cfg=cfg,
                physics_cfg=physics_cfg,
                path_cfg=path_cfg,
                model=model,
                manifest_rows=manifest_rows,
            )
        )

    print(
        format_lingbot_progress_summary(
            final_rows,
            title="LingBot Full-Val Final Summary",
        )
    )


if __name__ == "__main__":
    main()
