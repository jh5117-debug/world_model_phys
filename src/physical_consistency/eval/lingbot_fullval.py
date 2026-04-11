"""Chunked full-val LingBot generation plus rolling Physics-IQ/PSNR summaries."""

from __future__ import annotations

import argparse
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physical_consistency.common.io import (
    ensure_dir,
    read_csv_rows,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.common.summary_tables import format_lingbot_progress_summary
from physical_consistency.datasets.manifest_builder import materialize_dataset_view
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.eval.physics_iq import (
    run_physics_iq_for_seed,
)


@dataclass(slots=True)
class FullvalConfig:
    """Configuration for the chunked full-val pipeline."""

    manifest_path: str
    dataset_dir: str
    output_root: str
    base_model_dir: str
    stage1_ckpt_dir: str
    val_inf_root: str
    frame_num: int = 81
    sample_steps: int = 70
    guide_scale: float = 5.0
    height: int = 480
    width: int = 832
    num_gpus: int = 8
    ulysses_size: int = 8
    control_type: str = "act"
    seed: int = 0
    chunk_size: int = 10
    models: str = "both"
    video_filename: str = "video.mp4"
    video_suffix: str = "_gen.mp4"


@dataclass(slots=True)
class ModelSpec:
    """One LingBot model variant to run."""

    model_label: str
    subdir_name: str
    experiment_prefix: str
    base_model_dir: str
    ft_ckpt_dir: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunked full-val LingBot base/stage1 generation with rolling Physics-IQ summaries."
    )
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    parser.add_argument("--val_inf_root", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=0)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=70)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--control_type", type=str, default="act", choices=["act", "cam"])
    parser.add_argument("--models", type=str, default="both", choices=["both", "base", "stage1"])
    parser.add_argument("--video_filename", type=str, default="video.mp4")
    parser.add_argument("--video_suffix", type=str, default="_gen.mp4")
    return parser.parse_args()


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

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_count = 0
    if visible_devices:
        visible_count = len([item for item in visible_devices.split(",") if item.strip()])
    num_gpus = args.num_gpus if args.num_gpus > 0 else max(1, visible_count or 8)
    ulysses_size = args.ulysses_size if args.ulysses_size > 0 else num_gpus

    return FullvalConfig(
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        output_root=output_root,
        base_model_dir=base_model_dir,
        stage1_ckpt_dir=stage1_ckpt_dir,
        val_inf_root=val_inf_root,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        height=args.height,
        width=args.width,
        num_gpus=num_gpus,
        ulysses_size=ulysses_size,
        control_type=args.control_type,
        seed=args.seed,
        chunk_size=args.chunk_size,
        models=args.models,
        video_filename=args.video_filename,
        video_suffix=args.video_suffix,
    )


def chunk_rows(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    """Split manifest rows into sequential chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [rows[idx : idx + chunk_size] for idx in range(0, len(rows), chunk_size)]


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


def _field_union(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    fieldnames: list[str] = []
    if preferred:
        fieldnames.extend(preferred)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _build_eval_report(
    rows: list[dict[str, Any]],
    *,
    manifest_count: int,
    cfg: FullvalConfig,
    model: ModelSpec,
) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error") and _safe_float(row.get("psnr")) is not None]
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

    for metric_name in ["psnr", "ssim", "lpips", "gen_time_s"]:
        values = [
            _safe_float(row.get(metric_name))
            for row in valid
            if _safe_float(row.get(metric_name)) is not None
        ]
        filtered = [value for value in values if value is not None]
        if filtered:
            mean_value = sum(filtered) / len(filtered)
            report["aggregate_metrics"][metric_name] = {
                "mean": round(mean_value, 4),
                "std": round((sum((value - mean_value) ** 2 for value in filtered) / len(filtered)) ** 0.5, 4),
                "min": round(min(filtered), 4),
                "max": round(max(filtered), 4),
            }
    return report


def _build_physics_rollup_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    if not rows:
        return {"seeds": [], "means": {}}

    summary = summarize_physics_iq_outputs_from_rows(rows)
    return {
        "seeds": [
            {
                "seed": seed,
                "count": summary["count"],
                "means": summary["means"],
            }
        ],
        "means": summary["means"],
    }


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


def _write_model_rollups(
    *,
    model_root: Path,
    cfg: FullvalConfig,
    model: ModelSpec,
    manifest_count: int,
    metrics_rows: list[dict[str, Any]],
    physics_rows: list[dict[str, Any]],
) -> None:
    metrics_fieldnames = _field_union(metrics_rows)
    if metrics_fieldnames:
        write_csv_rows(model_root / "metrics.csv", metrics_rows, metrics_fieldnames)
    write_json(
        model_root / "eval_report.json",
        _build_eval_report(rows=metrics_rows, manifest_count=manifest_count, cfg=cfg, model=model),
    )

    physics_fieldnames = [
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
    write_csv_rows(model_root / "physics_iq_output_pairs.csv", physics_rows, physics_fieldnames)
    write_json(
        model_root / "physics_iq_summary.json",
        _build_physics_rollup_summary(physics_rows, seed=cfg.seed),
    )


def _load_completed_rows(model_root: Path, *, seed: int) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    completed_chunks: list[int] = []
    metrics_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    chunks_root = model_root / "chunks"
    if not chunks_root.exists():
        return completed_chunks, metrics_rows, physics_rows

    for chunk_dir in sorted(chunks_root.glob("chunk_*")):
        done_path = chunk_dir / "completed.ok"
        if not done_path.exists():
            continue
        chunk_idx = int(chunk_dir.name.split("_")[-1])
        chunk_metrics = chunk_dir / "metrics.csv"
        chunk_physics = chunk_dir / "physics_iq" / f"seed_{seed}" / "output_pairs.csv"
        if not chunk_metrics.exists() or not chunk_physics.exists():
            raise FileNotFoundError(
                f"Completed marker exists but chunk outputs are incomplete: {chunk_dir}"
            )
        completed_chunks.append(chunk_idx)
        metrics_rows.extend(read_csv_rows(chunk_metrics))
        physics_rows.extend(read_csv_rows(chunk_physics))
    return completed_chunks, metrics_rows, physics_rows


def _resolve_visible_devices(num_gpus: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible
    return ",".join(str(idx) for idx in range(num_gpus))


def _require_existing_path(label: str, value: str) -> None:
    path = Path(value)
    if not value or not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def run_model_fullval(
    *,
    cfg: FullvalConfig,
    model: ModelSpec,
    path_cfg: Any,
    manifest_rows: list[dict[str, str]],
) -> dict[str, Any]:
    model_root = ensure_dir(Path(cfg.val_inf_root) / model.subdir_name)
    ensure_dir(model_root / "videos")
    manifests_root = ensure_dir(model_root / "manifests")
    chunks_root = ensure_dir(model_root / "chunks")
    cache_root = ensure_dir(Path(cfg.output_root) / "cache" / "fullval_dataset_views" / model.subdir_name)
    logs_root = ensure_dir(Path(cfg.output_root) / "logs" / "fullval")
    write_csv_rows(model_root / "run_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    chunks = chunk_rows(manifest_rows, cfg.chunk_size)
    completed_chunks, metrics_rows, physics_rows = _load_completed_rows(model_root, seed=cfg.seed)
    completed_set = set(completed_chunks)

    if completed_chunks:
        _write_model_rollups(
            model_root=model_root,
            cfg=cfg,
            model=model,
            manifest_count=len(manifest_rows),
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
        )
        print(
            format_lingbot_progress_summary(
                [
                    build_progress_row(
                        model_label=model.model_label,
                        processed_count=len(metrics_rows),
                        total_count=len(manifest_rows),
                        metrics_rows=metrics_rows,
                        physics_rows=physics_rows,
                    )
                ],
                title=f"LingBot Rolling Summary: {model.model_label} (resume)",
            )
        )

    effective_ft_ckpt_dir = ""
    if model.ft_ckpt_dir:
        effective_ft_ckpt_dir = str(
            materialize_eval_checkpoint_bundle(
                ft_ckpt_dir=model.ft_ckpt_dir,
                output_root=cfg.output_root,
                experiment_name=f"{model.experiment_prefix}_fullval",
                stage1_ckpt_dir=cfg.stage1_ckpt_dir,
                allow_stage1_fallback=False,
            )
        )

    common_env = {
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": _resolve_visible_devices(cfg.num_gpus),
    }

    for chunk_idx, chunk in enumerate(chunks):
        chunk_name = f"chunk_{chunk_idx:04d}"
        if chunk_idx in completed_set:
            continue

        chunk_dir = ensure_dir(chunks_root / chunk_name)
        chunk_manifest = manifests_root / f"{chunk_name}.csv"
        if not chunk_manifest.exists():
            write_csv_rows(chunk_manifest, chunk, list(chunk[0].keys()))

        view_dir = cache_root / chunk_name
        if view_dir.exists():
            shutil.rmtree(view_dir)
        materialize_dataset_view(cfg.dataset_dir, chunk_manifest, view_dir)

        batch_log = logs_root / f"{model.subdir_name}_{chunk_name}_eval_batch.log"
        batch_cmd = [
            "torchrun",
            f"--nproc_per_node={max(1, cfg.num_gpus)}",
            "eval_batch.py",
            "--ckpt_dir",
            model.base_model_dir,
            "--lingbot_code_dir",
            path_cfg.lingbot_code_dir,
            "--dataset_dir",
            str(view_dir),
            "--output_dir",
            str(model_root),
            "--split",
            "val",
            "--max_samples",
            "0",
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
            str(max(1, min(cfg.ulysses_size, cfg.num_gpus))),
        ]
        if cfg.num_gpus > 1:
            batch_cmd.extend(["--dit_fsdp", "--t5_fsdp"])
        if effective_ft_ckpt_dir:
            batch_cmd.extend(["--ft_ckpt_dir", effective_ft_ckpt_dir])

        run_command(
            batch_cmd,
            cwd=path_cfg.finetune_code_dir,
            env=common_env,
            log_path=batch_log,
        )

        root_metrics = model_root / "metrics.csv"
        root_report = model_root / "eval_report.json"
        if not root_metrics.exists():
            raise FileNotFoundError(f"Chunk metrics.csv missing after eval_batch: {root_metrics}")

        _copy_if_exists(root_metrics, chunk_dir / "metrics.csv")
        _copy_if_exists(root_report, chunk_dir / "eval_report.json")
        chunk_metrics_rows = read_csv_rows(root_metrics)

        physics_seed_dir = chunk_dir / "physics_iq" / f"seed_{cfg.seed}"
        run_physics_iq_for_seed(
            manifest_csv=chunk_manifest,
            reference_source_root=cfg.dataset_dir,
            candidate_source_root=model_root / "videos",
            output_dir=physics_seed_dir,
            seed=cfg.seed,
            compare_seconds=5.0,
            sample_frames=40,
            resize_divisor=4,
            mask_threshold=10,
            reference_source_mode="dataset_clip",
            candidate_source_mode="generated",
            video_filename=cfg.video_filename,
            video_suffix=cfg.video_suffix,
            max_samples=0,
        )
        chunk_physics_rows = read_csv_rows(physics_seed_dir / "output_pairs.csv")

        metrics_rows.extend(chunk_metrics_rows)
        physics_rows.extend(chunk_physics_rows)
        _write_model_rollups(
            model_root=model_root,
            cfg=cfg,
            model=model,
            manifest_count=len(manifest_rows),
            metrics_rows=metrics_rows,
            physics_rows=physics_rows,
        )
        (chunk_dir / "completed.ok").write_text("ok\n", encoding="utf-8")

        processed_count = len(metrics_rows)
        print(
            format_lingbot_progress_summary(
                [
                    build_progress_row(
                        model_label=model.model_label,
                        processed_count=processed_count,
                        total_count=len(manifest_rows),
                        metrics_rows=metrics_rows,
                        physics_rows=physics_rows,
                    )
                ],
                title=f"LingBot Rolling Summary: {model.model_label} ({processed_count}/{len(manifest_rows)})",
            )
        )

    return build_progress_row(
        model_label=model.model_label,
        processed_count=len(metrics_rows),
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
    if cfg.models in {"both", "stage1"}:
        _require_existing_path("stage1_ckpt_dir", cfg.stage1_ckpt_dir)

    manifest_rows = read_csv_rows(cfg.manifest_path)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {cfg.manifest_path}")
    ensure_dir(cfg.val_inf_root)

    model_specs: list[ModelSpec] = []
    if cfg.models in {"both", "base"}:
        model_specs.append(
            ModelSpec(
                model_label="LingBot-base",
                subdir_name="lingbotbase",
                experiment_prefix="lingbotbase",
                base_model_dir=cfg.base_model_dir,
            )
        )
    if cfg.models in {"both", "stage1"}:
        model_specs.append(
            ModelSpec(
                model_label="LingBot-Stage1",
                subdir_name="lingbotstage1",
                experiment_prefix="lingbotstage1",
                base_model_dir=cfg.base_model_dir,
                ft_ckpt_dir=cfg.stage1_ckpt_dir,
            )
        )

    final_rows: list[dict[str, Any]] = []
    for model in model_specs:
        final_rows.append(
            run_model_fullval(
                cfg=cfg,
                model=model,
                path_cfg=path_cfg,
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
