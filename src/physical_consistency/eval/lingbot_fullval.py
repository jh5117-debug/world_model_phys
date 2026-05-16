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
import torch

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
from physical_consistency.eval.video_utils import write_labeled_side_by_side_video


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
    stage1_label: str = "LingBot-Stage1"
    run_fvd: bool = False
    fvd_device: str = "cuda:0"
    lpips_device: str = "auto"


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
    parser.add_argument("--stage1_label", type=str, default="LingBot-Stage1")
    parser.add_argument("--run_fvd", action="store_true")
    parser.add_argument("--fvd_device", type=str, default="cuda:0")
    parser.add_argument("--lpips_device", type=str, default="auto")
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
    dataset_dir_arg = getattr(args, "dataset_dir", "")
    manifest_path_arg = getattr(args, "manifest_path", "")
    output_root_arg = getattr(args, "output_root", "")
    base_model_dir_arg = getattr(args, "base_model_dir", "")
    stage1_ckpt_dir_arg = getattr(args, "stage1_ckpt_dir", "")
    val_inf_root_arg = getattr(args, "val_inf_root", "")
    physics_config_arg = getattr(args, "physics_config", "")

    dataset_dir = resolve_project_path(dataset_dir_arg) if dataset_dir_arg else path_cfg.dataset_dir
    manifest_path = (
        resolve_project_path(manifest_path_arg)
        if manifest_path_arg
        else str(Path(dataset_dir) / "metadata_val.csv")
    )
    output_root = resolve_project_path(output_root_arg) if output_root_arg else path_cfg.output_root
    base_model_dir = (
        resolve_project_path(base_model_dir_arg) if base_model_dir_arg else path_cfg.base_model_dir
    )
    stage1_ckpt_dir = (
        resolve_project_path(stage1_ckpt_dir_arg)
        if stage1_ckpt_dir_arg
        else path_cfg.stage1_ckpt_dir
    )
    val_inf_root = (
        resolve_project_path(val_inf_root_arg)
        if val_inf_root_arg
        else str(Path(dataset_dir) / "val_inf_result")
    )
    physics_config = (
        resolve_project_path(physics_config_arg)
        if physics_config_arg
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
        stage1_label=args.stage1_label,
        run_fvd=bool(args.run_fvd),
        fvd_device=args.fvd_device,
        lpips_device=args.lpips_device,
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
    mean_ssim = _compute_mean(metrics_rows, "ssim")
    mean_lpips = _compute_mean(metrics_rows, "lpips")
    return {
        "Model": model_label,
        "Processed": processed_count,
        "Total": total_count,
        "PMF ↑": mean_score if mean_score is not None else "",
        "PSNR ↑": mean_psnr if mean_psnr is not None else "",
        "SSIM ↑": mean_ssim if mean_ssim is not None else "",
        "LPIPS ↓": mean_lpips if mean_lpips is not None else "",
        "FVD ↓": "",
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


def compute_video_ssim(
    *,
    reference_videopath: str | Path,
    candidate_videopath: str | Path,
    frame_num: int,
    height: int,
    width: int,
) -> float:
    """Compute mean frame SSIM for a generated/reference pair."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")
    gt_frames = _read_video_frames(reference_videopath, max_frames=frame_num, height=height, width=width)
    gen_frames = _read_video_frames(candidate_videopath, max_frames=frame_num, height=height, width=width)
    min_frames = min(len(gt_frames), len(gen_frames))
    scores = [
        float(ssim(gen_frames[idx], gt_frames[idx], channel_axis=2, data_range=255))
        for idx in range(min_frames)
    ]
    return float(sum(scores) / len(scores)) if scores else float("nan")


class LPIPSMeter:
    """Lazy LPIPS evaluator shared across generated clips."""

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model: Any | None = None
        self._unavailable = False

    def _load(self) -> Any | None:
        if self._unavailable:
            return None
        if self._model is not None:
            return self._model
        try:
            import lpips
        except ImportError:
            self._unavailable = True
            return None
        self._model = lpips.LPIPS(net="alex").to(self.device).eval()
        return self._model

    def compute(
        self,
        *,
        reference_videopath: str | Path,
        candidate_videopath: str | Path,
        frame_num: int,
        height: int,
        width: int,
    ) -> float:
        model = self._load()
        if model is None:
            return float("nan")
        gt_frames = _read_video_frames(reference_videopath, max_frames=frame_num, height=height, width=width)
        gen_frames = _read_video_frames(candidate_videopath, max_frames=frame_num, height=height, width=width)
        min_frames = min(len(gt_frames), len(gen_frames))
        scores: list[float] = []
        with torch.no_grad():
            for idx in range(min_frames):
                gen = torch.from_numpy(gen_frames[idx]).permute(2, 0, 1).float() / 127.5 - 1.0
                ref = torch.from_numpy(gt_frames[idx]).permute(2, 0, 1).float() / 127.5 - 1.0
                gen = gen.unsqueeze(0).to(self.device)
                ref = ref.unsqueeze(0).to(self.device)
                scores.append(float(model(gen, ref).item()))
        return float(sum(scores) / len(scores)) if scores else float("nan")


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
    for key in ["psnr", "ssim", "lpips"]:
        values = [_safe_float(row.get(key)) for row in valid]
        filtered = [value for value in values if value is not None]
        if filtered:
            mean_value = sum(filtered) / len(filtered)
            report["aggregate_metrics"][key] = {
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
                "ssim",
                "lpips",
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
                model_label=cfg.stage1_label,
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


def _prepare_single_gpu_worker_env(base_env: dict[str, str], gpu_id: str) -> dict[str, str]:
    """Drop inherited distributed-launch state for single-GPU eval workers."""
    env = dict(base_env)
    for key in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_DYNAMO_BACKEND",
        "DEEPSPEED_CONFIG_FILE",
    ]:
        env.pop(key, None)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    return env


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
        env = _prepare_single_gpu_worker_env(os.environ.copy(), gpu_id)
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
    lpips_meter: LPIPSMeter,
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
    ssim = compute_video_ssim(
        reference_videopath=reference_path,
        candidate_videopath=worker_video_path,
        frame_num=cfg.frame_num,
        height=cfg.height,
        width=cfg.width,
    )
    lpips_value = lpips_meter.compute(
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
        "ssim": round(ssim, 4) if math.isfinite(ssim) else "",
        "lpips": round(lpips_value, 4) if math.isfinite(lpips_value) else "",
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
    lpips_meter: LPIPSMeter,
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
                    lpips_meter=lpips_meter,
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
        _run_fvd_eval_if_requested(cfg=cfg, path_cfg=path_cfg, model_root=model_root, model=model)
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
    lpips_meter = LPIPSMeter(cfg.lpips_device)
    next_report_threshold = ((len(processed_clips) // cfg.report_every) + 1) * cfg.report_every
    failed_workers: list[tuple[str, int]] = []

    try:
        while True:
            _scan_new_outputs(
                cfg=cfg,
                physics_cfg=physics_cfg,
                lpips_meter=lpips_meter,
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
            lpips_meter=lpips_meter,
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

    if len(processed_clips) < len(manifest_rows):
        missing_count = len(manifest_rows) - len(processed_clips)
        log_paths = ", ".join(str(worker.log_path) for worker in workers)
        raise RuntimeError(
            f"LingBot workers finished for {model.model_label}, but only "
            f"{len(processed_clips)}/{len(manifest_rows)} clips produced scoreable videos "
            f"({missing_count} missing). Check worker logs: {log_paths}"
        )

    _run_fvd_eval_if_requested(cfg=cfg, path_cfg=path_cfg, model_root=model_root, model=model)

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


def _run_fvd_eval_if_requested(
    *,
    cfg: FullvalConfig,
    path_cfg: Any,
    model_root: Path,
    model: ModelSpec,
) -> dict[str, Any]:
    """Run the external FID/FVD script when available."""
    report_path = model_root / "fid_fvd" / "eval_fid_fvd_report.json"
    if not cfg.run_fvd:
        return read_json(report_path) if report_path.exists() else {}
    if report_path.exists():
        return read_json(report_path)

    script_path = Path(path_cfg.finetune_code_dir) / "eval_fid_fvd.py"
    if not script_path.exists():
        error = {
            "error": f"eval_fid_fvd.py not found under finetune_code_dir: {script_path}",
            "model_label": model.model_label,
        }
        write_json(model_root / "fid_fvd" / "eval_fid_fvd_error.json", error)
        return error

    view_dir = Path(cfg.output_root) / "cache" / "fullval_fvd_views" / model.subdir_name
    if view_dir.exists():
        shutil.rmtree(view_dir)
    materialize_dataset_view(cfg.dataset_dir, cfg.manifest_path, view_dir)
    real_dir = view_dir / "val" / "clips"
    output_dir = ensure_dir(model_root / "fid_fvd")
    log_path = Path(cfg.output_root) / "logs" / "fullval" / f"{model.subdir_name}_fid_fvd.log"
    ensure_dir(log_path.parent)
    command = [
        sys.executable,
        "eval_fid_fvd.py",
        "--gen_dir",
        str(model_root / "videos"),
        "--real_dir",
        str(real_dir),
        "--output_dir",
        str(output_dir),
        "--device",
        cfg.fvd_device,
    ]
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            cwd=path_cfg.finetune_code_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    return read_json(report_path) if report_path.exists() else {}


def _metric_mean_from_eval_report(model_root: Path, key: str) -> float | None:
    report_path = model_root / "eval_report.json"
    if not report_path.exists():
        return None
    report = read_json(report_path)
    return _safe_float(report.get("aggregate_metrics", {}).get(key, {}).get("mean"))


def _pmf_from_physics_summary(model_root: Path) -> float | None:
    summary_path = model_root / "physics_iq_summary.json"
    if not summary_path.exists():
        return None
    summary = read_json(summary_path)
    return _safe_float(summary.get("means", {}).get("physics_iq_style_score", {}).get("mean"))


def _fvd_from_report(model_root: Path) -> float | None:
    report_path = model_root / "fid_fvd" / "eval_fid_fvd_report.json"
    if not report_path.exists():
        return None
    report = read_json(report_path)
    for key_path in [
        ("fvd",),
        ("aggregate_metrics", "fvd", "mean"),
        ("metrics", "fvd"),
    ]:
        payload: Any = report
        for key in key_path:
            if not isinstance(payload, dict) or key not in payload:
                payload = None
                break
            payload = payload[key]
        value = _safe_float(payload)
        if value is not None:
            return value
    return None


def _round_or_blank(value: float | None, digits: int = 4) -> float | str:
    if value is None or not math.isfinite(value):
        return ""
    return round(value, digits)


def _model_metrics_row(cfg: FullvalConfig, model: ModelSpec) -> dict[str, Any]:
    model_root = Path(cfg.val_inf_root) / model.subdir_name
    metrics_path = model_root / "metrics.csv"
    processed = len(read_csv_rows(metrics_path)) if metrics_path.exists() else 0
    return {
        "model_label": model.model_label,
        "subdir_name": model.subdir_name,
        "processed": processed,
        "pmf": _pmf_from_physics_summary(model_root),
        "psnr": _metric_mean_from_eval_report(model_root, "psnr"),
        "ssim": _metric_mean_from_eval_report(model_root, "ssim"),
        "lpips": _metric_mean_from_eval_report(model_root, "lpips"),
        "fvd": _fvd_from_report(model_root),
    }


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Method", "PMF ↑", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "FVD ↓"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row["Method"]),
            str(row["PMF ↑"]),
            str(row["PSNR ↑"]),
            str(row["SSIM ↑"]),
            str(row["LPIPS ↓"]),
            str(row["FVD ↓"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_dir_name(label: str) -> str:
    return "base_vs_stage2" if "stage2" in label.lower() else "base_vs_stage1"


def _write_base_vs_stage_videos(cfg: FullvalConfig, manifest_rows: list[dict[str, str]]) -> Path | None:
    if cfg.models != "both":
        return None
    base_dir = Path(cfg.val_inf_root) / "lingbotbase" / "videos"
    stage_dir = Path(cfg.val_inf_root) / "lingbotstage1" / "videos"
    if not base_dir.exists() or not stage_dir.exists():
        return None
    comparison_name = _comparison_dir_name(cfg.stage1_label)
    out_dir = ensure_dir(Path(cfg.val_inf_root) / "qualitative" / comparison_name)
    for row in manifest_rows:
        clip_name = _clip_name(row)
        video_name = _candidate_video_name(cfg, clip_name)
        left = base_dir / video_name
        right = stage_dir / video_name
        if not left.exists() or not right.exists():
            continue
        write_labeled_side_by_side_video(
            left_videopath=left,
            right_videopath=right,
            output_path=out_dir / f"{clip_name}_{comparison_name}.mp4",
            left_label="LingBot-base",
            right_label=cfg.stage1_label,
            max_frames=cfg.frame_num,
            height=cfg.height,
            width=cfg.width,
        )
    return out_dir


def _write_final_summary(
    *,
    cfg: FullvalConfig,
    models: list[ModelSpec],
    manifest_rows: list[dict[str, str]],
) -> Path:
    quantitative_dir = ensure_dir(Path(cfg.val_inf_root) / "quantitative")
    qualitative_dir = _write_base_vs_stage_videos(cfg, manifest_rows)
    metric_rows = [_model_metrics_row(cfg, model) for model in models]
    table_rows = [
        {
            "Method": row["model_label"],
            "PMF ↑": _round_or_blank(row["pmf"]),
            "PSNR ↑": _round_or_blank(row["psnr"]),
            "SSIM ↑": _round_or_blank(row["ssim"]),
            "LPIPS ↓": _round_or_blank(row["lpips"]),
            "FVD ↓": _round_or_blank(row["fvd"]),
        }
        for row in metric_rows
    ]
    write_csv_rows(
        quantitative_dir / "metrics_summary.csv",
        table_rows,
        ["Method", "PMF ↑", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "FVD ↓"],
    )
    _write_markdown_table(quantitative_dir / "metrics_summary.md", table_rows)
    summary = {
        "metrics_mode": "physinone",
        "directions": {
            "pmf": "higher",
            "psnr": "higher",
            "ssim": "higher",
            "lpips": "lower",
            "fvd": "lower",
        },
        "manifest_path": cfg.manifest_path,
        "dataset_dir": cfg.dataset_dir,
        "val_inf_root": cfg.val_inf_root,
        "qualitative_dir": str(qualitative_dir) if qualitative_dir is not None else "",
        "quantitative_dir": str(quantitative_dir),
        "rows": metric_rows,
    }
    summary_path = quantitative_dir / "metrics_summary.json"
    write_json(summary_path, summary)
    write_json(Path(cfg.val_inf_root) / "summary.json", summary)
    return summary_path


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
    models = _build_models(cfg)
    for model in models:
        final_rows.append(
            run_model_fullval(
                cfg=cfg,
                physics_cfg=physics_cfg,
                path_cfg=path_cfg,
                model=model,
                manifest_rows=manifest_rows,
            )
        )
    summary_path = _write_final_summary(cfg=cfg, models=models, manifest_rows=manifest_rows)
    summary_md_path = Path(summary_path).with_name("metrics_summary.md")
    if summary_md_path.exists():
        print("LingBot Full-Val Final Summary")
        print(summary_md_path.read_text(encoding="utf-8").strip())
    print(f"PhysInOne metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
