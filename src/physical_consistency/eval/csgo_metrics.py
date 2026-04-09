"""Wrappers around the existing Stage-1 evaluation scripts."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from physical_consistency.common.defaults import DEFAULT_SEED_LIST
from physical_consistency.common.io import (
    ensure_dir,
    read_csv_rows,
    read_json,
    read_yaml,
    resolve_project_path,
    write_json,
)
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.subprocess_utils import run_command
from physical_consistency.eval.checkpoint_bundle import materialize_eval_checkpoint_bundle
from physical_consistency.datasets.manifest_builder import hash_manifest, materialize_dataset_view

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CSGOEvalConfig:
    """Evaluation settings for a single experiment."""

    experiment_name: str
    split: str
    manifest_path: str
    frame_num: int
    sample_steps: int
    guide_scale: float
    height: int
    width: int
    num_gpus: int
    ulysses_size: int
    run_fid_fvd: bool
    run_action_control: bool
    run_videophy2: bool
    base_model_dir: str
    ft_ckpt_dir: str
    output_root: str
    seed_list: list[int]
    stage1_ckpt_dir: str = ""
    allow_stage1_fallback: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CSGOEvalConfig":
        """Load config from YAML."""
        payload = read_yaml(path)
        payload.setdefault("seed_list", DEFAULT_SEED_LIST)
        for key in [
            "manifest_path",
            "base_model_dir",
            "ft_ckpt_dir",
            "output_root",
            "stage1_ckpt_dir",
        ]:
            if key in payload:
                payload[key] = resolve_project_path(payload[key])
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the CSGO metrics wrapper."""
    parser = argparse.ArgumentParser(description="Run existing CSGO metric scripts.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--ft_ckpt_dir", type=str, default="")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--lingbot_code_dir", type=str, default="")
    parser.add_argument("--finetune_code_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--ulysses_size", type=int, default=0)
    return parser.parse_args()


def run_single_seed_eval(
    cfg: CSGOEvalConfig,
    *,
    finetune_code_dir: str,
    dataset_dir: str,
    output_root: str,
    seed: int,
    base_model_dir: str,
    lingbot_code_dir: str,
    ft_ckpt_dir: str = "",
) -> dict[str, str]:
    """Run batch eval, FID/FVD, and action-control for one seed."""
    _require_existing_path("finetune_code_dir", finetune_code_dir)
    _require_existing_path("dataset_dir", dataset_dir)
    _require_existing_path("base_model_dir", base_model_dir)
    _require_existing_path("lingbot_code_dir", lingbot_code_dir)

    manifest_hash = hash_manifest(cfg.manifest_path)
    view_dir = Path(output_root) / "cache" / "dataset_views" / f"{cfg.experiment_name}_{manifest_hash}"
    materialize_dataset_view(dataset_dir, cfg.manifest_path, view_dir)
    effective_ft_ckpt_dir = ""
    if ft_ckpt_dir:
        effective_ft_ckpt_dir = str(
            materialize_eval_checkpoint_bundle(
                ft_ckpt_dir=ft_ckpt_dir,
                output_root=output_root,
                experiment_name=cfg.experiment_name,
                stage1_ckpt_dir=cfg.stage1_ckpt_dir,
                allow_stage1_fallback=cfg.allow_stage1_fallback,
            )
        )

    run_dir = Path(output_root) / "runs" / "eval" / cfg.experiment_name / f"seed_{seed}"
    metrics_dir = run_dir / "csgo_metrics"
    fid_dir = run_dir / "fid_fvd"
    action_dir = run_dir / "action_control"
    logs_dir = Path(output_root) / "logs" / "eval"
    ensure_dir(metrics_dir)
    ensure_dir(fid_dir)
    ensure_dir(action_dir)
    ensure_dir(logs_dir)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    common_env = {"TOKENIZERS_PARALLELISM": "false"}
    common_env["CUDA_VISIBLE_DEVICES"] = (
        visible_devices
        if visible_devices
        else ",".join(str(idx) for idx in range(cfg.num_gpus))
    )

    batch_cmd = [
        "torchrun",
        f"--nproc_per_node={cfg.num_gpus}",
        "eval_batch.py",
        "--ckpt_dir",
        base_model_dir,
        "--lingbot_code_dir",
        lingbot_code_dir,
        "--dataset_dir",
        str(view_dir),
        "--output_dir",
        str(metrics_dir),
        "--split",
        cfg.split,
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
        str(seed),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(cfg.ulysses_size),
        "--skip_existing",
    ]
    if effective_ft_ckpt_dir:
        batch_cmd.extend(["--ft_ckpt_dir", effective_ft_ckpt_dir])

    run_command(
        batch_cmd,
        cwd=finetune_code_dir,
        env=common_env,
        log_path=logs_dir / f"{cfg.experiment_name}_seed{seed}_eval_batch.log",
    )

    if cfg.run_fid_fvd:
        run_command(
            [
                "python",
                "eval_fid_fvd.py",
                "--gen_dir",
                str(metrics_dir / "videos"),
                "--real_dir",
                str(Path(view_dir) / "val" / "clips"),
                "--output_dir",
                str(fid_dir),
                "--device",
                "cuda:0",
            ],
            cwd=finetune_code_dir,
            env=common_env,
            log_path=logs_dir / f"{cfg.experiment_name}_seed{seed}_fid_fvd.log",
        )

    if cfg.run_action_control:
        run_command(
            [
                "python",
                "eval_action_control.py",
                "--gen_dir",
                str(metrics_dir / "videos"),
                "--clip_dir",
                str(Path(view_dir) / "val" / "clips"),
                "--output_dir",
                str(action_dir),
            ],
            cwd=finetune_code_dir,
            env=common_env,
            log_path=logs_dir / f"{cfg.experiment_name}_seed{seed}_action.log",
        )

    return {
        "run_dir": str(run_dir),
        "metrics_report": str(metrics_dir / "eval_report.json"),
        "fid_fvd_report": str(fid_dir / "eval_fid_fvd_report.json"),
        "action_report": str(action_dir / "eval_action_control_report.json"),
        "video_dir": str(metrics_dir / "videos"),
        "manifest_view_dir": str(view_dir),
    }


def _require_existing_path(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} is empty. Pass --{label} or set it in the env file.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def summarize_eval_suite(run_root: str | Path) -> dict:
    """Aggregate metrics across seed runs."""
    root = Path(run_root)
    seed_summaries: list[dict] = []
    for seed_dir in sorted(root.glob("seed_*")):
        metrics_path = seed_dir / "csgo_metrics" / "eval_report.json"
        if not metrics_path.exists():
            continue
        summary = {
            "seed": int(seed_dir.name.split("_")[-1]),
            "csgo_metrics": read_json(metrics_path),
        }
        fid_path = seed_dir / "fid_fvd" / "eval_fid_fvd_report.json"
        action_path = seed_dir / "action_control" / "eval_action_control_report.json"
        if fid_path.exists():
            summary["fid_fvd"] = read_json(fid_path)
        if action_path.exists():
            summary["action_control"] = read_json(action_path)
        seed_summaries.append(summary)

    aggregate = {"seeds": seed_summaries, "means": {}}
    metric_collector: dict[str, list[float]] = {}
    for summary in seed_summaries:
        csgo_agg = summary["csgo_metrics"].get("aggregate_metrics", {})
        for metric_name in ["psnr", "ssim", "lpips", "gen_time_s"]:
            maybe = csgo_agg.get(metric_name, {}).get("mean")
            if maybe is not None:
                metric_collector.setdefault(metric_name, []).append(float(maybe))
        fid = summary.get("fid_fvd", {})
        for metric_name in ["fid", "fvd"]:
            maybe = fid.get(metric_name)
            if maybe is not None:
                metric_collector.setdefault(metric_name, []).append(float(maybe))
        action = summary.get("action_control", {})
        for metric_name in ["flow_direction_accuracy", "trajectory_consistency", "turn_direction_accuracy"]:
            maybe = action.get("aggregate_metrics", {}).get(metric_name, {}).get("mean")
            if maybe is not None:
                metric_collector.setdefault(metric_name, []).append(float(maybe))

    for key, values in metric_collector.items():
        aggregate["means"][key] = {
            "mean": sum(values) / len(values),
            "count": len(values),
        }
    return aggregate


def write_summary(output_root: str | Path, experiment_name: str) -> Path:
    """Write the aggregated summary JSON for one experiment."""
    run_root = Path(output_root) / "runs" / "eval" / experiment_name
    summary = summarize_eval_suite(run_root)
    summary_path = run_root / "summary.json"
    write_json(summary_path, summary)
    return summary_path


def main() -> None:
    """CLI main entry."""
    args = parse_args()
    cfg = CSGOEvalConfig.from_yaml(args.config)
    if args.seed >= 0:
        cfg.seed_list = [args.seed]
    if args.ft_ckpt_dir:
        cfg.ft_ckpt_dir = resolve_project_path(args.ft_ckpt_dir)
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.base_model_dir:
        cfg.base_model_dir = resolve_project_path(args.base_model_dir)
    if args.stage1_ckpt_dir:
        cfg.stage1_ckpt_dir = resolve_project_path(args.stage1_ckpt_dir)
    if args.output_root:
        cfg.output_root = resolve_project_path(args.output_root)
    if args.manifest_path:
        cfg.manifest_path = resolve_project_path(args.manifest_path)
    if args.num_gpus > 0:
        cfg.num_gpus = args.num_gpus
    if args.ulysses_size > 0:
        cfg.ulysses_size = args.ulysses_size

    path_cfg = resolve_path_config(args, env_file=args.env_file or None)
    cfg.base_model_dir = cfg.base_model_dir or path_cfg.base_model_dir
    cfg.output_root = cfg.output_root or path_cfg.output_root
    cfg.stage1_ckpt_dir = cfg.stage1_ckpt_dir or path_cfg.stage1_ckpt_dir
    _require_existing_path("manifest_path", cfg.manifest_path)
    if cfg.ft_ckpt_dir:
        _require_existing_path("ft_ckpt_dir", cfg.ft_ckpt_dir)
    if cfg.stage1_ckpt_dir:
        _require_existing_path("stage1_ckpt_dir", cfg.stage1_ckpt_dir)
    for seed in cfg.seed_list:
        run_single_seed_eval(
            cfg,
            finetune_code_dir=path_cfg.finetune_code_dir,
            dataset_dir=path_cfg.dataset_dir,
            output_root=cfg.output_root,
            seed=seed,
            base_model_dir=cfg.base_model_dir,
            lingbot_code_dir=path_cfg.lingbot_code_dir,
            ft_ckpt_dir=cfg.ft_ckpt_dir,
        )
    summary_path = write_summary(cfg.output_root, cfg.experiment_name)
    print(json.dumps(read_json(summary_path), indent=2))


if __name__ == "__main__":
    main()
