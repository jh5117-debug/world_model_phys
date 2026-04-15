"""VideoPhy-2 AutoEval wrappers."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

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
from physical_consistency.common.summary_tables import format_videophy2_summary
from physical_consistency.common.subprocess_utils import run_command

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VideoPhy2Config:
    """Config for running VideoPhy-2 AutoEval."""

    model_name: str
    task_modes: list[str]
    seed_list: list[int]
    batch_size: int
    max_samples: int
    repo_dir: str
    checkpoint_dir: str
    output_root: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VideoPhy2Config":
        payload = read_yaml(path)
        payload.setdefault("seed_list", DEFAULT_SEED_LIST)
        for key in ["repo_dir", "checkpoint_dir", "output_root"]:
            if key in payload:
                payload[key] = resolve_project_path(payload[key])
        return cls(**payload)


def build_videophy2_input_csv(
    *,
    manifest_csv: str | Path,
    video_source_root: str | Path,
    output_csv: str | Path,
    task: str,
    source_mode: str = "generated",
    manifest_video_column: str = "videopath",
    manifest_caption_column: str = "prompt",
    video_filename: str = "video.mp4",
    video_suffix: str = "_gen.mp4",
) -> Path:
    """Build the CSV format expected by official VideoPhy-2 inference."""
    rows = read_csv_rows(manifest_csv)
    source_root = Path(video_source_root)

    output_rows: list[dict[str, str]] = []
    for row in rows:
        clip_path = row.get("clip_path", "")
        clip_name = Path(clip_path).name
        if source_mode == "generated":
            video_path = source_root / f"{clip_name}{video_suffix}"
        elif source_mode == "dataset_clip":
            video_path = source_root / clip_path / video_filename
        elif source_mode in {"manifest_videopath", "manifest_video_column"}:
            raw_video_path = row.get(manifest_video_column, "")
            video_path = Path(raw_video_path)
            if raw_video_path and not video_path.is_absolute():
                video_path = source_root / raw_video_path
        else:
            raise ValueError(f"Unsupported VideoPhy-2 source mode: {source_mode}")
        if not video_path.exists():
            continue
        payload = {"videopath": str(video_path)}
        if task == "sa":
            payload["caption"] = row.get(manifest_caption_column, "")
        output_rows.append(payload)

    if output_rows:
        fieldnames = list(output_rows[0].keys())
    else:
        fieldnames = ["videopath", "caption"] if task == "sa" else ["videopath"]
    write_csv_rows(output_csv, output_rows, fieldnames)
    return Path(output_csv)


def run_videophy2_for_seed(
    *,
    repo_dir: str,
    checkpoint_dir: str,
    manifest_csv: str,
    video_source_root: str,
    output_dir: str,
    batch_size: int,
    seed: int,
    task_modes: list[str],
    source_mode: str = "generated",
    manifest_video_column: str = "videopath",
    manifest_caption_column: str = "prompt",
    video_filename: str = "video.mp4",
    video_suffix: str = "_gen.mp4",
) -> dict[str, str]:
    """Run official VideoPhy-2 inference.py for selected tasks."""
    output_root = ensure_dir(output_dir)
    csv_dir = ensure_dir(output_root / "inputs")
    outputs: dict[str, str] = {}

    for task in task_modes:
        input_csv = build_videophy2_input_csv(
            manifest_csv=manifest_csv,
            video_source_root=video_source_root,
            output_csv=csv_dir / f"seed_{seed}_{task}.csv",
            task=task.lower(),
            source_mode=source_mode,
            manifest_video_column=manifest_video_column,
            manifest_caption_column=manifest_caption_column,
            video_filename=video_filename,
            video_suffix=video_suffix,
        )
        output_csv = output_root / f"output_{task.lower()}.csv"
        cmd = [
            sys.executable,
            "inference.py",
            "--input_csv",
            str(input_csv),
            "--checkpoint",
            checkpoint_dir,
            "--output_csv",
            str(output_csv),
            "--task",
            task.lower(),
            "--batch_size",
            str(batch_size),
        ]
        run_command(
            cmd,
            cwd=Path(repo_dir) / "VIDEOPHY2",
            log_path=output_root / f"videophy2_{task.lower()}.log",
        )
        outputs[task.lower()] = str(output_csv)
    return outputs


def summarize_videophy2_outputs(sa_csv: str | Path, pc_csv: str | Path) -> dict:
    """Compute SA mean, PC mean, and joint score from output CSVs."""
    def _read_scores(path: str | Path) -> list[float]:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [float(row["score"]) for row in reader if row.get("score") not in {"", None}]

    sa_scores = _read_scores(sa_csv)
    pc_scores = _read_scores(pc_csv)
    pair_count = min(len(sa_scores), len(pc_scores))
    joint = 0.0
    if pair_count > 0:
        joint = sum(
            1.0
            for idx in range(pair_count)
            if sa_scores[idx] >= 4.0 and pc_scores[idx] >= 4.0
        ) / pair_count
    return {
        "sa_mean": sum(sa_scores) / len(sa_scores) if sa_scores else 0.0,
        "pc_mean": sum(pc_scores) / len(pc_scores) if pc_scores else 0.0,
        "joint": joint,
        "count": pair_count,
    }


def write_videophy2_summary(output_dir: str | Path) -> Path:
    """Aggregate all seed subdirectories under one experiment root."""
    root = Path(output_dir)
    seed_summaries = []
    for seed_dir in sorted(root.glob("seed_*")):
        sa_csv = seed_dir / "output_sa.csv"
        pc_csv = seed_dir / "output_pc.csv"
        if sa_csv.exists() and pc_csv.exists():
            summary = summarize_videophy2_outputs(sa_csv, pc_csv)
            summary["seed"] = int(seed_dir.name.split("_")[-1])
            seed_summaries.append(summary)

    aggregate = {"seeds": seed_summaries, "means": {}}
    for key in ["sa_mean", "pc_mean", "joint"]:
        values = [item[key] for item in seed_summaries]
        if values:
            aggregate["means"][key] = {
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    summary_path = root / "summary.json"
    write_json(summary_path, aggregate)
    return summary_path


def main() -> None:
    """CLI main."""
    parser = argparse.ArgumentParser(description="Run VideoPhy-2 AutoEval.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, default="")
    parser.add_argument("--generated_root", type=str, default="")
    parser.add_argument(
        "--video_source_mode",
        type=str,
        default="generated",
        choices=["generated", "dataset_clip", "manifest_videopath", "manifest_video_column"],
    )
    parser.add_argument("--video_source_root", type=str, default="")
    parser.add_argument("--manifest_video_column", type=str, default="videopath")
    parser.add_argument("--manifest_caption_column", type=str, default="prompt")
    parser.add_argument("--video_filename", type=str, default="video.mp4")
    parser.add_argument("--video_suffix", type=str, default="_gen.mp4")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--videophy_repo_dir", type=str, default="")
    args = parser.parse_args()

    cfg = VideoPhy2Config.from_yaml(args.config)
    if args.seed >= 0:
        cfg.seed_list = [args.seed]
    path_cfg = resolve_path_config(args, env_file=args.env_file or None)
    repo_dir = (
        resolve_project_path(args.videophy_repo_dir)
        if args.videophy_repo_dir
        else cfg.repo_dir or path_cfg.videophy_repo_dir
    )
    checkpoint_dir = (
        resolve_project_path(args.checkpoint_dir)
        if args.checkpoint_dir
        else cfg.checkpoint_dir or path_cfg.videophy2_ckpt_dir
    )
    output_root = (
        resolve_project_path(args.output_root)
        if args.output_root
        else cfg.output_root or path_cfg.output_root
    )

    base_output = Path(output_root) / "runs" / "eval" / "videophy2" / args.experiment_name
    ensure_dir(base_output)

    if args.summary_only:
        summary_path = write_videophy2_summary(base_output)
        print(
            format_videophy2_summary(
                read_json(summary_path),
                title=f"VideoPhy-2 Summary: {args.experiment_name}",
            )
        )
        return

    if not args.manifest_csv:
        parser.error("--manifest_csv is required unless --summary_only is set")
    if not checkpoint_dir:
        parser.error("VideoPhy-2 checkpoint dir is empty. Pass --checkpoint_dir or set VIDEOPHY2_CKPT_DIR")

    manifest_csv = resolve_project_path(args.manifest_csv)
    if not Path(manifest_csv).exists():
        parser.error(f"Manifest CSV does not exist: {manifest_csv}")
    repo_entry = Path(repo_dir) / "VIDEOPHY2" / "inference.py"
    if not repo_entry.exists():
        parser.error(f"VideoPhy-2 repo not found under: {Path(repo_dir) / 'VIDEOPHY2'}")
    if not Path(checkpoint_dir).exists():
        parser.error(f"VideoPhy-2 checkpoint dir does not exist: {checkpoint_dir}")
    if args.video_source_root:
        video_source_root = resolve_project_path(args.video_source_root)
    elif args.generated_root:
        video_source_root = resolve_project_path(args.generated_root)
    else:
        parser.error("--generated_root or --video_source_root is required unless --summary_only is set")
    if args.video_source_mode != "generated" and not Path(video_source_root).exists():
        parser.error(f"Video source root does not exist: {video_source_root}")

    for seed in cfg.seed_list:
        if args.video_source_mode == "generated":
            source_root = str(
                Path(video_source_root) / f"seed_{seed}" / "csgo_metrics" / "videos"
            )
        else:
            source_root = video_source_root
        run_videophy2_for_seed(
            repo_dir=repo_dir,
            checkpoint_dir=checkpoint_dir,
            manifest_csv=manifest_csv,
            video_source_root=source_root,
            output_dir=str(base_output / f"seed_{seed}"),
            batch_size=cfg.batch_size,
            seed=seed,
            task_modes=cfg.task_modes,
            source_mode=args.video_source_mode,
            manifest_video_column=args.manifest_video_column,
            manifest_caption_column=args.manifest_caption_column,
            video_filename=args.video_filename,
            video_suffix=args.video_suffix,
        )

    summary_path = write_videophy2_summary(base_output)
    print(
        format_videophy2_summary(
            read_json(summary_path),
            title=f"VideoPhy-2 Summary: {args.experiment_name}",
        )
    )


if __name__ == "__main__":
    main()
