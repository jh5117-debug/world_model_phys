#!/usr/bin/env python3
"""Run managed validation jobs for TRD snapshot checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.io import ensure_dir, read_json, read_yaml, write_json, write_yaml
from physical_consistency.common.summary_tables import format_videophy2_summary


def _parse_env_file(path: Path) -> dict[str, str]:
    values = os.environ.copy()
    if not path.exists():
        return values
    pattern = re.compile(r"\$\{([^}]+)\}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        value = pattern.sub(lambda match: values.get(match.group(1), match.group(0)), value)
        values[key] = value
    return values


def _default_result_root() -> Path:
    typo_root = REPO_ROOT / "tets_result"
    if typo_root.exists():
        return typo_root
    return REPO_ROOT / "test_result"


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch_(\d+)$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**9, path.name)


def _discover_checkpoints(checkpoint_root: Path, names: list[str]) -> list[Path]:
    if names:
        checkpoints = [checkpoint_root / name if not Path(name).is_absolute() else Path(name) for name in names]
    else:
        checkpoints = sorted(checkpoint_root.glob("epoch_*"), key=_checkpoint_sort_key)
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    return checkpoints


def _gpu_list_to_items(gpu_list: str) -> list[str]:
    return [item.strip() for item in gpu_list.split(",") if item.strip()]


def _target_gpu_pids(gpus: list[str]) -> dict[str, list[str]]:
    if not gpus:
        return {}
    try:
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
        app_query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    uuid_to_index: dict[str, str] = {}
    for line in gpu_query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            uuid_to_index[parts[1]] = parts[0]
    busy: dict[str, list[str]] = {gpu: [] for gpu in gpus}
    for line in app_query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            index = uuid_to_index.get(parts[0])
            if index in busy:
                busy[index].append(parts[1])
    return {gpu: pids for gpu, pids in busy.items() if pids}


def _run_command(command: list[str], *, env: dict[str, str], log_path: Path, cwd: Path) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return process.wait()


def _write_managed_eval_config(
    *,
    source_config: Path,
    output_config: Path,
    checkpoint_path: Path,
    result_dir: Path,
    experiment_name: str,
    gpu_count: int,
    run_fid_fvd: bool,
    run_action_control: bool,
) -> dict[str, Any]:
    cfg = read_yaml(source_config)
    cfg["experiment_name"] = experiment_name
    cfg["output_root"] = str(result_dir.resolve())
    cfg["ft_ckpt_dir"] = str(checkpoint_path.resolve())
    cfg["num_gpus"] = gpu_count
    cfg["ulysses_size"] = max(1, min(int(cfg.get("ulysses_size", gpu_count) or gpu_count), gpu_count))
    cfg["run_fid_fvd"] = run_fid_fvd
    cfg["run_action_control"] = run_action_control
    write_yaml(output_config, cfg)
    return cfg


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _score_column(row: dict[str, str]) -> str:
    if "score" in row:
        return "score"
    for key in row:
        if key.lower().endswith("score"):
            return key
    raise KeyError(f"No score column found in VideoPhy-2 row: {sorted(row)}")


def _video_path_column(row: dict[str, str]) -> str | None:
    for key in ("videopath", "video_path", "video", "path"):
        if key in row:
            return key
    for key in row:
        if "video" in key.lower() and "path" in key.lower():
            return key
    return None


def _video_key(path_or_name: str, fallback: str) -> str:
    if not path_or_name:
        return fallback
    name = Path(path_or_name).name
    return name.removesuffix("_gen.mp4").removesuffix(".mp4")


def _read_videophy_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            video_col = _video_path_column(row)
            score_col = _score_column(row)
            video_path = row.get(video_col, "") if video_col else ""
            rows.append(
                {
                    "index": str(idx),
                    "video": _video_key(video_path, f"row_{idx:04d}"),
                    "video_path": video_path,
                    "score": row.get(score_col, ""),
                }
            )
        return rows


def _write_merged_videophy_scores(*, seed_dir: Path, output_csv: Path) -> bool:
    sa_csv = seed_dir / "output_sa.csv"
    pc_csv = seed_dir / "output_pc.csv"
    if not sa_csv.exists() or not pc_csv.exists():
        return False

    sa_rows = _read_videophy_scores(sa_csv)
    pc_rows = _read_videophy_scores(pc_csv)
    merged_rows: list[dict[str, str]] = []
    for idx, (sa_row, pc_row) in enumerate(zip(sa_rows, pc_rows)):
        video = sa_row["video"] if not sa_row["video"].startswith("row_") else pc_row["video"]
        video_path = sa_row["video_path"] or pc_row["video_path"]
        sa_score = float(sa_row["score"])
        pc_score = float(pc_row["score"])
        merged_rows.append(
            {
                "index": str(idx),
                "video": video,
                "video_path": video_path,
                "sa_score": f"{sa_score:.4f}",
                "pc_score": f"{pc_score:.4f}",
                "joint_ge_4": "1" if sa_score >= 4.0 and pc_score >= 4.0 else "0",
            }
        )

    ensure_dir(output_csv.parent)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["index", "video", "video_path", "sa_score", "pc_score", "joint_ge_4"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return True


def _materialize_clean_outputs(
    *,
    result_root: Path,
    result_dir: Path,
    timestamp: str,
    checkpoint_label: str,
    experiment_name: str,
    seeds: list[int],
) -> list[str]:
    """Copy only the artifacts humans inspect: generated mp4s and per-video scores."""
    clean_dirs = [
        result_dir / "clean",
        result_root / "clean_results" / timestamp / checkpoint_label,
    ]
    materialized: list[str] = []
    for clean_dir in clean_dirs:
        videos_dir = ensure_dir(clean_dir / "videos")
        for seed in seeds:
            source_videos = (
                result_dir
                / "runs"
                / "eval"
                / experiment_name
                / f"seed_{seed}"
                / "csgo_metrics"
                / "videos"
            )
            if source_videos.exists():
                for video in sorted(source_videos.glob("*.mp4")):
                    shutil.copy2(video, videos_dir / video.name)

            seed_dir = (
                result_dir
                / "runs"
                / "eval"
                / "videophy2"
                / experiment_name
                / f"seed_{seed}"
            )
            _write_merged_videophy_scores(
                seed_dir=seed_dir,
                output_csv=clean_dir / f"videophy2_scores_seed_{seed}.csv",
            )

        _copy_if_exists(result_dir / "videophy2_summary.md", clean_dir / "videophy2_summary.md")
        materialized.append(str(clean_dir.resolve()))
    return materialized


def _run_one_checkpoint(
    *,
    checkpoint_path: Path,
    result_root: Path,
    timestamp: str,
    env_file: Path,
    gpu_list: str,
    videophy_gpu: str,
    videophy_python: str,
    run_fid_fvd: bool,
    run_action_control: bool,
    dry_run: bool,
) -> dict[str, Any]:
    validation_export = checkpoint_path / "validation_export"
    source_eval_config = validation_export / "eval_trd_snapshot.yaml"
    if not source_eval_config.exists():
        raise FileNotFoundError(f"Missing validation config: {source_eval_config}")

    checkpoint_root_name = checkpoint_path.parent.name
    label = f"{timestamp}__{checkpoint_root_name}__{checkpoint_path.name}"
    result_dir = ensure_dir(result_root / label)
    config_dir = ensure_dir(result_dir / "configs")
    logs_dir = ensure_dir(result_dir / "logs")
    _copy_if_exists(source_eval_config, config_dir / "eval_trd_snapshot.original.yaml")
    _copy_if_exists(validation_export / "validation_request.json", config_dir / "validation_request.original.json")
    _copy_if_exists(validation_export / "run_validation.sh", config_dir / "run_validation.original.sh")

    gpu_items = _gpu_list_to_items(gpu_list)
    experiment_name = f"{checkpoint_root_name}_{checkpoint_path.name}_{timestamp}"
    managed_config = config_dir / "eval_trd_snapshot.managed.yaml"
    cfg = _write_managed_eval_config(
        source_config=source_eval_config,
        output_config=managed_config,
        checkpoint_path=checkpoint_path,
        result_dir=result_dir,
        experiment_name=experiment_name,
        gpu_count=len(gpu_items),
        run_fid_fvd=run_fid_fvd,
        run_action_control=run_action_control,
    )

    common_env = os.environ.copy()
    common_env["PYTHONPATH"] = f"{SRC_ROOT}:{common_env.get('PYTHONPATH', '')}"
    common_env["PATH"] = f"{Path(sys.executable).parent}:{common_env.get('PATH', '')}"
    common_env["CUDA_VISIBLE_DEVICES"] = gpu_list
    common_env["GPU_LIST"] = gpu_list
    common_env["TOKENIZERS_PARALLELISM"] = "false"

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_label": checkpoint_path.name,
        "result_dir": str(result_dir.resolve()),
        "experiment_name": experiment_name,
        "env_file": str(env_file.resolve()),
        "train_python": sys.executable,
        "train_bin_dir": str(Path(sys.executable).parent),
        "videophy_python": videophy_python,
        "gpu_list": gpu_list,
        "videophy_gpu": videophy_gpu,
        "sample_steps": cfg.get("sample_steps"),
        "manifest_path": cfg.get("manifest_path"),
        "run_fid_fvd": cfg.get("run_fid_fvd"),
        "run_action_control": cfg.get("run_action_control"),
        "status": "planned" if dry_run else "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(result_dir / "metadata.json", metadata)

    csgo_cmd = [
        sys.executable,
        "-m",
        "physical_consistency.cli.run_csgo_metrics",
        "--config",
        str(managed_config),
        "--env_file",
        str(env_file),
        "--ft_ckpt_dir",
        str(checkpoint_path.resolve()),
        "--experiment_name",
        experiment_name,
        "--output_root",
        str(result_dir.resolve()),
        "--num_gpus",
        str(len(gpu_items)),
        "--ulysses_size",
        str(cfg["ulysses_size"]),
    ]
    seeds = [int(seed) for seed in cfg.get("seed_list", [42])]
    videophy_cmds = []
    for seed in seeds:
        videophy_cmds.append(
            [
                videophy_python,
                "-m",
                "physical_consistency.cli.run_videophy2",
                "--config",
                str(REPO_ROOT / "configs" / "videophy2_eval.yaml"),
                "--env_file",
                str(env_file),
                "--experiment_name",
                experiment_name,
                "--manifest_csv",
                str(cfg["manifest_path"]),
                "--generated_root",
                str(result_dir / "runs" / "eval" / experiment_name),
                "--output_root",
                str(result_dir.resolve()),
                "--seed",
                str(seed),
            ]
        )
    summary_cmd = [
        videophy_python,
        "-m",
        "physical_consistency.cli.run_videophy2",
        "--config",
        str(REPO_ROOT / "configs" / "videophy2_eval.yaml"),
        "--env_file",
        str(env_file),
        "--experiment_name",
        experiment_name,
        "--summary_only",
        "--output_root",
        str(result_dir.resolve()),
    ]
    metadata["commands"] = {
        "csgo_metrics": csgo_cmd,
        "videophy2": videophy_cmds,
        "videophy2_summary": summary_cmd,
    }
    write_json(result_dir / "metadata.json", metadata)
    (result_dir / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        f"CUDA_VISIBLE_DEVICES={gpu_list} {' '.join(csgo_cmd)}\n"
        + "\n".join(f"CUDA_VISIBLE_DEVICES={videophy_gpu} {' '.join(cmd)}" for cmd in videophy_cmds)
        + f"\nCUDA_VISIBLE_DEVICES={videophy_gpu} {' '.join(summary_cmd)}\n",
        encoding="utf-8",
    )

    if dry_run:
        metadata["status"] = "dry_run"
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(result_dir / "metadata.json", metadata)
        return metadata

    csgo_status = _run_command(csgo_cmd, env=common_env, log_path=logs_dir / "01_csgo_metrics.log", cwd=REPO_ROOT)
    if csgo_status != 0:
        metadata["status"] = "failed_csgo_metrics"
        metadata["returncode"] = csgo_status
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(result_dir / "metadata.json", metadata)
        return metadata

    videophy_env = common_env.copy()
    videophy_env["CUDA_VISIBLE_DEVICES"] = videophy_gpu
    for idx, cmd in enumerate(videophy_cmds, start=1):
        status = _run_command(cmd, env=videophy_env, log_path=logs_dir / f"02_videophy2_seed_{seeds[idx - 1]}.log", cwd=REPO_ROOT)
        if status != 0:
            metadata["status"] = "failed_videophy2"
            metadata["returncode"] = status
            metadata["failed_seed"] = seeds[idx - 1]
            metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(result_dir / "metadata.json", metadata)
            return metadata

    summary_status = _run_command(summary_cmd, env=videophy_env, log_path=logs_dir / "03_videophy2_summary.log", cwd=REPO_ROOT)
    metadata["summary_path"] = str(
        result_dir / "runs" / "eval" / "videophy2" / experiment_name / "summary.json"
    )
    if summary_status == 0 and Path(metadata["summary_path"]).exists():
        summary = read_json(metadata["summary_path"])
        rendered = format_videophy2_summary(summary, title=f"Lingbot_VideoREPA: {checkpoint_path.name}")
        (result_dir / "videophy2_summary.md").write_text(rendered + "\n", encoding="utf-8")
        metadata["videophy2_summary"] = summary
        metadata["clean_result_dirs"] = _materialize_clean_outputs(
            result_root=result_root,
            result_dir=result_dir,
            timestamp=timestamp,
            checkpoint_label=checkpoint_path.name,
            experiment_name=experiment_name,
            seeds=seeds,
        )
        metadata["status"] = "ok"
    else:
        metadata["status"] = "failed_videophy2_summary"
        metadata["returncode"] = summary_status
    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(result_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run managed TRD snapshot validation jobs.")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints/exp_stage1_epoch2_trd_v1")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--result-root", type=str, default="")
    parser.add_argument("--env-file", type=str, default="configs/path_config_cluster.env")
    parser.add_argument("--gpu-list", type=str, default=os.environ.get("GPU_LIST", ""))
    parser.add_argument("--videophy-gpu", type=str, default="")
    parser.add_argument("--videophy-python", type=str, default="")
    parser.add_argument("--timestamp", type=str, default="")
    parser.add_argument("--require-free-gpus", action="store_true")
    parser.add_argument("--run-fid-fvd", action="store_true")
    parser.add_argument("--run-action-control", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    result_root = Path(args.result_root) if args.result_root else _default_result_root()
    if not result_root.is_absolute():
        result_root = REPO_ROOT / result_root
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = REPO_ROOT / env_file
    env_values = _parse_env_file(env_file)
    gpu_list = args.gpu_list or env_values.get("GPU_LIST", "")
    gpu_items = _gpu_list_to_items(gpu_list)
    if not gpu_items:
        raise SystemExit("--gpu-list is required, for example --gpu-list 0,1,2,3")
    if args.require_free_gpus:
        busy = _target_gpu_pids(gpu_items)
        if busy:
            raise SystemExit(f"Refusing to start because target GPUs are busy: {busy}")
    videophy_gpu = args.videophy_gpu or gpu_items[0]
    videophy_python = args.videophy_python or env_values.get("VIDEOPHY2_PYTHON", "")
    if not videophy_python:
        raise SystemExit("--videophy-python is required or VIDEOPHY2_PYTHON must be set in the env file")
    if not Path(videophy_python).exists():
        raise SystemExit(f"VideoPhy-2 python not found: {videophy_python}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints = _discover_checkpoints(checkpoint_root, args.checkpoints)
    ensure_dir(result_root)
    batch_manifest = {
        "timestamp": timestamp,
        "checkpoint_root": str(checkpoint_root.resolve()),
        "result_root": str(result_root.resolve()),
        "checkpoints": [path.name for path in checkpoints],
        "gpu_list": gpu_list,
        "videophy_gpu": videophy_gpu,
        "videophy_python": videophy_python,
        "dry_run": args.dry_run,
        "jobs": [],
    }
    write_json(result_root / f"batch_{timestamp}.json", batch_manifest)

    for checkpoint in checkpoints:
        print(f"[RUN] checkpoint={checkpoint.name} result_root={result_root}")
        result = _run_one_checkpoint(
            checkpoint_path=checkpoint,
            result_root=result_root,
            timestamp=timestamp,
            env_file=env_file,
            gpu_list=gpu_list,
            videophy_gpu=videophy_gpu,
            videophy_python=videophy_python,
            run_fid_fvd=args.run_fid_fvd,
            run_action_control=args.run_action_control,
            dry_run=args.dry_run,
        )
        batch_manifest["jobs"].append(result)
        write_json(result_root / f"batch_{timestamp}.json", batch_manifest)
        if result.get("status") not in {"ok", "dry_run"}:
            raise SystemExit(f"Validation failed for {checkpoint.name}: {result.get('status')}")
    print(f"[DONE] Managed TRD snapshot tests written under {result_root}")


if __name__ == "__main__":
    main()
