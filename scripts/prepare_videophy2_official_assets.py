#!/usr/bin/env python
"""Prepare official VideoPhy-2 benchmark assets for local evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BOOTSTRAP_PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from physical_consistency.common.defaults import PROJECT_ROOT
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.eval.videophy2_official_assets import (
    read_csv_header,
    write_official_video_subset_manifest,
    write_official_prompt_manifests,
)

LOGGER = logging.getLogger("prepare_videophy2_official_assets")


def _copy_from_hf(
    *,
    repo_id: str,
    repo_type: str,
    filename: str,
    dest_path: Path,
) -> Path:
    source = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
        )
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest_path)
    return dest_path


def _download_checkpoint(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="videophysics/videophy_2_auto",
        repo_type="model",
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official VideoPhy-2 benchmark assets.")
    parser.add_argument(
        "--env_file",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "path_config_cluster.env"),
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "videophy2_official" / "raw"),
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests"),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Override the resolved VideoPhy-2 checkpoint directory.",
    )
    parser.add_argument(
        "--download_checkpoint",
        action="store_true",
        help="Download the official VideoPhy-2 AutoEval checkpoint into checkpoint_dir.",
    )
    parser.add_argument(
        "--skip_train_csv",
        action="store_true",
        help="Skip downloading the small training metadata CSV.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="",
        help="Where to write the preparation summary JSON.",
    )
    parser.add_argument(
        "--download_test_videos",
        action="store_true",
        help="Download a local official test-video subset for AutoEval smoke testing.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "videophy2_official" / "videos" / "test_subset"),
        help="Destination directory for downloaded official test videos.",
    )
    parser.add_argument(
        "--video_manifest_path",
        type=str,
        default="",
        help="Where to write the official test-video subset manifest.",
    )
    parser.add_argument(
        "--video_prompt_mode",
        type=str,
        default="original",
        choices=["original", "upsampled"],
        help="Which prompt field to expose as `prompt` in the official video manifest.",
    )
    parser.add_argument(
        "--video_hard_only",
        action="store_true",
        help="Restrict the downloaded official test-video subset to `is_hard=1` rows.",
    )
    parser.add_argument(
        "--video_limit",
        type=int,
        default=0,
        help="Optional cap on the number of official test videos to materialize.",
    )
    parser.add_argument(
        "--video_models",
        type=str,
        default="",
        help="Optional comma-separated list of official model_name values to keep.",
    )
    parser.add_argument(
        "--overwrite_videos",
        action="store_true",
        help="Re-download official test videos even if local files already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    path_cfg = resolve_path_config(env_file=args.env_file)
    raw_dir = Path(args.raw_dir)
    manifest_dir = Path(args.manifest_dir)
    checkpoint_dir = Path(args.checkpoint_dir or path_cfg.videophy2_ckpt_dir)
    summary_path = Path(args.summary_path) if args.summary_path else raw_dir.parent / "summary.json"
    video_root = Path(args.video_root)
    video_manifest_path = (
        Path(args.video_manifest_path)
        if args.video_manifest_path
        else manifest_dir
        / (
            "videophy2_test_official_"
            f"{args.video_prompt_mode}"
            f"{'_hard' if args.video_hard_only else ''}"
            "_videos.csv"
        )
    )
    selected_models = [item.strip() for item in args.video_models.split(",") if item.strip()]

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preparing official VideoPhy-2 assets under %s", raw_dir.parent)

    test_readme = _copy_from_hf(
        repo_id="videophysics/videophy2_test",
        repo_type="dataset",
        filename="README.md",
        dest_path=raw_dir / "videophy2_test_README.md",
    )
    test_csv = _copy_from_hf(
        repo_id="videophysics/videophy2_test",
        repo_type="dataset",
        filename="videophy2_test.csv",
        dest_path=raw_dir / "videophy2_test.csv",
    )
    upsampled_test_csv = _copy_from_hf(
        repo_id="videophysics/videophy2_upsampled_prompts",
        repo_type="dataset",
        filename="prompt-upsampled-test.csv",
        dest_path=raw_dir / "videophy2_prompt_upsampled_test.csv",
    )

    train_readme = None
    train_csv = None
    upsampled_train_csv = None
    if not args.skip_train_csv:
        train_readme = _copy_from_hf(
            repo_id="videophysics/videophy2_train",
            repo_type="dataset",
            filename="README.md",
            dest_path=raw_dir / "videophy2_train_README.md",
        )
        train_csv = _copy_from_hf(
            repo_id="videophysics/videophy2_train",
            repo_type="dataset",
            filename="videophy2_training.csv",
            dest_path=raw_dir / "videophy2_training.csv",
        )
        upsampled_train_csv = _copy_from_hf(
            repo_id="videophysics/videophy2_upsampled_prompts",
            repo_type="dataset",
            filename="prompt-upsampled-train.csv",
            dest_path=raw_dir / "videophy2_prompt_upsampled_train.csv",
        )

    manifests = write_official_prompt_manifests(
        official_test_csv=test_csv,
        manifest_dir=manifest_dir,
        official_upsampled_test_csv=upsampled_test_csv,
    )

    video_subset = None
    if args.download_test_videos or args.video_manifest_path or args.video_limit > 0 or selected_models or args.video_hard_only:
        LOGGER.info(
            "Preparing official test-video subset manifest: prompt_mode=%s hard_only=%s limit=%s models=%s",
            args.video_prompt_mode,
            args.video_hard_only,
            args.video_limit,
            ",".join(selected_models) if selected_models else "<all>",
        )
        video_subset = write_official_video_subset_manifest(
            official_test_csv=test_csv,
            manifest_path=video_manifest_path,
            video_root=video_root,
            prompt_mode=args.video_prompt_mode,
            hard_only=args.video_hard_only,
            model_names=selected_models,
            limit=args.video_limit,
            download_videos=args.download_test_videos,
            overwrite_videos=args.overwrite_videos,
        )

    checkpoint_present = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    if args.download_checkpoint:
        LOGGER.info("Downloading official VideoPhy-2 AutoEval checkpoint to %s", checkpoint_dir)
        _download_checkpoint(checkpoint_dir)
        checkpoint_present = True

    summary = {
        "raw_dir": str(raw_dir),
        "manifest_dir": str(manifest_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_present": checkpoint_present,
        "official_repo_expected": str(path_cfg.videophy_repo_dir),
        "files": {
            "test_readme": str(test_readme),
            "test_csv": str(test_csv),
            "upsampled_test_csv": str(upsampled_test_csv),
            "train_readme": str(train_readme) if train_readme else "",
            "train_csv": str(train_csv) if train_csv else "",
            "upsampled_train_csv": str(upsampled_train_csv) if upsampled_train_csv else "",
        },
        "test_csv_header": read_csv_header(test_csv),
        "manifests": {
            "original_manifest": str(manifests.original_manifest),
            "original_hard_manifest": str(manifests.original_hard_manifest),
            "upsampled_manifest": str(manifests.upsampled_manifest),
            "upsampled_hard_manifest": str(manifests.upsampled_hard_manifest),
            "original_count": manifests.original_count,
            "original_hard_count": manifests.original_hard_count,
            "upsampled_count": manifests.upsampled_count,
            "upsampled_hard_count": manifests.upsampled_hard_count,
        },
        "official_video_subset": (
            {
                "manifest_path": str(video_subset.manifest_path),
                "video_root": str(video_subset.video_root),
                "count": video_subset.count,
                "prompt_mode": video_subset.prompt_mode,
                "hard_only": video_subset.hard_only,
                "model_names": video_subset.model_names,
                "downloaded_videos": args.download_test_videos,
            }
            if video_subset
            else {}
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Official test CSV -> %s", test_csv)
    LOGGER.info(
        "Prompt manifests: original=%d hard=%d upsampled=%d upsampled_hard=%d",
        manifests.original_count,
        manifests.original_hard_count,
        manifests.upsampled_count,
        manifests.upsampled_hard_count,
    )
    if video_subset:
        LOGGER.info(
            "Official test-video subset: count=%d manifest=%s video_root=%s",
            video_subset.count,
            video_subset.manifest_path,
            video_subset.video_root,
        )
    if checkpoint_present:
        LOGGER.info("VideoPhy-2 AutoEval checkpoint is present at %s", checkpoint_dir)
    else:
        LOGGER.warning(
            "VideoPhy-2 AutoEval checkpoint is still missing at %s. "
            "Re-run with --download_checkpoint or point --checkpoint_dir to an existing snapshot.",
            checkpoint_dir,
        )
    LOGGER.info("Summary JSON -> %s", summary_path)


if __name__ == "__main__":
    main()
