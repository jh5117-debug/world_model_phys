"""Generation-only LingBot pipeline for multi-GPU sharded inference."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

from physical_consistency.common.io import ensure_dir, read_csv_rows, write_csv_rows
from physical_consistency.common.path_config import resolve_path_config
from physical_consistency.common.summary_tables import format_lingbot_generation_summary
from physical_consistency.eval.lingbot_fullval import (
    _build_config,
    _build_models,
    _clip_name,
    _require_existing_path,
    _spawn_workers,
    shard_rows,
)
from physical_consistency.eval.video_utils import (
    VideoValidationError,
    validate_video_readable,
    write_side_by_side_video,
)

REFERENCE_VIDEO_COLUMNS = (
    "reference_videopath",
    "reference_video",
    "gt_videopath",
    "gt_video",
    "videopath",
    "video_path",
    "video",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sharded single-GPU LingBot generation without Physics-IQ/PSNR scoring."
    )
    parser.add_argument("--env_file", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--base_model_dir", type=str, default="")
    parser.add_argument("--stage1_ckpt_dir", type=str, default="")
    parser.add_argument("--val_inf_root", type=str, default="")
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


def build_generation_progress_row(*, model_label: str, processed_count: int, total_count: int) -> dict[str, Any]:
    return {
        "Model": model_label,
        "Processed": processed_count,
        "Total": total_count,
    }


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _manifest_path_string(path: Path) -> str:
    if not str(path):
        return ""
    return str(path.resolve())


def _reference_video_candidates(row: dict[str, str], cfg: Any) -> list[Path]:
    candidates: list[Path] = []
    dataset_dir = Path(cfg.dataset_dir)

    def add_raw(raw: str) -> None:
        if not raw:
            return
        path = Path(raw)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(dataset_dir / path)
            candidates.append(path)

    for column in REFERENCE_VIDEO_COLUMNS:
        add_raw(str(row.get(column, "")).strip())

    clip_path = str(row.get("clip_path", "")).strip()
    if clip_path:
        clip = Path(clip_path)
        if clip.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            candidates.append(dataset_dir / clip)
        candidates.append(dataset_dir / clip / cfg.video_filename)

    return _unique_paths(candidates)


def _resolve_reference_video(row: dict[str, str], cfg: Any) -> Path:
    candidates = _reference_video_candidates(row, cfg)
    for candidate in candidates:
        try:
            validate_video_readable(candidate, min_frames=1)
        except VideoValidationError:
            continue
        return candidate
    return candidates[0] if candidates else Path("")


def _generated_manifest_row(
    *,
    cfg: Any,
    row: dict[str, str],
    clip_name: str,
    candidate_videopath: Path,
    comparison_path: Path,
) -> dict[str, str]:
    reference_path = _resolve_reference_video(row, cfg)
    comparison_videopath = ""
    if str(reference_path):
        try:
            write_side_by_side_video(
                reference_videopath=reference_path,
                candidate_videopath=candidate_videopath,
                output_path=comparison_path,
                max_frames=int(cfg.frame_num),
                height=int(cfg.height),
                width=int(cfg.width),
            )
            comparison_videopath = str(comparison_path)
        except VideoValidationError as exc:
            print(
                f"[WARN] Skipping side-by-side preview for {clip_name}: {exc}",
                flush=True,
            )

    return {
        "clip_name": clip_name,
        "clip_path": row["clip_path"],
        "prompt": row.get("prompt", ""),
        "reference_videopath": _manifest_path_string(reference_path),
        "candidate_videopath": _manifest_path_string(candidate_videopath),
        "comparison_videopath": _manifest_path_string(Path(comparison_videopath)) if comparison_videopath else "",
    }


def _load_existing_generated(
    model_root: Path,
    *,
    cfg: Any,
    row_by_clip: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], set[str]]:
    min_frames = min(8, max(1, int(cfg.frame_num)))
    generated_manifest = model_root / "generated_videos.csv"
    if generated_manifest.exists():
        rows = []
        processed = set()
        for row in read_csv_rows(generated_manifest):
            clip_name = str(row.get("clip_name", ""))
            candidate = row.get("candidate_videopath", "")
            if not clip_name or not candidate:
                continue
            try:
                validate_video_readable(candidate, min_frames=min_frames)
            except VideoValidationError:
                continue
            row["candidate_videopath"] = _manifest_path_string(Path(candidate))
            comparison = row.get("comparison_videopath", "")
            if comparison:
                row["comparison_videopath"] = _manifest_path_string(Path(comparison))
            reference = row.get("reference_videopath", "")
            if reference:
                row["reference_videopath"] = _manifest_path_string(Path(reference))
            rows.append(row)
            processed.add(clip_name)
        return rows, processed

    videos_dir = model_root / "videos"
    rows: list[dict[str, str]] = []
    processed: set[str] = set()
    if videos_dir.exists():
        for video_path in sorted(videos_dir.glob("*_gen.mp4")):
            clip_name = video_path.name[: -len("_gen.mp4")]
            row = row_by_clip.get(clip_name)
            if row is None:
                continue
            try:
                validate_video_readable(video_path, min_frames=min_frames)
            except VideoValidationError:
                continue
            comparison_path = model_root / "comparisons" / video_path.name
            rows.append(
                _generated_manifest_row(
                    cfg=cfg,
                    row=row,
                    clip_name=clip_name,
                    candidate_videopath=video_path,
                    comparison_path=comparison_path,
                )
            )
            processed.add(clip_name)
    return rows, processed


def _scan_new_outputs(
    *,
    cfg: Any,
    model_root: Path,
    workers: list[Any],
    row_by_clip: dict[str, dict[str, str]],
    processed_clips: set[str],
    generated_rows: list[dict[str, str]],
) -> int:
    new_count = 0
    common_videos_dir = ensure_dir(model_root / "videos")
    comparisons_dir = ensure_dir(model_root / "comparisons")
    min_frames = min(8, max(1, int(cfg.frame_num)))
    for worker in workers:
        worker_videos_dir = worker.output_dir / "videos"
        for video_path in sorted(worker_videos_dir.glob(f"*{cfg.video_suffix}")):
            clip_name = video_path.name[: -len(cfg.video_suffix)]
            if clip_name in processed_clips:
                continue
            row = row_by_clip.get(clip_name)
            if row is None:
                continue
            try:
                validate_video_readable(video_path, min_frames=min_frames)
            except VideoValidationError:
                continue
            common_video_path = common_videos_dir / video_path.name
            shutil.copy2(video_path, common_video_path)
            comparison_path = comparisons_dir / video_path.name
            generated_rows.append(
                _generated_manifest_row(
                    cfg=cfg,
                    row=row,
                    clip_name=clip_name,
                    candidate_videopath=common_video_path,
                    comparison_path=comparison_path,
                )
            )
            processed_clips.add(clip_name)
            new_count += 1
    return new_count


def _write_generated_rollup(model_root: Path, generated_rows: list[dict[str, str]]) -> None:
    if not generated_rows:
        return
    write_csv_rows(
        model_root / "generated_videos.csv",
        generated_rows,
        [
            "clip_name",
            "clip_path",
            "prompt",
            "reference_videopath",
            "candidate_videopath",
            "comparison_videopath",
        ],
    )


def run_model_generation(*, cfg: Any, path_cfg: Any, model: Any, manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    model_root = ensure_dir(Path(cfg.val_inf_root) / model.subdir_name)
    ensure_dir(model_root / "videos")
    write_csv_rows(model_root / "run_manifest.csv", manifest_rows, list(manifest_rows[0].keys()))

    row_by_clip = {_clip_name(row): row for row in manifest_rows}
    generated_rows, processed_clips = _load_existing_generated(model_root, cfg=cfg, row_by_clip=row_by_clip)

    if processed_clips:
        print(
            format_lingbot_generation_summary(
                [
                    build_generation_progress_row(
                        model_label=model.model_label,
                        processed_count=len(processed_clips),
                        total_count=len(manifest_rows),
                    )
                ],
                title=f"LingBot Generation Summary: {model.model_label} (resume)",
            )
        )

    remaining_rows = [row for row in manifest_rows if _clip_name(row) not in processed_clips]
    if not remaining_rows:
        return build_generation_progress_row(
            model_label=model.model_label,
            processed_count=len(processed_clips),
            total_count=len(manifest_rows),
        )

    shard_payloads = shard_rows(remaining_rows, len(cfg.gpu_list))
    workers = _spawn_workers(cfg=cfg, model=model, path_cfg=path_cfg, shard_payloads=shard_payloads)
    next_report_threshold = ((len(processed_clips) // cfg.report_every) + 1) * cfg.report_every
    failed_workers: list[tuple[str, int]] = []

    try:
        while True:
            _scan_new_outputs(
                cfg=cfg,
                model_root=model_root,
                workers=workers,
                row_by_clip=row_by_clip,
                processed_clips=processed_clips,
                generated_rows=generated_rows,
            )
            _write_generated_rollup(model_root, generated_rows)

            while len(processed_clips) >= next_report_threshold:
                print(
                    format_lingbot_generation_summary(
                        [
                            build_generation_progress_row(
                                model_label=model.model_label,
                                processed_count=len(processed_clips),
                                total_count=len(manifest_rows),
                            )
                        ],
                        title=f"LingBot Generation Summary: {model.model_label} ({len(processed_clips)}/{len(manifest_rows)})",
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
            model_root=model_root,
            workers=workers,
            row_by_clip=row_by_clip,
            processed_clips=processed_clips,
            generated_rows=generated_rows,
        )
        _write_generated_rollup(model_root, generated_rows)
    finally:
        for worker in workers:
            worker.log_handle.close()

    if failed_workers:
        details = ", ".join(f"gpu={gpu} rc={rc}" for gpu, rc in failed_workers)
        raise RuntimeError(f"LingBot workers failed for {model.model_label}: {details}")
    if len(processed_clips) != len(manifest_rows):
        missing = sorted(_clip_name(row) for row in manifest_rows if _clip_name(row) not in processed_clips)
        raise RuntimeError(
            f"LingBot generation incomplete for {model.model_label}: "
            f"{len(processed_clips)}/{len(manifest_rows)} valid videos. "
            f"Missing/invalid clips: {', '.join(missing[:8])}"
        )

    print(
        format_lingbot_generation_summary(
            [
                build_generation_progress_row(
                    model_label=model.model_label,
                    processed_count=len(processed_clips),
                    total_count=len(manifest_rows),
                )
            ],
            title=f"LingBot Generation Summary: {model.model_label} (final)",
        )
    )
    return build_generation_progress_row(
        model_label=model.model_label,
        processed_count=len(processed_clips),
        total_count=len(manifest_rows),
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

    final_rows = [
        run_model_generation(cfg=cfg, path_cfg=path_cfg, model=model, manifest_rows=manifest_rows)
        for model in _build_models(cfg)
    ]
    print(format_lingbot_generation_summary(final_rows, title="LingBot Generation Final Summary"))


if __name__ == "__main__":
    main()
