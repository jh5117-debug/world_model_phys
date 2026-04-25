#!/usr/bin/env python3
"""Inspect raw CSGO roots and estimate how many 81-frame clips they can provide."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2

import prepprocess_data as prep


EXCLUDED_JSON_SUFFIXES = (
    "_episode_info.json",
    "_video_manifest.json",
    "_game_manifest.json",
    "_seg_colormap.json",
)


def iter_episode_dirs(input_dir: str):
    for split_name in ("train", "test"):
        split_dir = Path(input_dir) / split_name
        if not split_dir.is_dir():
            continue
        for ep_dir in sorted(split_dir.iterdir()):
            if ep_dir.is_dir() and ep_dir.name.startswith("Ep_"):
                yield split_name, ep_dir


def count_raw_layout(input_dirs: list[str]) -> tuple[dict[str, dict], dict[str, int]]:
    per_source = {}
    totals = Counter()

    for input_dir in input_dirs:
        source_name = Path(input_dir).name
        source_id = source_name[:8]
        stats = Counter()
        source_episode_ids = set()
        global_episode_ids = set()

        for split_name, ep_dir in iter_episode_dirs(input_dir):
            del split_name
            source_episode_ids.add((source_id, ep_dir.name))
            global_episode_ids.add(ep_dir.name)
            files = list(ep_dir.iterdir())
            stats["episode_dirs"] += 1
            stats["player_mp4_files"] += sum(1 for p in files if p.suffix == ".mp4")
            stats["episode_info_files"] += sum(1 for p in files if p.name.endswith("_episode_info.json"))
            stats["action_json_files"] += sum(
                1
                for p in files
                if p.suffix == ".json" and not p.name.endswith(EXCLUDED_JSON_SUFFIXES)
            )

            for info_path in sorted(p for p in files if p.name.endswith("_episode_info.json")):
                stem = info_path.name.replace("_episode_info.json", "")
                video_path = ep_dir / f"{stem}.mp4"
                action_path = ep_dir / f"{stem}.json"

                if not video_path.exists():
                    stats["missing_video"] += 1
                    continue
                if not action_path.exists():
                    stats["missing_action"] += 1
                    continue

                try:
                    with open(info_path, "r", encoding="utf-8") as handle:
                        info = json.load(handle)
                except Exception:
                    stats["broken_episode_info"] += 1
                    continue

                if info.get("encountered_error", False):
                    stats["encountered_error"] += 1
                    continue
                stats["discoverable_streams_scan"] += 1

        stats["unique_source_episodes"] = len(source_episode_ids)
        stats["unique_episode_ids_only"] = len(global_episode_ids)
        per_source[source_id] = dict(stats)
        totals.update(stats)

    return per_source, dict(totals)


def safe_video_frame_count(video_path: str, fallback_json_path: str | None = None) -> int:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_count > 0:
        return frame_count
    if fallback_json_path and os.path.exists(fallback_json_path):
        with open(fallback_json_path, "r", encoding="utf-8") as handle:
            return len(json.load(handle))
    return 0


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "mean": 0, "max": 0, "sum": 0}
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "sum": sum(values),
    }


def compute_capacity(
    streams: list[dict],
    *,
    clip_frames: int,
    target_fps: int,
    stride: int,
) -> tuple[list[dict], dict[str, dict], dict[str, float]]:
    per_stream_rows = []
    by_source = defaultdict(lambda: defaultdict(list))

    for stream in streams:
        video_path = stream["video_path"]
        action_path = stream["action_path"]
        raw_frames = safe_video_frame_count(video_path, action_path)

        source_fps = float(stream.get("video_fps", 32) or 32)
        skip = max(1, round(source_fps / float(target_fps)))
        downsampled_frames = (raw_frames + skip - 1) // skip if raw_frames > 0 else 0
        dense_windows = max(0, downsampled_frames - clip_frames + 1)
        stride_windows = 0
        if downsampled_frames >= clip_frames:
            stride_windows = ((downsampled_frames - clip_frames) // stride) + 1

        row = {
            "source_id": stream["source_id"],
            "stream_uid": stream["stream_uid"],
            "episode_id": stream["episode_id"],
            "stem": stream["stem"],
            "split": stream["split"],
            "video_fps": source_fps,
            "raw_frames": raw_frames,
            "downsampled_frames": downsampled_frames,
            "max_dense_81f_windows": dense_windows,
            "max_stride_windows": stride_windows,
        }
        per_stream_rows.append(row)

        by_source[stream["source_id"]]["raw_frames"].append(raw_frames)
        by_source[stream["source_id"]]["downsampled_frames"].append(downsampled_frames)
        by_source[stream["source_id"]]["dense_windows"].append(dense_windows)
        by_source[stream["source_id"]]["stride_windows"].append(stride_windows)

    totals = {
        "stream_count": len(per_stream_rows),
        "dense_windows_total": sum(r["max_dense_81f_windows"] for r in per_stream_rows),
        "stride_windows_total": sum(r["max_stride_windows"] for r in per_stream_rows),
        "raw_frames_total": sum(r["raw_frames"] for r in per_stream_rows),
    }
    return per_stream_rows, by_source, totals


def count_existing_dataset(dataset_dir: str) -> dict[str, int]:
    root = Path(dataset_dir)
    out = {}
    for split in ("train", "val"):
        csv_path = root / f"metadata_{split}.csv"
        if not csv_path.exists():
            out[split] = 0
            continue
        with open(csv_path, "r", encoding="utf-8") as handle:
            out[split] = sum(1 for _ in csv.DictReader(handle))
    out["total"] = out["train"] + out["val"]
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw Phy-CSGO capacity before mining clips.")
    parser.add_argument("--input_dirs", type=str, required=True, help="Comma-separated raw source roots")
    parser.add_argument("--clip_frames", type=int, default=81)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--compare_dataset_dir", type=str, default="", help="Optional existing mined dataset root")
    parser.add_argument("--report_csv", type=str, default="", help="Optional CSV path for per-stream stats")
    parser.add_argument("--topn", type=int, default=10, help="How many streams to show in the max-window leaderboard")
    args = parser.parse_args()

    input_dirs = [item.strip() for item in args.input_dirs.split(",") if item.strip()]
    if not input_dirs:
        raise SystemExit("No input dirs provided")

    per_source_layout, totals_layout = count_raw_layout(input_dirs)
    streams = prep.find_player_streams(input_dirs)
    per_stream_rows, by_source_capacity, totals_capacity = compute_capacity(
        streams,
        clip_frames=args.clip_frames,
        target_fps=args.target_fps,
        stride=args.stride,
    )

    print("\n=== Raw Layout ===")
    for source_id in sorted(per_source_layout):
        stats = per_source_layout[source_id]
        print(
            f"[{source_id}] episode_dirs={stats.get('episode_dirs', 0)} "
            f"episode_info={stats.get('episode_info_files', 0)} "
            f"player_mp4={stats.get('player_mp4_files', 0)} "
            f"action_json={stats.get('action_json_files', 0)} "
            f"discoverable_scan={stats.get('discoverable_streams_scan', 0)} "
            f"encountered_error={stats.get('encountered_error', 0)} "
            f"missing_video={stats.get('missing_video', 0)} "
            f"missing_action={stats.get('missing_action', 0)}"
        )

    raw_video_total = totals_layout.get("player_mp4_files", 0)
    print(
        f"\nRaw totals: player_mp4={raw_video_total}, "
        f"discoverable_streams={len(streams)}, "
        f"required_two_thirds={math.ceil(raw_video_total * 2 / 3)}"
    )

    print("\n=== Capacity By Source ===")
    for source_id in sorted(by_source_capacity):
        cap = by_source_capacity[source_id]
        dense_stats = summarize(cap["dense_windows"])
        stride_stats = summarize(cap["stride_windows"])
        frame_stats = summarize(cap["downsampled_frames"])
        print(
            f"[{source_id}] streams={len(cap['dense_windows'])} "
            f"downsampled_frames(min/p50/mean/max)="
            f"{frame_stats['min']}/{frame_stats['p50']:.1f}/{frame_stats['mean']:.1f}/{frame_stats['max']} "
            f"dense_windows(sum/max)={dense_stats['sum']}/{dense_stats['max']} "
            f"stride_windows(sum/max)={stride_stats['sum']}/{stride_stats['max']}"
        )

    print("\n=== Capacity Totals ===")
    dense_max = max((r["max_dense_81f_windows"] for r in per_stream_rows), default=0)
    stride_max = max((r["max_stride_windows"] for r in per_stream_rows), default=0)
    print(f"streams={totals_capacity['stream_count']}")
    print(f"total_dense_81f_windows={totals_capacity['dense_windows_total']}")
    print(f"total_stride_windows={totals_capacity['stride_windows_total']}")
    print(f"max_dense_81f_windows_per_stream={dense_max}")
    print(f"max_stride_windows_per_stream={stride_max}")

    print("\n=== Top Streams By Dense 81f Windows ===")
    top_dense = sorted(
        per_stream_rows,
        key=lambda row: (row["max_dense_81f_windows"], row["downsampled_frames"]),
        reverse=True,
    )[: args.topn]
    for row in top_dense:
        print(
            f"{row['stream_uid']}: raw_frames={row['raw_frames']} "
            f"downsampled={row['downsampled_frames']} "
            f"dense_windows={row['max_dense_81f_windows']} "
            f"stride_windows={row['max_stride_windows']}"
        )

    if args.compare_dataset_dir:
        existing = count_existing_dataset(args.compare_dataset_dir)
        print("\n=== Existing Dataset Comparison ===")
        print(
            f"dataset={args.compare_dataset_dir}\n"
            f"clips train/val/total = {existing['train']}/{existing['val']}/{existing['total']}"
        )
        if raw_video_total > 0:
            print(f"ratio_vs_raw_videos = {existing['total'] / raw_video_total:.4f}")
        if totals_capacity["dense_windows_total"] > 0:
            print(f"ratio_vs_dense_windows = {existing['total'] / totals_capacity['dense_windows_total']:.8f}")
        if totals_capacity["stride_windows_total"] > 0:
            print(f"ratio_vs_stride_windows = {existing['total'] / totals_capacity['stride_windows_total']:.8f}")

    if args.report_csv:
        write_csv(args.report_csv, per_stream_rows)
        print(f"\nWrote per-stream report -> {args.report_csv}")


if __name__ == "__main__":
    main()
