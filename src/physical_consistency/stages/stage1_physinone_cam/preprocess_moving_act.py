"""Preprocess PhysInOne moving-camera clips into act-compatible samples."""

from __future__ import annotations

import argparse
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from physical_consistency.common.io import ensure_dir, write_csv_rows, write_json
from physical_consistency.common.logging_utils import configure_logging

from .preprocess import (
    _compute_intrinsics,
    _discover_zips,
    _read_rgb_frame,
    _slice_members_into_windows,
    _split_zips,
    _trajectory_prompt,
    _write_video_mp4,
    _zip_bucket_name,
)

LOGGER = logging.getLogger(__name__)
MOVING_CAMERA_NAME = "CineCamera_Moving"
ACTION_CHANNELS = ("forward", "strafe", "yaw", "pitch")
FIXED_CAMERA_PATTERN = re.compile(r"/(CineCamera_\d+)/rgb/")


@dataclass(slots=True)
class MovingActPreprocessArgs:
    input_root: str
    output_dir: str
    clip_frames: int
    sampling_mode: str
    window_stride: int
    output_height: int
    output_width: int
    target_fps: int
    default_camera_angle_x: float
    val_ratio: float
    seed: int
    max_zips: int
    action_normalize_percentile: float
    min_action_scale: float
    include_static_cameras: bool = False
    static_camera_ids: str = "all"
    moving_repeat: int = 1


def parse_args() -> MovingActPreprocessArgs:
    parser = argparse.ArgumentParser(
        description="Preprocess PhysInOne CineCamera_Moving into act-compatible clips."
    )
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_frames", type=int, default=81)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=("contiguous_windows", "uniform_single"),
        default="uniform_single",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=81,
        help="Stride for contiguous windows. Values <= 0 fall back to clip_frames.",
    )
    parser.add_argument("--output_height", type=int, default=384)
    parser.add_argument("--output_width", type=int, default=384)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--default_camera_angle_x", type=float, default=1.166746013987712)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_zips", type=int, default=0)
    parser.add_argument(
        "--action_normalize_percentile",
        type=float,
        default=95.0,
        help="Global absolute percentile used to normalize pseudo-actions.",
    )
    parser.add_argument(
        "--min_action_scale",
        type=float,
        default=1.0e-6,
        help="Lower bound for each action normalization scale.",
    )
    parser.add_argument(
        "--include_static_cameras",
        action="store_true",
        help="Also export fixed CineCamera_N samples with zero action.npy.",
    )
    parser.add_argument(
        "--static_camera_ids",
        type=str,
        default="all",
        help="Comma-separated fixed camera ids/names to export, or 'all'. Used with --include_static_cameras.",
    )
    parser.add_argument(
        "--moving_repeat",
        type=int,
        default=1,
        help="Repeat moving-camera rows in metadata_train.csv to oversample non-zero pseudo-actions.",
    )
    args = parser.parse_args()
    return MovingActPreprocessArgs(**vars(args))


def _camera_sort_key(name: str) -> tuple[int, str]:
    if name == MOVING_CAMERA_NAME:
        return (10**9, name)
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (10**9 - 1, name)


def _find_camera_json_member(zf: zipfile.ZipFile, camera_name: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(f"blender_{camera_name}.json")]
    if not matches:
        raise FileNotFoundError(f"No blender_{camera_name}.json found inside trajectory zip")
    if len(matches) > 1:
        LOGGER.warning("Multiple %s json files found; using %s", camera_name, matches[0])
    return matches[0]


def _find_moving_json_member(zf: zipfile.ZipFile) -> str:
    return _find_camera_json_member(zf, MOVING_CAMERA_NAME)


def _load_camera_metadata(zf: zipfile.ZipFile, camera_name: str) -> dict:
    member = _find_camera_json_member(zf, camera_name)
    return json.loads(zf.read(member).decode("utf-8"))


def _load_moving_metadata(zf: zipfile.ZipFile) -> dict:
    return _load_camera_metadata(zf, MOVING_CAMERA_NAME)


def _sorted_camera_rgb_members(zf: zipfile.ZipFile, camera_name: str) -> list[str]:
    prefix = f"/{camera_name}/rgb/"
    members = [name for name in zf.namelist() if prefix in name and name.lower().endswith(".jpg")]
    return sorted(members, key=lambda name: int(Path(name).stem))


def _sorted_moving_rgb_members(zf: zipfile.ZipFile) -> list[str]:
    return _sorted_camera_rgb_members(zf, MOVING_CAMERA_NAME)


def _camera_poses_from_metadata(metadata: dict, *, camera_name: str) -> np.ndarray:
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{camera_name} metadata has no frames list")
    sorted_frames = sorted(frames, key=lambda item: int(item.get("frame", 0)))
    poses = []
    for frame in sorted_frames:
        matrix = frame.get("transform_matrix")
        if matrix is None:
            raise ValueError(f"{camera_name} frame {frame.get('frame')} lacks transform_matrix")
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"Unexpected transform_matrix shape {arr.shape}; expected (4, 4)")
        poses.append(arr)
    return np.stack(poses, axis=0)


def _moving_poses_from_metadata(metadata: dict) -> np.ndarray:
    return _camera_poses_from_metadata(metadata, camera_name=MOVING_CAMERA_NAME)


def _discover_static_camera_names(zf: zipfile.ZipFile) -> list[str]:
    cameras = {
        match.group(1)
        for name in zf.namelist()
        if (match := FIXED_CAMERA_PATTERN.search(name))
    }
    return sorted(cameras, key=_camera_sort_key)


def _selected_static_camera_names(zf: zipfile.ZipFile, spec: str) -> list[str]:
    available = _discover_static_camera_names(zf)
    normalized = str(spec or "all").strip()
    if normalized.lower() in {"", "all", "*"}:
        return available

    available_set = set(available)
    selected: list[str] = []
    for raw_token in normalized.split(","):
        token = raw_token.strip()
        if not token:
            continue
        name = token if token.startswith("CineCamera_") else f"CineCamera_{token}"
        if name in available_set:
            selected.append(name)
        else:
            LOGGER.warning("Requested static camera %s is absent in zip; available=%s", name, available)
    return sorted(dict.fromkeys(selected), key=_camera_sort_key)


def _raw_pose_delta_actions(poses: np.ndarray) -> np.ndarray:
    """Convert c2w pose deltas into [forward, strafe, yaw, pitch] actions."""

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    actions = np.zeros((len(poses), 4), dtype=np.float32)
    if len(poses) <= 1:
        return actions

    for index in range(len(poses) - 1):
        rot0 = poses[index, :3, :3]
        pos0 = poses[index, :3, 3]
        rot1 = poses[index + 1, :3, :3]
        pos1 = poses[index + 1, :3, 3]

        delta_local = rot0.T @ (pos1 - pos0)
        relative_rot = rot0.T @ rot1
        rotvec, _ = cv2.Rodrigues(relative_rot.astype(np.float64))
        rotvec = rotvec.reshape(3).astype(np.float32)

        actions[index, 0] = float(delta_local[2])
        actions[index, 1] = float(delta_local[0])
        actions[index, 2] = float(rotvec[1])
        actions[index, 3] = float(rotvec[0])

    actions[-1] = actions[-2]
    return actions


def _compute_action_scale(
    zips: list[Path],
    *,
    args: MovingActPreprocessArgs,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for zpath in zips:
        with zipfile.ZipFile(zpath, "r") as zf:
            rgb_members = _sorted_moving_rgb_members(zf)
            if not rgb_members:
                continue
            poses_all = _moving_poses_from_metadata(_load_moving_metadata(zf))
            windows = _slice_members_into_windows(
                rgb_members,
                clip_frames=args.clip_frames,
                sampling_mode=args.sampling_mode,
                window_stride=args.window_stride,
            )
            for _sampled_members, sampled_indices in windows:
                max_index = max(sampled_indices)
                if max_index >= len(poses_all):
                    raise ValueError(
                        f"{zpath} sampled frame index {max_index} exceeds pose count {len(poses_all)}"
                    )
                actions = _raw_pose_delta_actions(poses_all[sampled_indices])
                chunks.append(np.abs(actions))

    if not chunks:
        raise RuntimeError("No moving-camera actions found while computing normalization scale")

    all_actions = np.concatenate(chunks, axis=0)
    percentile = float(args.action_normalize_percentile)
    if not (0.0 < percentile <= 100.0):
        raise ValueError(f"action_normalize_percentile must be in (0, 100], got {percentile}")
    scale = np.percentile(all_actions, percentile, axis=0).astype(np.float32)
    return np.maximum(scale, float(args.min_action_scale)).astype(np.float32)


def _normalize_actions(actions: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if scale.shape != (4,):
        raise ValueError(f"action scale must have shape (4,), got {scale.shape}")
    return np.clip(actions / scale[None, :], -1.0, 1.0).astype(np.float32)


def _write_moving_clip(
    *,
    output_dir: Path,
    split: str,
    trajectory_name: str,
    camera_name: str,
    clip_index: int,
    frames: list[np.ndarray],
    fps: int,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    actions: np.ndarray,
    metadata: dict,
) -> dict[str, str]:
    clip_name = f"{trajectory_name}__{camera_name}_clip{clip_index:04d}"
    clip_dir = output_dir / split / "clips" / clip_name
    ensure_dir(clip_dir)
    _write_video_mp4(frames, clip_dir / "video.mp4", fps)
    cv2.imwrite(str(clip_dir / "image.jpg"), frames[0])
    np.save(clip_dir / "poses.npy", poses.astype(np.float32))
    np.save(clip_dir / "intrinsics.npy", intrinsics.astype(np.float32))
    np.save(clip_dir / "action.npy", actions.astype(np.float32))
    (clip_dir / "prompt.txt").write_text(str(metadata["prompt"]), encoding="utf-8")
    write_json(clip_dir / "source_metadata.json", metadata)
    action_scale = metadata["action_scale"]
    return {
        "clip_path": f"{split}/clips/{clip_name}",
        "prompt": str(metadata["prompt"]),
        "trajectory_name": str(metadata["trajectory_name"]),
        "camera_id": str(metadata["camera_id"]),
        "physics_bucket": str(metadata["physics_bucket"]),
        "source_zip": str(metadata["source_zip"]),
        "source_height": str(metadata["source_height"]),
        "source_width": str(metadata["source_width"]),
        "raw_frame_count": str(metadata["raw_frame_count"]),
        "clip_frames": str(metadata["clip_frames"]),
        "target_fps": str(metadata["target_fps"]),
        "sampling_mode": str(metadata["sampling_mode"]),
        "window_index": str(metadata["window_index"]),
        "window_start_frame": str(metadata["window_start_frame"]),
        "window_end_frame": str(metadata["window_end_frame"]),
        "control_type": "act",
        "action_source": str(metadata["action_source"]),
        "action_channels": ",".join(ACTION_CHANNELS),
        "action_scale_forward": f"{float(action_scale[0]):.9g}",
        "action_scale_strafe": f"{float(action_scale[1]):.9g}",
        "action_scale_yaw": f"{float(action_scale[2]):.9g}",
        "action_scale_pitch": f"{float(action_scale[3]):.9g}",
    }


def _process_camera(
    zf: zipfile.ZipFile,
    zpath: Path,
    *,
    camera_name: str,
    action_source: str,
    split: str,
    args: MovingActPreprocessArgs,
    output_dir: Path,
    action_scale: np.ndarray,
) -> list[dict[str, str]]:
    rgb_members = _sorted_camera_rgb_members(zf, camera_name)
    if not rgb_members:
        raise ValueError(f"No {camera_name}/rgb frames found inside {zpath}")
    camera_meta = _load_camera_metadata(zf, camera_name)
    poses_all = _camera_poses_from_metadata(camera_meta, camera_name=camera_name)
    camera_angle_x = float(camera_meta.get("camera_angle_x", args.default_camera_angle_x))
    trajectory_name = zpath.stem
    prompt = _trajectory_prompt(trajectory_name)
    intrinsics_single = _compute_intrinsics(camera_angle_x, args.output_width, args.output_height)
    windows = _slice_members_into_windows(
        rgb_members,
        clip_frames=args.clip_frames,
        sampling_mode=args.sampling_mode,
        window_stride=args.window_stride,
    )
    rows: list[dict[str, str]] = []
    for clip_index, (sampled_members, sampled_indices) in enumerate(windows):
        max_index = max(sampled_indices)
        if max_index >= len(poses_all):
            raise ValueError(
                f"{zpath} sampled frame index {max_index} exceeds pose count {len(poses_all)}"
            )
        frames = [
            _read_rgb_frame(
                zf,
                member,
                width=args.output_width,
                height=args.output_height,
            )
            for member in sampled_members
        ]
        poses = poses_all[sampled_indices]
        intrinsics = np.repeat(intrinsics_single[None, ...], args.clip_frames, axis=0)
        if action_source == "moving_camera_pose_delta":
            raw_actions = _raw_pose_delta_actions(poses)
            actions = _normalize_actions(raw_actions, action_scale)
        else:
            actions = np.zeros((args.clip_frames, len(ACTION_CHANNELS)), dtype=np.float32)
        row = _write_moving_clip(
            output_dir=output_dir,
            split=split,
            trajectory_name=trajectory_name,
            camera_name=camera_name,
            clip_index=clip_index,
            frames=frames,
            fps=args.target_fps,
            poses=poses,
            intrinsics=intrinsics,
            actions=actions,
            metadata={
                "prompt": prompt,
                "trajectory_name": trajectory_name,
                "camera_id": camera_name,
                "physics_bucket": _zip_bucket_name(zpath, Path(args.input_root)),
                "source_zip": str(zpath),
                "source_height": args.output_height,
                "source_width": args.output_width,
                "raw_height": int(camera_meta.get("img_h", 0) or 0),
                "raw_width": int(camera_meta.get("img_w", 0) or 0),
                "raw_fps": float(camera_meta.get("fps", 0.0) or 0.0),
                "raw_frame_count": len(rgb_members),
                "json_frame_count": len(poses_all),
                "clip_frames": args.clip_frames,
                "target_fps": args.target_fps,
                "sampling_mode": args.sampling_mode,
                "window_index": clip_index,
                "window_start_frame": sampled_indices[0],
                "window_end_frame": sampled_indices[-1],
                "sampled_frame_indices": sampled_indices,
                "control_type": "act",
                "action_source": action_source,
                "action_channels": list(ACTION_CHANNELS),
                "action_scale": [float(value) for value in action_scale],
                "action_normalize_percentile": args.action_normalize_percentile,
            },
        )
        rows.append(row)
    return rows


def _process_zip(
    zpath: Path,
    *,
    split: str,
    args: MovingActPreprocessArgs,
    output_dir: Path,
    action_scale: np.ndarray,
) -> list[dict[str, str]]:
    LOGGER.info("Processing act-compatible cameras %s -> %s", zpath.name, split)
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zpath, "r") as zf:
        rows.extend(
            _process_camera(
                zf,
                zpath,
                camera_name=MOVING_CAMERA_NAME,
                action_source="moving_camera_pose_delta",
                split=split,
                args=args,
                output_dir=output_dir,
                action_scale=action_scale,
            )
        )
        if args.include_static_cameras:
            for camera_name in _selected_static_camera_names(zf, args.static_camera_ids):
                rows.extend(
                    _process_camera(
                        zf,
                        zpath,
                        camera_name=camera_name,
                        action_source="static_camera_zero",
                        split=split,
                        args=args,
                        output_dir=output_dir,
                        action_scale=action_scale,
                    )
                )
    return rows


def _repeat_moving_rows(rows: list[dict[str, str]], repeat: int) -> list[dict[str, str]]:
    repeat = max(int(repeat), 1)
    if repeat <= 1:
        return rows
    moving_rows = [row for row in rows if row.get("camera_id") == MOVING_CAMERA_NAME]
    expanded = list(rows)
    for _ in range(repeat - 1):
        expanded.extend(dict(row) for row in moving_rows)
    return expanded


def run_preprocess(args: MovingActPreprocessArgs) -> dict:
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir / "logs" / "preprocess_physinone_moving_act.log")
    ensure_dir(output_dir)
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"PhysInOne input_root does not exist: {input_root}")

    all_zips = _discover_zips(input_root)
    if args.max_zips > 0:
        all_zips = all_zips[: args.max_zips]
    train_zips, val_zips = _split_zips(all_zips, val_ratio=args.val_ratio, seed=args.seed)
    LOGGER.info(
        "Found %s trajectory zips (train=%s, val=%s)",
        len(all_zips),
        len(train_zips),
        len(val_zips),
    )

    scale_zips = train_zips if train_zips else all_zips
    action_scale = _compute_action_scale(scale_zips, args=args)
    LOGGER.info(
        "Global action scale (%sth percentile): %s",
        args.action_normalize_percentile,
        dict(zip(ACTION_CHANNELS, [float(value) for value in action_scale])),
    )

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for zpath in train_zips:
        train_rows.extend(
            _process_zip(
                zpath,
                split="train",
                args=args,
                output_dir=output_dir,
                action_scale=action_scale,
            )
        )
    for zpath in val_zips:
        val_rows.extend(
            _process_zip(
                zpath,
                split="val",
                args=args,
                output_dir=output_dir,
                action_scale=action_scale,
            )
        )

    train_unique_clip_count = len(train_rows)
    val_unique_clip_count = len(val_rows)
    train_rows = _repeat_moving_rows(train_rows, args.moving_repeat)

    fieldnames = [
        "clip_path",
        "prompt",
        "trajectory_name",
        "camera_id",
        "physics_bucket",
        "source_zip",
        "source_height",
        "source_width",
        "raw_frame_count",
        "clip_frames",
        "target_fps",
        "sampling_mode",
        "window_index",
        "window_start_frame",
        "window_end_frame",
        "control_type",
        "action_source",
        "action_channels",
        "action_scale_forward",
        "action_scale_strafe",
        "action_scale_yaw",
        "action_scale_pitch",
    ]
    write_csv_rows(output_dir / "metadata_train.csv", train_rows, fieldnames)
    write_csv_rows(output_dir / "metadata_val.csv", val_rows, fieldnames)
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "trajectory_zip_count": len(all_zips),
        "train_zip_count": len(train_zips),
        "val_zip_count": len(val_zips),
        "train_clip_count": len(train_rows),
        "val_clip_count": len(val_rows),
        "train_unique_clip_count": train_unique_clip_count,
        "val_unique_clip_count": val_unique_clip_count,
        "train_moving_row_count": sum(1 for row in train_rows if row.get("camera_id") == MOVING_CAMERA_NAME),
        "train_static_row_count": sum(1 for row in train_rows if row.get("action_source") == "static_camera_zero"),
        "action_scale_zip_count": len(scale_zips),
        "clip_frames": args.clip_frames,
        "sampling_mode": args.sampling_mode,
        "window_stride": args.window_stride if args.window_stride > 0 else args.clip_frames,
        "output_height": args.output_height,
        "output_width": args.output_width,
        "target_fps": args.target_fps,
        "camera_id": MOVING_CAMERA_NAME,
        "include_static_cameras": bool(args.include_static_cameras),
        "static_camera_ids": str(args.static_camera_ids),
        "moving_repeat": max(int(args.moving_repeat), 1),
        "control_type": "act",
        "action_source": "moving_camera_pose_delta+static_camera_zero"
        if args.include_static_cameras
        else "moving_camera_pose_delta",
        "action_channels": list(ACTION_CHANNELS),
        "action_normalize_percentile": args.action_normalize_percentile,
        "action_scale": [float(value) for value in action_scale],
    }
    write_json(output_dir / "preprocess_summary.json", summary)
    LOGGER.info("train: %s clips -> %s", len(train_rows), output_dir / "metadata_train.csv")
    LOGGER.info("val: %s clips -> %s", len(val_rows), output_dir / "metadata_val.csv")
    LOGGER.info("summary: %s", summary)
    return summary


def main() -> None:
    run_preprocess(parse_args())


if __name__ == "__main__":
    main()
