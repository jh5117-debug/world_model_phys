"""Preprocess PhysInOne raw zips into standardized camera-only training clips."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from physical_consistency.common.io import ensure_dir, write_csv_rows, write_json
from physical_consistency.common.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
CAMERA_PATTERN = re.compile(r"(CineCamera_\d+)")


@dataclass(slots=True)
class PreprocessArgs:
    input_root: str
    output_dir: str
    clip_frames: int
    output_height: int
    output_width: int
    target_fps: int
    default_camera_angle_x: float
    val_ratio: float
    seed: int
    max_zips: int


def parse_args() -> PreprocessArgs:
    parser = argparse.ArgumentParser(description="Preprocess PhysInOne raw trajectories.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_frames", type=int, default=81)
    parser.add_argument("--output_height", type=int, default=480)
    parser.add_argument("--output_width", type=int, default=480)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--default_camera_angle_x", type=float, default=1.166746013987712)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_zips", type=int, default=0)
    args = parser.parse_args()
    return PreprocessArgs(**vars(args))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        raise ValueError("Cannot normalize a near-zero vector while parsing cameras.ply")
    return vector / norm


def _camera_sort_key(name: str) -> tuple[int, str]:
    match = CAMERA_PATTERN.search(name)
    return (int(match.group(1).split("_")[-1]), name) if match else (10**9, name)


def _trajectory_prompt(trajectory_name: str) -> str:
    core = trajectory_name.removesuffix("_trajectory")
    physics_name = core.split("__")[0]
    return physics_name.replace("_", " ").strip()


def _list_camera_names(namelist: list[str]) -> list[str]:
    cameras = {match.group(1) for name in namelist if (match := CAMERA_PATTERN.search(name))}
    return sorted(cameras, key=_camera_sort_key)


def _read_special_metadata(zf: zipfile.ZipFile) -> dict:
    json_member = next((name for name in zf.namelist() if name.endswith("transforms_train.json")), "")
    if not json_member:
        return {}
    return json.loads(zf.read(json_member).decode("utf-8"))


def _parse_ply_camera_groups(zf: zipfile.ZipFile) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    ply_member = next((name for name in zf.namelist() if name.endswith("cameras.ply")), "")
    if not ply_member:
        raise FileNotFoundError("No cameras.ply found inside trajectory zip")
    lines = zf.read(ply_member).decode("utf-8", errors="replace").splitlines()
    header_end = next((idx for idx, line in enumerate(lines) if line.strip() == "end_header"), None)
    if header_end is None:
        raise ValueError("Invalid cameras.ply: missing end_header")
    vertex_count = 0
    for line in lines[: header_end + 1]:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
    if vertex_count <= 0:
        raise ValueError("Invalid cameras.ply: no vertex count")
    body = lines[header_end + 1 :]
    vertices = body[:vertex_count]
    if len(vertices) % 9 != 0:
        raise ValueError(f"Unexpected cameras.ply vertex count {len(vertices)}; expected groups of 9")

    groups: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for start in range(0, len(vertices), 9):
        points = []
        colors = []
        for raw_line in vertices[start : start + 9]:
            parts = raw_line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed cameras.ply vertex line: {raw_line!r}")
            x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
            color = (int(parts[3]), int(parts[4]), int(parts[5]))
            points.append([x, y, z])
            colors.append(color)
        color = colors[0]
        groups.append((color, np.asarray(points, dtype=np.float32)))
    return groups


def _build_c2w_from_group(points: np.ndarray) -> np.ndarray:
    center = points[0]
    x_axis = _normalize(points[6] - center)
    y_hint = points[7] - center
    y_axis = _normalize(y_hint - np.dot(y_hint, x_axis) * x_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    target_z = _normalize(-(points[8] - center))
    if float(np.dot(z_axis, target_z)) < 0:
        z_axis = -z_axis
    y_axis = _normalize(np.cross(z_axis, x_axis))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = center
    return c2w


def _extract_camera_pose_map(zf: zipfile.ZipFile, camera_names: list[str]) -> dict[str, np.ndarray]:
    groups = _parse_ply_camera_groups(zf)
    static_groups = [(color, points) for color, points in groups if color != (255, 0, 0)]
    if len(static_groups) != len(camera_names):
        raise ValueError(
            "Could not align cameras.ply static frustums with camera directories: "
            f"{len(static_groups)} non-red groups vs {len(camera_names)} cameras"
        )
    pose_map: dict[str, np.ndarray] = {}
    for camera_name, (_color, points) in zip(sorted(camera_names, key=_camera_sort_key), static_groups):
        pose_map[camera_name] = _build_c2w_from_group(points)
    return pose_map


def _compute_intrinsics(camera_angle_x: float, width: int, height: int) -> np.ndarray:
    fx = 0.5 * float(width) / math.tan(float(camera_angle_x) / 2.0)
    fy = fx
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.asarray([fx, fy, cx, cy], dtype=np.float32)


def _sample_members(members: list[str], clip_frames: int) -> tuple[list[str], list[int]]:
    if not members:
        raise ValueError("No RGB frames found for requested camera")
    if len(members) == 1:
        return [members[0]] * clip_frames, [0] * clip_frames
    raw_indices = np.linspace(0, len(members) - 1, clip_frames).round().astype(int)
    return [members[idx] for idx in raw_indices], raw_indices.tolist()


def _sorted_frame_members(zf: zipfile.ZipFile, camera_name: str, kind: str) -> list[str]:
    prefix = f"/{camera_name}/{kind}/"
    members = [name for name in zf.namelist() if prefix in name and name.lower().endswith(".jpg")]
    return sorted(members, key=lambda name: int(Path(name).stem))


def _read_rgb_frame(zf: zipfile.ZipFile, member: str, *, width: int, height: int) -> np.ndarray:
    encoded = np.frombuffer(zf.read(member), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode {member}")
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return frame


def _write_video_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frames[0].shape[1], frames[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def _zip_bucket_name(zpath: Path, input_root: Path) -> str:
    relative_parent = zpath.parent.relative_to(input_root)
    if str(relative_parent) == ".":
        return "TopLevel"
    return relative_parent.name


def _write_clip(
    *,
    output_dir: Path,
    split: str,
    trajectory_name: str,
    camera_name: str,
    frames: list[np.ndarray],
    fps: int,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    metadata: dict,
) -> dict[str, str]:
    clip_name = f"{trajectory_name}__{camera_name}_clip0000"
    clip_dir = output_dir / split / "clips" / clip_name
    ensure_dir(clip_dir)
    _write_video_mp4(frames, clip_dir / "video.mp4", fps)
    np.save(clip_dir / "poses.npy", poses)
    np.save(clip_dir / "intrinsics.npy", intrinsics)
    (clip_dir / "prompt.txt").write_text(str(metadata["prompt"]), encoding="utf-8")
    write_json(clip_dir / "source_metadata.json", metadata)
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
        "control_type": "cam",
    }


def _process_zip(zpath: Path, *, split: str, args: PreprocessArgs, output_dir: Path) -> list[dict[str, str]]:
    LOGGER.info("Processing %s -> %s", zpath.name, split)
    with zipfile.ZipFile(zpath, "r") as zf:
        namelist = zf.namelist()
        camera_names = _list_camera_names(namelist)
        if not camera_names:
            raise ValueError(f"No CineCamera_* directories found inside {zpath}")
        special_meta = _read_special_metadata(zf)
        camera_angle_x = float(special_meta.get("camera_angle_x", args.default_camera_angle_x))
        pose_map = _extract_camera_pose_map(zf, camera_names)
        trajectory_name = zpath.stem
        prompt = _trajectory_prompt(trajectory_name)
        intrinsics_single = _compute_intrinsics(camera_angle_x, args.output_width, args.output_height)
        rows: list[dict[str, str]] = []
        for camera_name in camera_names:
            rgb_members = _sorted_frame_members(zf, camera_name, "rgb")
            sampled_members, sampled_indices = _sample_members(rgb_members, args.clip_frames)
            frames = [
                _read_rgb_frame(
                    zf,
                    member,
                    width=args.output_width,
                    height=args.output_height,
                )
                for member in sampled_members
            ]
            poses = np.repeat(pose_map[camera_name][None, ...], args.clip_frames, axis=0)
            intrinsics = np.repeat(intrinsics_single[None, ...], args.clip_frames, axis=0)
            row = _write_clip(
                output_dir=output_dir,
                split=split,
                trajectory_name=trajectory_name,
                camera_name=camera_name,
                frames=frames,
                fps=args.target_fps,
                poses=poses,
                intrinsics=intrinsics,
                metadata={
                    "prompt": prompt,
                    "trajectory_name": trajectory_name,
                    "camera_id": camera_name,
                    "physics_bucket": _zip_bucket_name(zpath, Path(args.input_root)),
                    "source_zip": str(zpath),
                    "source_height": args.output_height,
                    "source_width": args.output_width,
                    "raw_frame_count": len(rgb_members),
                    "clip_frames": args.clip_frames,
                    "target_fps": args.target_fps,
                    "sampled_frame_indices": sampled_indices,
                },
            )
            rows.append(row)
    return rows


def _discover_zips(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*_trajectory.zip"))


def _split_zips(zips: list[Path], *, val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if val_ratio <= 0.0:
        return zips, []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(zips))
    rng.shuffle(indices)
    val_count = int(round(len(zips) * val_ratio))
    val_indices = set(indices[:val_count].tolist())
    train, val = [], []
    for idx, zpath in enumerate(zips):
        (val if idx in val_indices else train).append(zpath)
    return train, val


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    configure_logging(output_dir / "logs" / "preprocess_physinone.log")
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

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for zpath in train_zips:
        train_rows.extend(_process_zip(zpath, split="train", args=args, output_dir=output_dir))
    for zpath in val_zips:
        val_rows.extend(_process_zip(zpath, split="val", args=args, output_dir=output_dir))

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
        "control_type",
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
        "clip_frames": args.clip_frames,
        "output_height": args.output_height,
        "output_width": args.output_width,
        "target_fps": args.target_fps,
        "default_camera_angle_x": args.default_camera_angle_x,
    }
    write_json(output_dir / "preprocess_summary.json", summary)
    LOGGER.info("train: %s clips -> %s", len(train_rows), output_dir / "metadata_train.csv")
    LOGGER.info("val: %s clips -> %s", len(val_rows), output_dir / "metadata_val.csv")
    LOGGER.info("summary: %s", summary)


if __name__ == "__main__":
    main()
