"""
Data Preprocessor for dust2-460 Dataset (4.9data) + LingBot-World-Fast
======================================================================
Adapted for the new dataset format (April 2026):
  - Multiple data subdirectories (different matches)
  - frame_count is consecutive (gap=1), no sparsity
  - render_transform always available
  - New action fields: scope, flash_alpha, inspect, plant, defuse, etc.
  - New files: _seg.mkv, _seg_colormap.json (not used for training)

Action dimensions (8): forward, back, left, right, jump, crouch, walk, fire
  All 8 are binary (0/1). flash_alpha removed (confirmed invalid: constant 1.0 in new data).

Target: 16fps clips for Fast model (chunk-based autoregressive)

April 2026 refactor:
  - Switch from dense fixed-stride clipping to motion-aware clip mining
  - Score every 81-frame window with camera motion + camera rotation + action intensity + world events
  - Skip low-information freeze-time / static windows
  - Keep only top-k aggressive clips per player stream

Usage:
    python preprocess_4.9data.py \\
        --input_dirs /path/to/dir1,/path/to/dir2 \\
        --output_dir /path/to/processed_4.9data \\
        --clip_frames 81 \\
        --target_fps 16 \\
        --top_k_per_stream 3 \\
        --val_episodes "000010,000020"
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


TEAM_STEM_RE = re.compile(r"_team_(\d+)_")
EPISODE_PLAYER_CACHE = {}


# ============================================================
# Discovery
# ============================================================

def find_player_streams(input_dirs, skip_episodes=None, val_episodes=None):
    """
    Walk multiple data directories and find all per-player streams.

    Supports the new dust2-460 structure:
        input_dir/
          train/
            Ep_000003/
              <stem>.json, <stem>.mp4, <stem>_episode_info.json, ...

    Args:
        input_dirs: List of data root directories
        skip_episodes: Set of episode IDs to skip entirely
        val_episodes: Set of episode IDs to assign to val split
    """
    skip_set = set(skip_episodes or [])
    val_set = set(val_episodes or [])
    streams = []

    for input_dir in input_dirs:
        source_dir_name = Path(input_dir).name
        source_id = source_dir_name[:8]
        for split_name in ["train", "test"]:
            split_dir = os.path.join(input_dir, split_name)
            if not os.path.isdir(split_dir):
                continue

            for ep_dir_name in sorted(os.listdir(split_dir)):
                ep_dir = os.path.join(split_dir, ep_dir_name)
                if not os.path.isdir(ep_dir) or not ep_dir_name.startswith("Ep_"):
                    continue

                episode_id = ep_dir_name.replace("Ep_", "")

                if episode_id in skip_set:
                    print(f"[SKIP] Episode {episode_id}")
                    continue

                # Determine split
                assigned_split = "val" if episode_id in val_set else split_name

                # Detect map name
                navmesh_path = os.path.join(input_dir, "navmesh.json")
                map_name = "de_dust2"  # default
                if os.path.exists(navmesh_path):
                    try:
                        with open(navmesh_path) as f:
                            nm = json.load(f)
                        map_name = nm.get("map_name", "de_dust2")
                    except Exception:
                        pass

                # Find episode_info files to discover players
                for fname in sorted(os.listdir(ep_dir)):
                    if not fname.endswith("_episode_info.json"):
                        continue

                    stem = fname.replace("_episode_info.json", "")
                    info_path = os.path.join(ep_dir, fname)
                    video_path = os.path.join(ep_dir, f"{stem}.mp4")
                    action_path = os.path.join(ep_dir, f"{stem}.json")

                    if not os.path.exists(video_path):
                        print(f"[WARN] Video not found: {video_path}")
                        continue
                    if not os.path.exists(action_path):
                        print(f"[WARN] Action JSON not found: {action_path}")
                        continue

                    with open(info_path) as f:
                        info = json.load(f)

                    if info.get("encountered_error", False):
                        print(f"[WARN] {stem}: encountered_error=true, skipping")
                        continue

                    streams.append({
                        "stem": stem,
                        "stream_uid": f"{source_id}__{stem}",
                        "source_id": source_id,
                        "episode_id": episode_id,
                        "split": assigned_split,
                        "video_path": video_path,
                        "action_path": action_path,
                        "info": info,
                        "ep_dir": ep_dir,
                        "map_name": map_name,
                        "video_fps": info.get("video_fps", 32),
                        "tf_ratio": info.get("tf_ratio", 4),
                        "tickrate": info.get("tickrate", 128),
                        "data_root": input_dir,
                    })

    print(f"Found {len(streams)} player streams across "
          f"{len(set(s['episode_id'] for s in streams))} episodes")
    return streams


# ============================================================
# Coordinate conversion
# ============================================================

def csgo_to_pose_matrix(yaw_deg, pitch_deg, x, y, z):
    """Convert CSGO yaw/pitch/position to 4x4 camera-to-world matrix (OpenCV)."""
    if yaw_deg > 180:
        yaw_deg -= 360
    if pitch_deg > 180:
        pitch_deg -= 360

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    forward = np.array([cp * cy, cp * sy, -sp])
    right = np.array([-sy, cy, 0.0])
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    R_opencv = np.stack([right, -up, forward], axis=1)

    pose = np.eye(4)
    pose[:3, :3] = R_opencv
    pose[:3, 3] = np.array([x, y, z])
    return pose


def fov_to_intrinsics(fov_deg, height, width):
    """Convert horizontal FOV to camera intrinsics [fx, fy, cx, cy]."""
    fov_rad = np.radians(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


# ============================================================
# Video extraction
# ============================================================

def extract_video_frames(video_path, video_fps, target_fps):
    """Extract frames, downsampling from video_fps to target_fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0 and abs(actual_fps - video_fps) < 5:
        source_fps = actual_fps
    else:
        source_fps = video_fps

    skip = max(1, round(source_fps / target_fps))

    frames = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frames.append(frame)
            frame_indices.append(idx)
        idx += 1

    cap.release()
    return frames, frame_indices, source_fps


# ============================================================
# Action extraction and motion mining
# ============================================================

EVENT_WEIGHTS = {
    "player_death": 4.0,
    "player_hurt": 3.0,
    "flashbang_detonate": 2.0,
    "hegrenade_detonate": 2.0,
    "smokegrenade_detonate": 2.0,
    "molotov_detonate": 2.0,
    "inferno_startburn": 2.0,
}

UTILITY_EVENT_TYPES = {
    "flashbang_detonate",
    "hegrenade_detonate",
    "smokegrenade_detonate",
    "molotov_detonate",
    "inferno_startburn",
}


def wrap_degrees(delta_deg):
    """Wrap an angle difference into [-180, 180)."""
    return ((delta_deg + 180.0) % 360.0) - 180.0


def parse_team_id_from_stem(stem):
    """Parse `team_<id>` from a player-stream stem."""
    match = TEAM_STEM_RE.search(stem)
    return int(match.group(1)) if match else None


def load_world_events(ep_dir):
    """Load per-episode world events as JSONL."""
    events_path = os.path.join(ep_dir, "world_events.jsonl")
    events = []
    if not os.path.exists(events_path):
        return events

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_episode_player_tracks(ep_dir):
    """
    Load all player tracks for one episode so we can estimate enemy-in-view for ego clips.

    The cache is intentionally lightweight: only compact per-frame state keyed by frame_count is stored.
    """
    cache_key = os.path.abspath(ep_dir)
    cached = EPISODE_PLAYER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tracks = []
    for fname in sorted(os.listdir(ep_dir)):
        if not fname.endswith("_episode_info.json"):
            continue

        stem = fname.replace("_episode_info.json", "")
        action_path = os.path.join(ep_dir, f"{stem}.json")
        info_path = os.path.join(ep_dir, fname)
        if not os.path.exists(action_path):
            continue

        try:
            with open(info_path) as f:
                info = json.load(f)
            with open(action_path) as f:
                action_frames = json.load(f)
        except Exception:
            continue

        team_id = info.get("team_id", parse_team_id_from_stem(stem))
        states = {}
        for af in action_frames:
            frame_count = af.get("frame_count")
            state = extract_state_fields(af)
            if frame_count is None or state is None:
                continue
            states[frame_count] = {
                "x": state["x"],
                "y": state["y"],
                "z": state["z"],
                "yaw": state["yaw"],
                "pitch": state["pitch"],
                "render_x": state["render_x"],
                "render_y": state["render_y"],
                "render_z": state["render_z"],
                "crouching": state["crouching"],
            }

        tracks.append({
            "stem": stem,
            "team_id": int(team_id) if team_id is not None else None,
            "states": states,
        })

    EPISODE_PLAYER_CACHE[cache_key] = tracks
    return tracks


def event_weight(event):
    """Map a world event to a scalar importance used for motion mining."""
    return EVENT_WEIGHTS.get(event.get("event_type"), 0.0)


def find_first_event_frame(events, event_type):
    """Return the earliest frame_count for a given event type, if present."""
    frame_counts = [
        e.get("frame_count")
        for e in events
        if e.get("event_type") == event_type and e.get("frame_count") is not None
    ]
    return min(frame_counts) if frame_counts else None


def aggregate_events_to_video_frames(video_frame_indices, world_events, total_source_frames):
    """
    Aggregate raw world events into the kept video-frame bins after FPS downsampling.

    If source FPS is 32 and target FPS is 16, each kept frame owns the source-frame
    interval [current_kept_idx, next_kept_idx).
    """
    if not world_events:
        return [[] for _ in video_frame_indices]

    events_by_frame = {}
    for event in world_events:
        frame_count = event.get("frame_count")
        if frame_count is None:
            continue
        events_by_frame.setdefault(frame_count, []).append(event)

    aligned_events = []
    for idx, start_fc in enumerate(video_frame_indices):
        if idx + 1 < len(video_frame_indices):
            end_fc = video_frame_indices[idx + 1]
        else:
            end_fc = total_source_frames

        bucket = []
        for frame_count in range(start_fc, end_fc):
            bucket.extend(events_by_frame.get(frame_count, []))
        aligned_events.append(bucket)

    return aligned_events


def action_to_vector(act):
    """Convert an action dict to the 8-dim LingBot action format."""
    return np.array([
        int(bool(act.get("forward", False))),
        int(bool(act.get("back", False))),
        int(bool(act.get("left", False))),
        int(bool(act.get("right", False))),
        int(bool(act.get("jump", False))),
        int(bool(act.get("crouch", False))),
        int(bool(act.get("walk", False))),
        int(bool(act.get("fire", False))),
    ], dtype=np.float32)


def action_intensity(act):
    """
    Score how dynamic the current action state is.

    This keeps the first-stage miner lightweight: it only depends on the action JSON
    that is already available for every player stream.
    """
    move_flag = max(
        int(bool(act.get("forward", False))),
        int(bool(act.get("back", False))),
        int(bool(act.get("left", False))),
        int(bool(act.get("right", False))),
        int(bool(act.get("walk", False))),
    )
    look_mag = math.sqrt(
        float(act.get("look_dx", 0.0)) ** 2 + float(act.get("look_dy", 0.0)) ** 2
    )

    return (
        0.6 * move_flag
        + 0.3 * look_mag
        + 2.0 * int(bool(act.get("fire", False)))
        + 1.0 * int(bool(act.get("jump", False)))
        + 0.8 * int(bool(act.get("crouch", False)))
        + 0.6 * int(bool(act.get("reload", False)))
        + 0.4 * int(bool(act.get("scope", False)))
        + 0.3 * int(bool(act.get("use", False)))
    )


def extract_state_fields(af):
    """
    Extract pose-relevant state from a single action frame.

    Returns None for invalid/dead frames.
    """
    if af is None:
        return None
    if af.get("health", 100) <= 0:
        return None

    rt = af.get("render_transform")
    cam_pos = af.get("camera_position")
    if rt is not None and rt.get("x") is not None:
        x = float(rt["x"])
        y = float(rt["y"])
        z = float(cam_pos[2]) if cam_pos is not None else float(rt["z"])
    elif cam_pos is not None:
        x = float(cam_pos[0])
        y = float(cam_pos[1])
        z = float(cam_pos[2])
    else:
        x = float(af.get("x", 0.0))
        y = float(af.get("y", 0.0))
        z = float(af.get("z", 0.0))

    cam_rot = af.get("camera_rotation")
    if cam_rot is not None:
        pitch = float(cam_rot[1])
        yaw = float(cam_rot[2])
    else:
        yaw = float(af.get("yaw", 0.0))
        pitch = float(af.get("pitch", 0.0))

    act = af.get("action", {})
    render_x = float(rt["x"]) if rt is not None and rt.get("x") is not None else x
    render_y = float(rt["y"]) if rt is not None and rt.get("y") is not None else y
    render_z = float(rt["z"]) if rt is not None and rt.get("z") is not None else (z - 64.0)
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "pitch": pitch,
        "render_x": render_x,
        "render_y": render_y,
        "render_z": render_z,
        "crouching": bool(act.get("crouch", False)),
        "action_dict": act,
        "action_vec": action_to_vector(act),
    }


def extract_pose_and_action(af, default_fov=106.26):
    """
    Extract camera pose and 8-dim action from a single frame.
    Returns (pose_4x4, action_8d, fov) or None if invalid.
    """
    state = extract_state_fields(af)
    if state is None:
        return None

    pose = csgo_to_pose_matrix(
        state["yaw"], state["pitch"], state["x"], state["y"], state["z"]
    )
    return pose, state["action_vec"], default_fov


def lookup_track_state(track, frame_count, max_offset=2):
    """Lookup a compact player state by frame_count with a tiny nearest-neighbor fallback."""
    state = track["states"].get(frame_count)
    if state is not None:
        return state
    for offset in range(1, max_offset + 1):
        state = track["states"].get(frame_count + offset)
        if state is not None:
            return state
        state = track["states"].get(frame_count - offset)
        if state is not None:
            return state
    return None


def lookup_track_position(track, frame_count, max_offset=2):
    """Backward-compatible helper that returns only the head position."""
    state = lookup_track_state(track, frame_count, max_offset=max_offset)
    if state is None:
        return None
    return (state["x"], state["y"], state["z"])


def project_world_point_to_image(world_xyz, c2w, intrinsics, width, height):
    """Project a world-space point into the ego image plane using OpenCV coordinates."""
    w2c = np.linalg.inv(c2w)
    world_h = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=np.float32)
    cam_xyz = w2c @ world_h
    x_cam, y_cam, z_cam = float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2])
    if z_cam <= 1e-4:
        return None

    fx, fy, cx, cy = [float(x) for x in intrinsics]
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    if not (0.0 <= u < float(width) and 0.0 <= v < float(height)):
        return None
    return u, v, z_cam


def decode_depth_frame(bgr_frame):
    """
    Decode 16-bit depth from ffv1/rgb24 frame.

    Depth is packed as depth_16 = R * 256 + G.
    OpenCV loads RGB24 as BGR, so R=channel 2 and G=channel 1.
    """
    r = bgr_frame[:, :, 2].astype(np.uint32)
    g = bgr_frame[:, :, 1].astype(np.uint32)
    return (r * 256 + g).astype(np.float32)


def depth_to_game_units(depth_16, max_game_units=2048.0):
    """Convert raw 16-bit depth to game units using the per-stream far plane."""
    return depth_16 / 65535.0 * float(max_game_units)


def load_depth_stream_info(ep_dir, stem, default_far_units=2048.0):
    """Load the depth-video path and far-plane units for a player stream."""
    depth_path = os.path.join(ep_dir, f"{stem}_depth.mkv")
    manifest_path = os.path.join(ep_dir, f"{stem}_video_manifest.json")

    info = {
        "depth_path": depth_path if os.path.exists(depth_path) else None,
        "depth_far_units": float(default_far_units),
    }
    if not os.path.exists(manifest_path):
        return info

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:
        return info

    if isinstance(manifest, list):
        for item in manifest:
            if item.get("stream_type") == "depth":
                info["depth_far_units"] = float(item.get("depth_far_units", default_far_units))
                break

    return info


def load_seg_stream_info(ep_dir, stem):
    """Load the segmentation-video path and quantized palette for a player stream."""
    seg_path = os.path.join(ep_dir, f"{stem}_seg.mkv")
    colormap_path = os.path.join(ep_dir, f"{stem}_seg_colormap.json")
    info = {
        "seg_path": seg_path if os.path.exists(seg_path) else None,
        "seg_palette_lookup": {},
    }
    if not os.path.exists(colormap_path):
        return info

    try:
        with open(colormap_path) as f:
            colormap = json.load(f)
    except Exception:
        return info

    if not isinstance(colormap, dict):
        return info

    palette_lookup = {}
    for entity_id, entry in colormap.items():
        color = entry.get("color")
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            continue
        quantized_color = tuple(
            int(np.clip(np.round(float(c) / 16.0) * 16.0, 0, 255))
            for c in color
        )
        palette_lookup[quantized_color] = {
            "entity_id": entity_id,
            "index": entry.get("index"),
        }
    info["seg_palette_lookup"] = palette_lookup
    return info


def open_depth_frame_reader(depth_path, width, height, max_game_units):
    """
    Open a sequential depth reader that returns resized depth maps for requested source-frame indices.

    We deliberately stream depth frames instead of caching them all in memory.
    """
    if depth_path is None or not os.path.exists(depth_path):
        return None

    cap = cv2.VideoCapture(depth_path)
    if not cap.isOpened():
        return None

    return {
        "cap": cap,
        "width": int(width),
        "height": int(height),
        "max_game_units": float(max_game_units),
        "source_index": -1,
        "last_frame": None,
    }


def close_depth_frame_reader(reader):
    """Release a depth reader opened by open_depth_frame_reader()."""
    if reader is not None and reader.get("cap") is not None:
        reader["cap"].release()


def open_seg_frame_reader(seg_path, width, height):
    """Open a sequential segmentation reader that returns nearest-neighbor resized frames."""
    if seg_path is None or not os.path.exists(seg_path):
        return None

    cap = cv2.VideoCapture(seg_path)
    if not cap.isOpened():
        return None

    return {
        "cap": cap,
        "width": int(width),
        "height": int(height),
        "source_index": -1,
        "last_frame": None,
    }


def close_seg_frame_reader(reader):
    """Release a segmentation reader opened by open_seg_frame_reader()."""
    if reader is not None and reader.get("cap") is not None:
        reader["cap"].release()


def quantize_seg_frame(seg_frame):
    """Quantize a segmentation frame to the 16-level palette used by seg_colormap."""
    return np.clip(
        np.round(seg_frame.astype(np.float32) / 16.0) * 16.0,
        0,
        255,
    ).astype(np.uint8)


def read_seg_frame(reader, target_source_index):
    """Advance a sequential segmentation reader until target_source_index and return that quantized frame."""
    if reader is None:
        return None
    if target_source_index < reader["source_index"]:
        return None

    while reader["source_index"] < target_source_index:
        ret, seg = reader["cap"].read()
        reader["source_index"] += 1
        if not ret:
            return None
        if reader["source_index"] == target_source_index:
            seg = cv2.resize(
                seg,
                (reader["width"], reader["height"]),
                interpolation=cv2.INTER_NEAREST,
            )
            seg = quantize_seg_frame(seg)
            reader["last_frame"] = seg
            return seg

    return reader["last_frame"] if reader["source_index"] == target_source_index else None


def read_depth_frame(reader, target_source_index):
    """Advance a sequential depth reader until target_source_index and return that decoded frame."""
    if reader is None:
        return None
    if target_source_index < reader["source_index"]:
        return None

    while reader["source_index"] < target_source_index:
        ret, bgr = reader["cap"].read()
        reader["source_index"] += 1
        if not ret:
            return None
        if reader["source_index"] == target_source_index:
            depth_16 = decode_depth_frame(bgr)
            depth_game = depth_to_game_units(depth_16, reader["max_game_units"])
            depth_game = cv2.resize(
                depth_game,
                (reader["width"], reader["height"]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32, copy=False)
            reader["last_frame"] = depth_game
            return depth_game

    return reader["last_frame"] if reader["source_index"] == target_source_index else None


def sample_local_depth(depth_frame, u, v, patch_radius=2, percentile=75.0):
    """
    Sample a local depth neighborhood around the projected point.

    Using an upper percentile is a practical compromise: it is stricter than taking the
    raw max depth in the patch, but more tolerant than using a single pixel that may land
    on a silhouette edge or weapon model.
    """
    if depth_frame is None:
        return None

    h, w = depth_frame.shape[:2]
    ix = int(round(u))
    iy = int(round(v))
    x0 = max(0, ix - patch_radius)
    x1 = min(w, ix + patch_radius + 1)
    y0 = max(0, iy - patch_radius)
    y1 = min(h, iy + patch_radius + 1)
    patch = depth_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, percentile))


def sample_seg_patch(
    seg_frame,
    u,
    v,
    seg_palette_lookup,
    patch_radius=3,
    min_dominant_ratio=0.12,
    min_pixels=3,
):
    """
    Inspect a local segmentation patch around a projected point.

    Because seg.mkv is video-compressed, we quantize colors back onto the 16-level palette and
    look for a dominant palette color in the patch. This acts as a lightweight occupancy check.
    """
    if seg_frame is None or not seg_palette_lookup:
        return None

    h, w = seg_frame.shape[:2]
    ix = int(round(u))
    iy = int(round(v))
    x0 = max(0, ix - patch_radius)
    x1 = min(w, ix + patch_radius + 1)
    y0 = max(0, iy - patch_radius)
    y1 = min(h, iy + patch_radius + 1)
    patch = seg_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    total_pixels = patch.shape[0] * patch.shape[1]
    counts = {}
    valid_pixels = 0
    for pixel in patch.reshape(-1, 3):
        color = tuple(int(c) for c in pixel)
        if color == (0, 0, 0):
            continue
        if color not in seg_palette_lookup:
            continue
        counts[color] = counts.get(color, 0) + 1
        valid_pixels += 1

    if not counts:
        return {
            "occupied": False,
            "dominant_label": None,
            "dominant_pixels": 0,
            "dominant_ratio": 0.0,
            "dominant_total_ratio": 0.0,
            "dominant_valid_ratio": 0.0,
            "valid_ratio": 0.0,
            "support_score": 0.0,
        }

    dominant_label, dominant_pixels = max(counts.items(), key=lambda kv: kv[1])
    dominant_total_ratio = dominant_pixels / float(total_pixels)
    valid_ratio = valid_pixels / float(total_pixels)
    dominant_valid_ratio = dominant_pixels / float(max(valid_pixels, 1))
    support_score = max(
        dominant_total_ratio,
        dominant_valid_ratio * min(1.0, valid_ratio * 3.0),
    )
    occupied = (
        dominant_pixels >= min_pixels
        and (
            dominant_total_ratio >= min_dominant_ratio
            or (
                valid_pixels >= min_pixels
                and valid_ratio >= 0.06
                and dominant_valid_ratio >= 0.55
            )
        )
    )
    return {
        "occupied": bool(occupied),
        "dominant_label": dominant_label,
        "dominant_pixels": int(dominant_pixels),
        "dominant_ratio": float(support_score),
        "dominant_total_ratio": float(dominant_total_ratio),
        "dominant_valid_ratio": float(dominant_valid_ratio),
        "valid_ratio": float(valid_ratio),
        "support_score": float(support_score),
    }


def compute_player_body_points(state):
    """
    Approximate a player with multiple keypoints instead of a single head point.

    We use the camera/head position plus torso points derived from the render/origin height
    and the player's yaw. This is intentionally lightweight but much more robust than a
    single head point, especially for edge peeks and partial cover.
    """
    head = np.array([state["x"], state["y"], state["z"]], dtype=np.float32)
    origin = np.array([state["render_x"], state["render_y"], state["render_z"]], dtype=np.float32)
    body_height = float(np.clip(head[2] - origin[2], 40.0, 72.0))
    if state.get("crouching", False):
        body_height = max(34.0, body_height * 0.82)

    yaw_rad = np.radians(float(state["yaw"]))
    right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=np.float32)
    shoulder_half_width = float(np.clip(0.18 * body_height, 8.0, 15.0))

    upper_torso = head.copy()
    upper_torso[2] = head[2] - 0.34 * body_height

    mid_torso = head.copy()
    mid_torso[2] = head[2] - 0.55 * body_height

    lower_torso = head.copy()
    lower_torso[2] = head[2] - 0.75 * body_height

    return [
        ("head", head),
        ("upper_torso", upper_torso),
        ("upper_left_torso", upper_torso + right * shoulder_half_width),
        ("upper_right_torso", upper_torso - right * shoulder_half_width),
        ("mid_torso", mid_torso),
        ("lower_torso", lower_torso),
    ]


def compute_frame_enemy_annotations(
    stream,
    tracks,
    frame_idx,
    ego_pose,
    intrinsics,
    width,
    height,
    depth_frame=None,
    seg_frame=None,
    seg_palette_lookup=None,
    ego_team=None,
    max_depth=4000.0,
    close_depth=1400.0,
    center_ratio=0.35,
    occlusion_depth_tolerance=96.0,
    occlusion_patch_radius=2,
    seg_patch_radius=3,
    seg_min_dominant_ratio=0.12,
    seg_min_pixels=3,
):
    """Project enemy head/torso keypoints into the ego frame and classify them as visible/occluded."""
    if ego_pose is None:
        return []

    cx, cy = float(intrinsics[2]), float(intrinsics[3])
    annotations = []
    for track in tracks:
        if track["stem"] == stream["stem"]:
            continue
        if ego_team is not None and track["team_id"] == ego_team:
            continue

        enemy_state = lookup_track_state(track, frame_idx)
        if enemy_state is None:
            continue

        point_annotations = []
        projected_points = 0
        visible_points = 0
        torso_visible_points = 0
        head_visible = False
        center_visible = False
        min_visible_depth = float(max_depth)
        label_counts = {}
        label_support = {}
        seg_supported_points = 0

        for label, world_xyz in compute_player_body_points(enemy_state):
            projection = project_world_point_to_image(
                world_xyz=world_xyz,
                c2w=ego_pose,
                intrinsics=intrinsics,
                width=width,
                height=height,
            )
            if projection is None:
                point_annotations.append({
                    "label": label,
                    "projected": False,
                    "visible": False,
                    "occluded": False,
                })
                continue

            u, v, z_cam = projection
            if z_cam > max_depth:
                point_annotations.append({
                    "label": label,
                    "projected": False,
                    "visible": False,
                    "occluded": False,
                })
                continue

            projected_points += 1
            local_depth = sample_local_depth(
                depth_frame=depth_frame,
                u=u,
                v=v,
                patch_radius=occlusion_patch_radius,
            ) if depth_frame is not None else None
            depth_occluded = bool(
                local_depth is not None and z_cam > local_depth + float(occlusion_depth_tolerance)
            )
            seg_info = sample_seg_patch(
                seg_frame=seg_frame,
                u=u,
                v=v,
                seg_palette_lookup=seg_palette_lookup,
                patch_radius=seg_patch_radius,
                min_dominant_ratio=seg_min_dominant_ratio,
                min_pixels=seg_min_pixels,
            ) if seg_frame is not None and seg_palette_lookup else None
            seg_supported = bool(seg_info is not None and seg_info["occupied"])
            depth_visible = not depth_occluded
            visible = depth_visible

            if visible:
                visible_points += 1
                min_visible_depth = min(min_visible_depth, z_cam)
                if label == "head":
                    head_visible = True
                else:
                    torso_visible_points += 1
                if abs(u - cx) <= width * center_ratio and abs(v - cy) <= height * center_ratio:
                    center_visible = True
                if seg_supported:
                    seg_supported_points += 1
                if seg_supported and seg_info is not None and seg_info["dominant_label"] is not None:
                    dom = seg_info["dominant_label"]
                    label_counts[dom] = label_counts.get(dom, 0) + 1
                    label_support[dom] = label_support.get(dom, 0.0) + float(seg_info.get("support_score", seg_info["dominant_ratio"]))

            point_annotations.append({
                "label": label,
                "projected": True,
                "visible": visible,
                "depth_visible": depth_visible,
                "occluded": depth_occluded,
                "seg_supported": seg_supported,
                "u": float(u),
                "v": float(v),
                "z_cam": float(z_cam),
                "local_depth": None if local_depth is None else float(local_depth),
                "seg_dominant_label": None if seg_info is None else seg_info["dominant_label"],
                "seg_dominant_ratio": 0.0 if seg_info is None else float(seg_info["dominant_ratio"]),
                "seg_valid_ratio": 0.0 if seg_info is None else float(seg_info["valid_ratio"]),
                "seg_support_score": 0.0 if seg_info is None else float(seg_info.get("support_score", seg_info["dominant_ratio"])),
            })

        if projected_points == 0:
            continue

        visible_fraction = visible_points / float(projected_points)
        dominant_label = None
        dominant_label_count = 0
        occupancy_strength = 0.0
        if label_counts:
            dominant_label, dominant_label_count = max(label_counts.items(), key=lambda kv: kv[1])
            occupancy_strength = label_support[dominant_label] / float(max(dominant_label_count, 1))

        if seg_frame is not None and seg_palette_lookup:
            enemy_visible = (
                (head_visible and seg_supported_points >= 1)
                or seg_supported_points >= 2
                or (visible_points >= 4 and visible_fraction >= 0.67)
                or (head_visible and visible_points >= 3 and visible_fraction >= 0.5)
                or (visible_points >= 3 and visible_fraction >= 0.5 and occupancy_strength >= 0.08)
            )
        else:
            enemy_visible = head_visible or torso_visible_points >= 2 or visible_fraction >= 0.34

        annotations.append({
            "stem": track["stem"],
            "projected_points": projected_points,
            "visible_points": visible_points,
            "visible_fraction": float(visible_fraction),
            "enemy_visible": bool(enemy_visible),
            "is_close": bool(enemy_visible and min_visible_depth <= close_depth),
            "is_center": bool(enemy_visible and center_visible),
            "min_visible_depth": float(min_visible_depth),
            "dominant_seg_label": dominant_label,
            "dominant_seg_label_count": int(dominant_label_count),
            "seg_supported_points": int(seg_supported_points),
            "occupancy_strength": float(occupancy_strength),
            "points": point_annotations,
        })

    return annotations


def compute_enemy_view_signals(
    stream,
    records,
    video_frame_indices,
    width,
    height,
    default_fov=106.26,
    max_depth=4000.0,
    close_depth=1400.0,
    center_ratio=0.35,
    occlusion_depth_tolerance=96.0,
    occlusion_patch_radius=2,
    seg_patch_radius=3,
    seg_min_dominant_ratio=0.12,
    seg_min_pixels=3,
):
    """
    Estimate ego-centric combat visibility by projecting enemy player positions into the ego view.

    This is intentionally approximate:
      - enemy position = that enemy player's camera/head position
      - visibility = inside frustum + in front of ego + within depth range
      - if depth is available, the projected enemy must also survive an occlusion check
    """
    tracks = load_episode_player_tracks(stream["ep_dir"])
    ego_team = stream["info"].get("team_id", parse_team_id_from_stem(stream["stem"]))
    intrinsics = fov_to_intrinsics(default_fov, height, width)
    cx, cy = float(intrinsics[2]), float(intrinsics[3])

    depth_info = load_depth_stream_info(stream["ep_dir"], stream["stem"])
    seg_info = load_seg_stream_info(stream["ep_dir"], stream["stem"])
    depth_reader = open_depth_frame_reader(
        depth_path=depth_info["depth_path"],
        width=width,
        height=height,
        max_game_units=depth_info["depth_far_units"],
    )
    seg_reader = open_seg_frame_reader(
        seg_path=seg_info["seg_path"],
        width=width,
        height=height,
    )
    depth_enabled = depth_reader is not None
    seg_enabled = seg_reader is not None and bool(seg_info["seg_palette_lookup"])
    effective_max_depth = min(float(max_depth), float(depth_info["depth_far_units"]) + float(occlusion_depth_tolerance))

    signals = []
    try:
        for frame_idx, record in zip(video_frame_indices, records):
            pose = record["pose"]
            depth_frame = read_depth_frame(depth_reader, frame_idx) if depth_enabled else None
            seg_frame = read_seg_frame(seg_reader, frame_idx) if seg_enabled else None
            if pose is None:
                signals.append({
                    "enemy_projected_count": 0,
                    "enemy_visible_count": 0,
                    "enemy_close_count": 0,
                    "enemy_center_count": 0,
                    "enemy_visibility_strength": 0.0,
                    "enemy_occupancy_strength": 0.0,
                    "enemy_min_depth": float(effective_max_depth),
                })
                continue

            frame_annotations = compute_frame_enemy_annotations(
                stream=stream,
                tracks=tracks,
                frame_idx=frame_idx,
                ego_pose=pose,
                intrinsics=intrinsics,
                width=width,
                height=height,
                depth_frame=depth_frame,
                seg_frame=seg_frame,
                seg_palette_lookup=seg_info["seg_palette_lookup"],
                ego_team=ego_team,
                max_depth=effective_max_depth,
                close_depth=close_depth,
                center_ratio=center_ratio,
                occlusion_depth_tolerance=occlusion_depth_tolerance,
                occlusion_patch_radius=occlusion_patch_radius,
                seg_patch_radius=seg_patch_radius,
                seg_min_dominant_ratio=seg_min_dominant_ratio,
                seg_min_pixels=seg_min_pixels,
            )
            projected_count = len(frame_annotations)
            visible_enemies = [ann for ann in frame_annotations if ann["enemy_visible"]]
            visible_count = len(visible_enemies)
            close_count = sum(int(ann["is_close"]) for ann in visible_enemies)
            center_count = sum(int(ann["is_center"]) for ann in visible_enemies)
            visibility_strength = float(
                np.mean([ann["visible_fraction"] for ann in frame_annotations])
            ) if frame_annotations else 0.0
            occupancy_strength = float(
                np.mean([ann["occupancy_strength"] for ann in visible_enemies])
            ) if visible_enemies else 0.0
            min_depth = min(
                [ann["min_visible_depth"] for ann in visible_enemies],
                default=float(effective_max_depth),
            )

            signals.append({
                "enemy_projected_count": projected_count,
                "enemy_visible_count": visible_count,
                "enemy_close_count": close_count,
                "enemy_center_count": center_count,
                "enemy_visibility_strength": visibility_strength,
                "enemy_occupancy_strength": occupancy_strength,
                "enemy_min_depth": min_depth,
            })
    finally:
        close_depth_frame_reader(depth_reader)
        close_seg_frame_reader(seg_reader)

    return signals, {
        "occlusion_enabled": depth_enabled,
        "seg_enabled": seg_enabled,
        "depth_far_units": float(depth_info["depth_far_units"]),
        "visibility_mode": (
            "occlusion_seg_aware_head_torso_multipoint"
            if depth_enabled and seg_enabled
            else ("occlusion_aware_head_torso_multipoint" if depth_enabled else "frustum_only_head_torso_multipoint")
        ),
        "visibility_target_mode": "head_torso_multipoint",
    }


def build_aligned_records(aligned_actions, aligned_events, default_fov=106.26):
    """
    Standardize all aligned frames so motion mining and clip export share one source.
    """
    records = []
    last_valid = None

    for af, event_bucket in zip(aligned_actions, aligned_events):
        state = extract_state_fields(af)
        valid = state is not None
        if valid:
            pose = csgo_to_pose_matrix(
                state["yaw"], state["pitch"], state["x"], state["y"], state["z"]
            )
            last_valid = {
                "state": state,
                "pose": pose,
            }
        elif last_valid is not None:
            state = dict(last_valid["state"])
            pose = last_valid["pose"]
        else:
            state = None
            pose = None

        records.append({
            "raw_frame": af,
            "valid": valid,
            "state": state,
            "pose": pose,
            "events": event_bucket,
            "event_score": float(sum(event_weight(e) for e in event_bucket)),
            "event_types": sorted({e.get("event_type", "unknown") for e in event_bucket}),
        })

    prev_state = None
    for record in records:
        state = record["state"]
        if state is None or prev_state is None:
            trans_step = 0.0
            rot_step = 0.0
        else:
            dx = state["x"] - prev_state["x"]
            dy = state["y"] - prev_state["y"]
            dz = state["z"] - prev_state["z"]
            trans_step = math.sqrt(dx * dx + dy * dy + 0.5 * dz * dz)
            yaw_step = abs(wrap_degrees(state["yaw"] - prev_state["yaw"]))
            pitch_step = abs(state["pitch"] - prev_state["pitch"])
            rot_step = yaw_step + 0.5 * pitch_step
            rot_step = min(rot_step, 60.0)

        if state is None:
            record["action_vec"] = np.zeros(8, dtype=np.float32)
            record["action_intensity"] = 0.0
        else:
            record["action_vec"] = state["action_vec"]
            record["action_intensity"] = float(action_intensity(state["action_dict"]))

        record["trans_step"] = float(trans_step)
        record["rot_step"] = float(rot_step)
        prev_state = state if state is not None else prev_state

    return records


def score_motion_windows(
    records,
    clip_frames,
    stride,
    round_freeze_end_idx=None,
    post_freeze_buffer_frames=24,
    min_valid_ratio=0.95,
):
    """
    Score every candidate 81-frame window and keep only physically informative ones.
    """
    if len(records) < clip_frames:
        return []

    min_start = 0
    if round_freeze_end_idx is not None:
        min_start = min(len(records) - clip_frames, round_freeze_end_idx + post_freeze_buffer_frames)
        min_start = max(min_start, 0)

    trans = np.array([r["trans_step"] for r in records], dtype=np.float32)
    rot = np.array([r["rot_step"] for r in records], dtype=np.float32)
    act = np.array([r["action_intensity"] for r in records], dtype=np.float32)
    evt = np.array([r["event_score"] for r in records], dtype=np.float32)
    valid = np.array([1.0 if r["valid"] else 0.0 for r in records], dtype=np.float32)

    candidates = []
    for start in range(min_start, len(records) - clip_frames + 1, stride):
        end = start + clip_frames
        sl = slice(start, end)
        valid_ratio = float(valid[sl].mean())
        if valid_ratio < min_valid_ratio:
            continue

        cam_p90 = float(np.percentile(trans[sl], 90))
        rot_p90 = float(np.percentile(rot[sl], 90))
        action_mean = float(act[sl].mean())
        event_score = float(evt[sl].sum())
        fire_frames = int(sum(
            int(bool(r["state"]["action_dict"].get("fire", False)))
            for r in records[start:end]
            if r["state"] is not None
        ))

        onset_left = slice(start, min(start + 8, end))
        onset_right = slice(min(start + 8, end), min(start + 24, end))
        onset_bonus = 0.0
        if onset_right.start < onset_right.stop:
            onset_bonus = float(trans[onset_right].mean() - trans[onset_left].mean())

        static_penalty = 0.0
        early = slice(start, min(start + 24, end))
        if float(np.percentile(trans[early], 90)) < 1.0 and float(evt[early].sum()) == 0.0:
            static_penalty = 1.0

        score = (
            0.35 * cam_p90
            + 0.25 * rot_p90
            + 0.20 * action_mean
            + 0.20 * event_score
            + 0.10 * max(onset_bonus, 0.0)
            - 0.25 * static_penalty
        )

        candidates.append({
            "start": start,
            "end": end,
            "motion_score": float(score),
            "cam_p90": cam_p90,
            "rot_p90": rot_p90,
            "action_mean": action_mean,
            "event_score": event_score,
            "fire_frames": fire_frames,
            "onset_bonus": float(onset_bonus),
            "static_penalty": static_penalty,
            "valid_ratio": valid_ratio,
            "event_types": sorted({
                event_type
                for r in records[start:end]
                for event_type in r["event_types"]
            }),
        })

    return candidates


def augment_candidates_with_combat(records, candidates, enemy_view_signals):
    """Augment motion candidates with ego-centric combat metrics."""
    if not candidates:
        return []

    fire = np.array([
        int(bool(r["state"]["action_dict"].get("fire", False))) if r["state"] is not None else 0
        for r in records
    ], dtype=np.int32)
    projected = np.array([s.get("enemy_projected_count", s["enemy_visible_count"]) for s in enemy_view_signals], dtype=np.float32)
    visible = np.array([s["enemy_visible_count"] for s in enemy_view_signals], dtype=np.float32)
    close = np.array([s["enemy_close_count"] for s in enemy_view_signals], dtype=np.float32)
    center = np.array([s["enemy_center_count"] for s in enemy_view_signals], dtype=np.float32)
    strength = np.array([s.get("enemy_visibility_strength", 0.0) for s in enemy_view_signals], dtype=np.float32)
    occupancy = np.array([s.get("enemy_occupancy_strength", 0.0) for s in enemy_view_signals], dtype=np.float32)

    for cand in candidates:
        start, end = cand["start"], cand["end"]
        proj_slice = projected[start:end]
        vis_slice = visible[start:end]
        close_slice = close[start:end]
        center_slice = center[start:end]
        strength_slice = strength[start:end]
        occupancy_slice = occupancy[start:end]
        fire_slice = fire[start:end]

        enemy_projected_ratio = float(np.mean(proj_slice > 0))
        enemy_visible_ratio = float(np.mean(vis_slice > 0))
        enemy_occluded_ratio = float(np.mean((proj_slice > 0) & (vis_slice <= 0)))
        enemy_visibility_strength = float(strength_slice.mean()) if len(strength_slice) else 0.0
        enemy_occupancy_strength = float(occupancy_slice.mean()) if len(occupancy_slice) else 0.0
        close_enemy_ratio = float(np.mean(close_slice > 0))
        center_enemy_ratio = float(np.mean(center_slice > 0))
        max_enemies_in_view = int(vis_slice.max()) if len(vis_slice) else 0
        mean_enemies_in_view = float(vis_slice.mean()) if len(vis_slice) else 0.0

        fire_with_enemy_frames = int(np.sum((fire_slice > 0) & (vis_slice > 0)))
        fire_burst_count = 0
        prev_fire = 0
        for cur_fire in fire_slice:
            if cur_fire and not prev_fire:
                fire_burst_count += 1
            prev_fire = cur_fire

        utility_frames = int(sum(
            any(event_type in UTILITY_EVENT_TYPES for event_type in r["event_types"])
            for r in records[start:end]
        ))

        combat_score = (
            3.0 * fire_with_enemy_frames
            + 1.5 * fire_burst_count
            + 8.0 * enemy_visible_ratio
            + 10.0 * close_enemy_ratio
            + 6.0 * center_enemy_ratio
            + 4.0 * enemy_visibility_strength
            + 2.5 * enemy_occupancy_strength
            + 1.5 * max_enemies_in_view
            + 0.5 * mean_enemies_in_view
            + 0.25 * cand["event_score"]
            + 0.10 * cand["motion_score"]
        )

        cand["enemy_projected_ratio"] = enemy_projected_ratio
        cand["enemy_visible_ratio"] = enemy_visible_ratio
        cand["enemy_occluded_ratio"] = enemy_occluded_ratio
        cand["enemy_visibility_strength"] = enemy_visibility_strength
        cand["enemy_occupancy_strength"] = enemy_occupancy_strength
        cand["close_enemy_ratio"] = close_enemy_ratio
        cand["center_enemy_ratio"] = center_enemy_ratio
        cand["max_enemies_in_view"] = max_enemies_in_view
        cand["mean_enemies_in_view"] = mean_enemies_in_view
        cand["fire_with_enemy_frames"] = fire_with_enemy_frames
        cand["fire_burst_count"] = fire_burst_count
        cand["utility_frames"] = utility_frames
        cand["combat_score"] = float(combat_score)

    return candidates


def passes_combat_gate(
    cand,
    min_enemy_visible_ratio=0.08,
    min_close_enemy_ratio=0.03,
    min_fire_enemy_frames=1,
    min_event_score=2.0,
):
    """Shared combat gate used both before and after per-window validation."""
    return (
        cand["fire_with_enemy_frames"] >= min_fire_enemy_frames
        or (
            cand["fire_frames"] >= 3
            and cand["enemy_visible_ratio"] >= max(0.05, min_enemy_visible_ratio * 0.5)
        )
        or (
            cand["enemy_visible_ratio"] >= min_enemy_visible_ratio
            and cand["event_score"] >= min_event_score
        )
        or (
            cand["enemy_visible_ratio"] >= min_enemy_visible_ratio
            and cand["close_enemy_ratio"] >= min_close_enemy_ratio
        )
        or cand["max_enemies_in_view"] >= 2
    )


def refresh_candidate_with_recomputed_combat(
    stream,
    records,
    video_frame_indices,
    candidate,
    width,
    height,
    default_fov=106.26,
    occlusion_depth_tolerance=96.0,
    occlusion_patch_radius=2,
    seg_patch_radius=3,
    seg_min_dominant_ratio=0.12,
    seg_min_pixels=3,
):
    """
    Recompute combat metrics for one selected window from scratch.

    This is a consistency guardrail: even if the stream-level mining pass drifts because of
    stateful readers or other bookkeeping issues, exported clips must still pass a local,
    freshly recomputed combat check.
    """
    start = candidate["start"]
    end = candidate["end"]
    window_records = records[start:end]
    window_source_indices = video_frame_indices[start:end]

    enemy_view_signals, _ = compute_enemy_view_signals(
        stream=stream,
        records=window_records,
        video_frame_indices=window_source_indices,
        width=width,
        height=height,
        default_fov=default_fov,
        occlusion_depth_tolerance=occlusion_depth_tolerance,
        occlusion_patch_radius=occlusion_patch_radius,
        seg_patch_radius=seg_patch_radius,
        seg_min_dominant_ratio=seg_min_dominant_ratio,
        seg_min_pixels=seg_min_pixels,
    )

    refreshed = dict(candidate)
    refreshed["start"] = 0
    refreshed["end"] = len(window_records)
    refreshed = augment_candidates_with_combat(
        records=window_records,
        candidates=[refreshed],
        enemy_view_signals=enemy_view_signals,
    )[0]
    refreshed["start"] = start
    refreshed["end"] = end
    return refreshed


def select_top_combat_windows(
    candidates,
    top_k,
    clip_frames,
    min_score_percentile=70.0,
    min_absolute_score=6.0,
    min_window_separation=None,
    min_enemy_visible_ratio=0.08,
    min_close_enemy_ratio=0.03,
    min_fire_enemy_frames=1,
    min_event_score=2.0,
):
    """
    Select non-overlapping top-k windows using combat_score, not just motion_score.
    """
    if not candidates:
        return []

    min_window_separation = (
        min_window_separation if min_window_separation is not None else clip_frames // 2
    )
    scores = np.array([c["combat_score"] for c in candidates], dtype=np.float32)
    score_floor = max(min_absolute_score, float(np.percentile(scores, min_score_percentile)))

    selected = []
    for cand in sorted(candidates, key=lambda x: x["combat_score"], reverse=True):
        if cand["combat_score"] < score_floor:
            continue

        if not passes_combat_gate(
            cand,
            min_enemy_visible_ratio=min_enemy_visible_ratio,
            min_close_enemy_ratio=min_close_enemy_ratio,
            min_fire_enemy_frames=min_fire_enemy_frames,
            min_event_score=min_event_score,
        ):
            continue

        too_close = False
        for chosen in selected:
            overlap = max(
                0,
                min(cand["end"], chosen["end"]) - max(cand["start"], chosen["start"]),
            )
            if overlap > clip_frames * 0.5 or abs(cand["start"] - chosen["start"]) < min_window_separation:
                too_close = True
                break
        if too_close:
            continue

        cand = dict(cand)
        cand["selected_rank"] = len(selected)
        selected.append(cand)
        if len(selected) >= top_k:
            break

    return selected


def is_static_clip(poses, rot_threshold_deg=1.0, trans_threshold=1e-3):
    """Check if a clip has almost no camera movement."""
    translations = poses[:, :3, 3]
    trans_range = translations.max(axis=0) - translations.min(axis=0)

    rotations = poses[:, :3, :3]
    rot_changes = []
    for i in range(1, len(rotations)):
        rel_rot = rotations[i] @ rotations[i - 1].T
        trace = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
        rot_changes.append(abs(np.arccos(trace)))
    max_rot = max(rot_changes) if rot_changes else 0

    return trans_range.max() < trans_threshold and max_rot < np.radians(rot_threshold_deg)


# ============================================================
# Clip extraction
# ============================================================

def process_stream(
    stream,
    output_dir,
    clip_frames=81,
    target_fps=16,
    height=480,
    width=832,
    stride=8,
    default_fov=106.26,
    top_k_per_stream=3,
    post_freeze_buffer_frames=24,
    min_valid_ratio=0.95,
    min_score_percentile=70.0,
    min_absolute_score=6.0,
    min_window_separation=None,
    min_enemy_visible_ratio=0.08,
    min_close_enemy_ratio=0.03,
    min_fire_enemy_frames=1,
    min_event_score=2.0,
    occlusion_depth_tolerance=96.0,
    occlusion_patch_radius=2,
    seg_patch_radius=3,
    seg_min_dominant_ratio=0.12,
    seg_min_pixels=3,
):
    """
    Process a single player stream by mining high-motion windows instead of dense slicing.

    The exported clip format stays compatible with LingBot:
      - video.mp4
      - image.jpg
      - poses.npy
      - action.npy
      - action_mask.npy
      - intrinsics.npy
      - prompt.txt
      - modality_flags.json
    """
    stem = stream["stem"]
    stream_uid = stream.get("stream_uid", stem)
    print(f"  Processing: {stem}")

    # Load action data
    with open(stream["action_path"]) as f:
        action_frames = json.load(f)
    if not action_frames:
        print(f"    [WARN] Empty action JSON")
        return []

    # Build frame_count lookup (new data is consecutive, but be safe)
    by_frame_count = {}
    for af in action_frames:
        fc = af.get("frame_count")
        if fc is not None:
            by_frame_count[fc] = af

    world_events = load_world_events(stream["ep_dir"])
    freeze_end_frame = find_first_event_frame(world_events, "round_freeze_end")

    # Extract video frames
    try:
        frames, video_frame_indices, source_fps = extract_video_frames(
            stream["video_path"], stream["video_fps"], target_fps
        )
    except Exception as e:
        print(f"    [ERROR] Video extraction failed: {e}")
        return []

    if len(frames) < clip_frames:
        print(f"    [WARN] Too short ({len(frames)} < {clip_frames})")
        return []

    # Align video frames to action data
    aligned_actions = []
    for vid_idx in video_frame_indices:
        af = by_frame_count.get(vid_idx)
        if af is None:
            for offset in [1, -1, 2, -2]:
                af = by_frame_count.get(vid_idx + offset)
                if af is not None:
                    break
        aligned_actions.append(af)

    aligned_events = aggregate_events_to_video_frames(
        video_frame_indices, world_events, total_source_frames=len(action_frames)
    )
    records = build_aligned_records(
        aligned_actions=aligned_actions,
        aligned_events=aligned_events,
        default_fov=default_fov,
    )

    freeze_end_idx = None
    if freeze_end_frame is not None:
        for idx, fc in enumerate(video_frame_indices):
            if fc >= freeze_end_frame:
                freeze_end_idx = idx
                break

    candidates = score_motion_windows(
        records=records,
        clip_frames=clip_frames,
        stride=stride,
        round_freeze_end_idx=freeze_end_idx,
        post_freeze_buffer_frames=post_freeze_buffer_frames,
        min_valid_ratio=min_valid_ratio,
    )
    enemy_view_signals, visibility_meta = compute_enemy_view_signals(
        stream=stream,
        records=records,
        video_frame_indices=video_frame_indices,
        width=width,
        height=height,
        default_fov=default_fov,
        occlusion_depth_tolerance=occlusion_depth_tolerance,
        occlusion_patch_radius=occlusion_patch_radius,
        seg_patch_radius=seg_patch_radius,
        seg_min_dominant_ratio=seg_min_dominant_ratio,
        seg_min_pixels=seg_min_pixels,
    )
    candidates = augment_candidates_with_combat(
        records=records,
        candidates=candidates,
        enemy_view_signals=enemy_view_signals,
    )
    selected_windows = select_top_combat_windows(
        candidates=candidates,
        top_k=max(top_k_per_stream * 5, top_k_per_stream),
        clip_frames=clip_frames,
        min_score_percentile=min_score_percentile,
        min_absolute_score=min_absolute_score,
        min_window_separation=min_window_separation,
        min_enemy_visible_ratio=min_enemy_visible_ratio,
        min_close_enemy_ratio=min_close_enemy_ratio,
        min_fire_enemy_frames=min_fire_enemy_frames,
        min_event_score=min_event_score,
    )
    validated_windows = []
    for window in selected_windows:
        refreshed = refresh_candidate_with_recomputed_combat(
            stream=stream,
            records=records,
            video_frame_indices=video_frame_indices,
            candidate=window,
            width=width,
            height=height,
            default_fov=default_fov,
            occlusion_depth_tolerance=occlusion_depth_tolerance,
            occlusion_patch_radius=occlusion_patch_radius,
            seg_patch_radius=seg_patch_radius,
            seg_min_dominant_ratio=seg_min_dominant_ratio,
            seg_min_pixels=seg_min_pixels,
        )
        if not passes_combat_gate(
            refreshed,
            min_enemy_visible_ratio=min_enemy_visible_ratio,
            min_close_enemy_ratio=min_close_enemy_ratio,
            min_fire_enemy_frames=min_fire_enemy_frames,
            min_event_score=min_event_score,
        ):
            continue
        refreshed["selected_rank"] = len(validated_windows)
        validated_windows.append(refreshed)
        if len(validated_windows) >= top_k_per_stream:
            break
    selected_windows = validated_windows

    split = stream["split"]
    map_name = stream["map_name"]

    clips = []
    if not selected_windows:
        print("    [WARN] No combat windows passed the combat filter")
        return clips

    selection_mode = (
        "topk_combat_occseg_aware"
        if visibility_meta.get("occlusion_enabled", False) and visibility_meta.get("seg_enabled", False)
        else (
            "topk_combat_occlusion_aware"
            if visibility_meta.get("occlusion_enabled", False)
            else "topk_combat"
        )
    )

    for clip_idx, window in enumerate(selected_windows):
        start = window["start"]
        end = window["end"]
        clip_video_frames = frames[start:end]
        clip_records = records[start:end]

        poses = [record["pose"] for record in clip_records]
        actions = [record["action_vec"] for record in clip_records]

        if any(pose is None for pose in poses):
            print(f"    [WARN] {stem} window {clip_idx}: unresolved invalid first frame, skipping")
            continue

        poses = np.array(poses, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        if is_static_clip(poses):
            continue

        intrinsics = np.tile(
            fov_to_intrinsics(default_fov, height, width),
            (clip_frames, 1)
        )
        action_mask = np.ones_like(actions, dtype=np.float32)

        clip_name = f"{stream_uid}_clip{clip_idx:04d}"
        clip_dir = os.path.join(output_dir, split, "clips", clip_name)
        os.makedirs(clip_dir, exist_ok=True)

        # Save video
        video_out = os.path.join(clip_dir, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_out, fourcc, target_fps, (width, height))
        for frame in clip_video_frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
        out.release()

        # Save first frame
        first_frame = cv2.resize(clip_video_frames[0], (width, height),
                                 interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(clip_dir, "image.jpg"), first_frame)

        # Save numpy arrays
        np.save(os.path.join(clip_dir, "poses.npy"), poses)
        np.save(os.path.join(clip_dir, "action.npy"), actions)
        np.save(os.path.join(clip_dir, "action_mask.npy"), action_mask)
        np.save(os.path.join(clip_dir, "intrinsics.npy"), intrinsics)

        # Prompt
        weapon = "rifle"
        if clip_records[0]["state"] is not None:
            weapon = clip_records[0]["state"]["action_dict"].get("weapon_slot", "rifle")

        if window["fire_with_enemy_frames"] > 0 or window["max_enemies_in_view"] >= 2:
            motion_phrase = "The player is in an active firefight with visible enemies, rapid camera motion, and sudden aim changes."
        elif window["enemy_visible_ratio"] >= 0.12 and window["close_enemy_ratio"] >= 0.05:
            motion_phrase = "The player is taking an aggressive duel with enemies repeatedly visible in front of the camera."
        elif window["event_score"] >= 4 or window["fire_frames"] > 0:
            motion_phrase = "The player is entering a tense combat sequence with noticeable camera motion and weapon handling."
        elif window["cam_p90"] >= 12 or window["rot_p90"] >= 18:
            motion_phrase = "The player is sprinting and turning quickly through the map with strong first-person camera motion."
        else:
            motion_phrase = "The player is moving quickly through the map with noticeable first-person camera motion."

        prompt = (
            f"First-person view of a competitive CS:GO match on {map_name}. "
            f"{motion_phrase} "
            f"The player is holding a {weapon}. "
            f"Photorealistic game rendering with detailed textures and lighting."
        )
        with open(os.path.join(clip_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        with open(os.path.join(clip_dir, "clip_stats.json"), "w") as f:
            json.dump(
                {
                    "selection_mode": selection_mode,
                    "source_id": stream.get("source_id"),
                    "stream_uid": stream_uid,
                    "visibility_mode": visibility_meta["visibility_mode"],
                    "visibility_target_mode": visibility_meta["visibility_target_mode"],
                    "depth_used_for_mining": visibility_meta["occlusion_enabled"],
                    "seg_used_for_mining": visibility_meta["seg_enabled"],
                    "depth_far_units": visibility_meta["depth_far_units"],
                    "start_index": start,
                    "end_index": end,
                    "motion_score": window["motion_score"],
                    "cam_p90": window["cam_p90"],
                    "rot_p90": window["rot_p90"],
                    "action_mean": window["action_mean"],
                    "event_score": window["event_score"],
                    "fire_frames": window["fire_frames"],
                    "onset_bonus": window["onset_bonus"],
                    "event_types": window["event_types"],
                    "valid_ratio": window["valid_ratio"],
                    "combat_score": window["combat_score"],
                    "enemy_projected_ratio": window["enemy_projected_ratio"],
                    "enemy_visible_ratio": window["enemy_visible_ratio"],
                    "enemy_occluded_ratio": window["enemy_occluded_ratio"],
                    "enemy_visibility_strength": window["enemy_visibility_strength"],
                    "enemy_occupancy_strength": window["enemy_occupancy_strength"],
                    "close_enemy_ratio": window["close_enemy_ratio"],
                    "center_enemy_ratio": window["center_enemy_ratio"],
                    "max_enemies_in_view": window["max_enemies_in_view"],
                    "mean_enemies_in_view": window["mean_enemies_in_view"],
                    "fire_with_enemy_frames": window["fire_with_enemy_frames"],
                    "fire_burst_count": window["fire_burst_count"],
                    "utility_frames": window["utility_frames"],
                },
                f,
                indent=2,
            )

        with open(os.path.join(clip_dir, "modality_flags.json"), "w") as f:
            json.dump(
                {
                    "source_domain": "csgo",
                    "source_id": stream.get("source_id"),
                    "stream_uid": stream_uid,
                    "has_action": True,
                    "has_pose": True,
                    "has_intrinsics": True,
                    "has_depth": False,
                    "has_segmentation": False,
                    "depth_used_for_mining": visibility_meta["occlusion_enabled"],
                    "seg_used_for_mining": visibility_meta["seg_enabled"],
                    "action_dim": int(actions.shape[-1]),
                    "selection_mode": selection_mode,
                    "visibility_mode": visibility_meta["visibility_mode"],
                    "visibility_target_mode": visibility_meta["visibility_target_mode"],
                },
                f,
                indent=2,
            )

        clips.append({
            "clip_name": clip_name,
            "clip_path": os.path.join(split, "clips", clip_name),
            "prompt": prompt,
            "split": split,
            "source_id": stream.get("source_id"),
            "stream_uid": stream_uid,
            "map": map_name,
            "episode_id": stream["episode_id"],
            "stem": stem,
            "num_frames": clip_frames,
            "motion_score": window["motion_score"],
            "combat_score": window["combat_score"],
            "cam_p90": window["cam_p90"],
            "rot_p90": window["rot_p90"],
            "action_mean": window["action_mean"],
            "event_score": window["event_score"],
            "fire_frames": window["fire_frames"],
            "onset_bonus": window["onset_bonus"],
            "enemy_projected_ratio": window["enemy_projected_ratio"],
            "enemy_visible_ratio": window["enemy_visible_ratio"],
            "enemy_occluded_ratio": window["enemy_occluded_ratio"],
            "enemy_visibility_strength": window["enemy_visibility_strength"],
            "enemy_occupancy_strength": window["enemy_occupancy_strength"],
            "close_enemy_ratio": window["close_enemy_ratio"],
            "center_enemy_ratio": window["center_enemy_ratio"],
            "max_enemies_in_view": window["max_enemies_in_view"],
            "mean_enemies_in_view": window["mean_enemies_in_view"],
            "fire_with_enemy_frames": window["fire_with_enemy_frames"],
            "fire_burst_count": window["fire_burst_count"],
            "utility_frames": window["utility_frames"],
            "selection_rank": clip_idx,
            "window_start": start,
            "window_end": end,
            "event_types": "|".join(window["event_types"]),
        })

    print(
        f"    Generated {len(clips)} clips "
        f"(candidates={len(candidates)}, selected={len(selected_windows)})"
    )
    return clips


# ============================================================
# Flash alpha check (run after full dataset download)
# ============================================================

def check_flash_alpha(input_dirs):
    """Scan all action JSONs for flash_alpha > 0 occurrences."""
    import glob
    total = 0
    flash_count = 0
    for d in input_dirs:
        for f in glob.glob(os.path.join(d, "train", "*", "*.json")):
            if any(k in f for k in ["episode_info", "video_manifest", "game_manifest", "seg_colormap"]):
                continue
            with open(f) as fh:
                data = json.load(fh)
            for frame in data:
                total += 1
                if frame.get("action", {}).get("flash_alpha", 0) > 0:
                    flash_count += 1
    print(f"flash_alpha > 0: {flash_count}/{total} frames ({100*flash_count/max(total,1):.2f}%)")
    return flash_count, total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess dust2-460 dataset for LingBot-World-Fast")
    parser.add_argument("--input_dirs", type=str, required=True,
                        help="Comma-separated data root directories")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_frames", type=int, default=81)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--stride", type=int, default=8,
                        help="Candidate window stride in downsampled frames")
    parser.add_argument("--default_fov", type=float, default=106.26)
    parser.add_argument("--top_k_per_stream", type=int, default=3,
                        help="Keep at most top-k aggressive clips per player stream")
    parser.add_argument("--post_freeze_buffer_frames", type=int, default=24,
                        help="Ignore windows that start too close to round_freeze_end")
    parser.add_argument("--min_valid_ratio", type=float, default=0.95,
                        help="Minimum ratio of alive/valid frames inside a clip")
    parser.add_argument("--min_score_percentile", type=float, default=70.0,
                        help="Per-stream percentile floor for combat-window selection")
    parser.add_argument("--min_absolute_score", type=float, default=6.0,
                        help="Absolute floor for combat-window selection")
    parser.add_argument("--min_window_separation", type=int, default=40,
                        help="Minimum start-index gap between selected windows")
    parser.add_argument("--min_enemy_visible_ratio", type=float, default=0.08,
                        help="Combat gate: minimum ratio of frames with at least one projected enemy in view")
    parser.add_argument("--min_close_enemy_ratio", type=float, default=0.03,
                        help="Combat gate: minimum ratio of frames with a nearby visible enemy")
    parser.add_argument("--min_fire_enemy_frames", type=int, default=1,
                        help="Combat gate: minimum number of frames with fire while an enemy is visible")
    parser.add_argument("--min_event_score", type=float, default=2.0,
                        help="Combat gate: minimum summed event score for event-driven windows")
    parser.add_argument("--occlusion_depth_tolerance", type=float, default=96.0,
                        help="Occlusion-aware mode: allow this many game units behind the local depth surface")
    parser.add_argument("--occlusion_patch_radius", type=int, default=2,
                        help="Occlusion-aware mode: sample a square depth patch with this radius around the projection")
    parser.add_argument("--seg_patch_radius", type=int, default=3,
                        help="Seg-aware mode: sample a square segmentation patch with this radius around the projection")
    parser.add_argument("--seg_min_dominant_ratio", type=float, default=0.12,
                        help="Seg-aware mode: minimum dominant-label coverage inside the segmentation patch")
    parser.add_argument("--seg_min_pixels", type=int, default=3,
                        help="Seg-aware mode: minimum dominant-label pixels required in the segmentation patch")
    parser.add_argument("--max_streams", type=int, default=0,
                        help="Debug option: only process the first N streams (0 = all)")
    parser.add_argument("--skip_episodes", type=str, default="")
    parser.add_argument("--val_episodes", type=str, default="",
                        help="Comma-separated episode IDs for validation")
    parser.add_argument("--check_flash", action="store_true",
                        help="Only scan for flash_alpha, don't preprocess")
    args = parser.parse_args()

    input_dirs = [d.strip() for d in args.input_dirs.split(",") if d.strip()]
    assert (args.clip_frames - 1) % 4 == 0, "clip_frames must be 4n+1"

    # Flash alpha check mode
    if args.check_flash:
        check_flash_alpha(input_dirs)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train", "clips"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val", "clips"), exist_ok=True)

    skip_episodes = [s.strip() for s in args.skip_episodes.split(",") if s.strip()]
    val_episodes = [s.strip() for s in args.val_episodes.split(",") if s.strip()]

    streams = find_player_streams(input_dirs, skip_episodes, val_episodes)
    if not streams:
        print("No player streams found!")
        sys.exit(1)

    if args.max_streams > 0:
        streams = streams[:args.max_streams]
        print(f"[DEBUG] Limiting preprocessing to {len(streams)} streams")

    all_clips = []
    for stream in streams:
        clips = process_stream(
            stream, args.output_dir,
            clip_frames=args.clip_frames,
            target_fps=args.target_fps,
            height=args.height,
            width=args.width,
            stride=args.stride,
            default_fov=args.default_fov,
            top_k_per_stream=args.top_k_per_stream,
            post_freeze_buffer_frames=args.post_freeze_buffer_frames,
            min_valid_ratio=args.min_valid_ratio,
            min_score_percentile=args.min_score_percentile,
            min_absolute_score=args.min_absolute_score,
            min_window_separation=args.min_window_separation,
            min_enemy_visible_ratio=args.min_enemy_visible_ratio,
            min_close_enemy_ratio=args.min_close_enemy_ratio,
            min_fire_enemy_frames=args.min_fire_enemy_frames,
            min_event_score=args.min_event_score,
            occlusion_depth_tolerance=args.occlusion_depth_tolerance,
            occlusion_patch_radius=args.occlusion_patch_radius,
            seg_patch_radius=args.seg_patch_radius,
            seg_min_dominant_ratio=args.seg_min_dominant_ratio,
            seg_min_pixels=args.seg_min_pixels,
        )
        all_clips.extend(clips)

    # Write metadata CSVs
    for split in ["train", "val"]:
        split_clips = [c for c in all_clips if c["split"] == split]
        csv_path = os.path.join(args.output_dir, f"metadata_{split}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "prompt", "video", "clip_path", "source_id", "stream_uid", "map", "episode_id", "stem", "num_frames"
            ])
            writer.writeheader()
            for clip in split_clips:
                writer.writerow({
                    "prompt": clip["prompt"],
                    "video": os.path.join(clip["clip_path"], "video.mp4"),
                    "clip_path": clip["clip_path"],
                    "source_id": clip.get("source_id"),
                    "stream_uid": clip.get("stream_uid"),
                    "map": clip["map"],
                    "episode_id": clip["episode_id"],
                    "stem": clip["stem"],
                    "num_frames": clip["num_frames"],
                })
        print(f"{split}: {len(split_clips)} clips -> {csv_path}")

        audit_path = os.path.join(args.output_dir, f"metadata_{split}_motion_audit.csv")
        with open(audit_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "clip_name", "clip_path", "source_id", "stream_uid", "stem", "episode_id", "map", "num_frames",
                "motion_score", "combat_score", "cam_p90", "rot_p90", "action_mean",
                "event_score", "fire_frames", "onset_bonus", "enemy_projected_ratio",
                "enemy_visible_ratio", "enemy_occluded_ratio", "enemy_visibility_strength",
                "enemy_occupancy_strength",
                "close_enemy_ratio", "center_enemy_ratio", "max_enemies_in_view",
                "mean_enemies_in_view", "fire_with_enemy_frames", "fire_burst_count",
                "utility_frames", "selection_rank", "window_start", "window_end",
                "event_types"
            ])
            writer.writeheader()
            for clip in split_clips:
                writer.writerow({
                    "clip_name": clip["clip_name"],
                    "clip_path": clip["clip_path"],
                    "source_id": clip.get("source_id"),
                    "stream_uid": clip.get("stream_uid"),
                    "stem": clip["stem"],
                    "episode_id": clip["episode_id"],
                    "map": clip["map"],
                    "num_frames": clip["num_frames"],
                    "motion_score": clip["motion_score"],
                    "combat_score": clip["combat_score"],
                    "cam_p90": clip["cam_p90"],
                    "rot_p90": clip["rot_p90"],
                    "action_mean": clip["action_mean"],
                    "event_score": clip["event_score"],
                    "fire_frames": clip["fire_frames"],
                    "onset_bonus": clip["onset_bonus"],
                    "enemy_projected_ratio": clip["enemy_projected_ratio"],
                    "enemy_visible_ratio": clip["enemy_visible_ratio"],
                    "enemy_occluded_ratio": clip["enemy_occluded_ratio"],
                    "enemy_visibility_strength": clip["enemy_visibility_strength"],
                    "enemy_occupancy_strength": clip["enemy_occupancy_strength"],
                    "close_enemy_ratio": clip["close_enemy_ratio"],
                    "center_enemy_ratio": clip["center_enemy_ratio"],
                    "max_enemies_in_view": clip["max_enemies_in_view"],
                    "mean_enemies_in_view": clip["mean_enemies_in_view"],
                    "fire_with_enemy_frames": clip["fire_with_enemy_frames"],
                    "fire_burst_count": clip["fire_burst_count"],
                    "utility_frames": clip["utility_frames"],
                    "selection_rank": clip["selection_rank"],
                    "window_start": clip["window_start"],
                    "window_end": clip["window_end"],
                    "event_types": clip["event_types"],
                })
        print(f"{split}: {len(split_clips)} clips -> {audit_path}")

    print(f"\nTotal: {len(all_clips)} clips")
    print(f"  Train: {len([c for c in all_clips if c['split'] == 'train'])}")
    print(f"  Val:   {len([c for c in all_clips if c['split'] == 'val'])}")


if __name__ == "__main__":
    main()
