import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from physical_consistency.stages.stage1_physinone_cam.preprocess_moving_act import (
    MovingActPreprocessArgs,
    _raw_pose_delta_actions,
    run_preprocess,
)


def _jpg_bytes(value: int) -> bytes:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _write_moving_camera_zip(path: Path) -> None:
    trajectory = "ToyMove__bg001__abcd_trajectory"
    frames = []
    with zipfile.ZipFile(path, "w") as zf:
        for index in range(4):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(index)
            angle = 0.05 * float(index)
            pose[:3, :3] = np.asarray(
                [
                    [np.cos(angle), 0.0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0.0, np.cos(angle)],
                ],
                dtype=np.float32,
            )
            frames.append(
                {
                    "frame": index,
                    "transform_matrix": pose.tolist(),
                    "file_path": f"CineCamera_Moving/rgb/{index:04d}",
                    "time": float(index) / 30.0,
                    "time_abs": float(index) / 30.0,
                }
            )
            zf.writestr(f"{trajectory}/CineCamera_Moving/rgb/{index:04d}.jpg", _jpg_bytes(32 + index))
        zf.writestr(
            f"{trajectory}/blender_CineCamera_Moving.json",
            json.dumps(
                {
                    "camera_angle_x": 1.0,
                    "img_h": 8,
                    "img_w": 8,
                    "trajectory_name": trajectory,
                    "total_frames": 4,
                    "fps": 30,
                    "frames": frames,
                }
            ),
        )


def test_raw_pose_delta_actions_capture_motion():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0)
    poses[1, 2, 3] = 0.5
    poses[2, 2, 3] = 1.0

    actions = _raw_pose_delta_actions(poses)

    assert actions.shape == (3, 4)
    assert np.allclose(actions[:, 0], [0.5, 0.5, 0.5])


def test_moving_act_preprocess_writes_actions_and_framewise_poses(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_moving_camera_zip(raw / "ToyMove__bg001__abcd_trajectory.zip")
    output = tmp_path / "PhysInOne_moving_act"

    summary = run_preprocess(
        MovingActPreprocessArgs(
            input_root=str(raw),
            output_dir=str(output),
            clip_frames=4,
            sampling_mode="uniform_single",
            window_stride=4,
            output_height=8,
            output_width=8,
            target_fps=16,
            default_camera_angle_x=1.0,
            val_ratio=0.0,
            seed=0,
            max_zips=0,
            action_normalize_percentile=95.0,
            min_action_scale=1.0e-6,
        )
    )

    clip_dir = output / "train" / "clips" / "ToyMove__bg001__abcd_trajectory__CineCamera_Moving_clip0000"
    assert summary["train_clip_count"] == 1
    assert (clip_dir / "video.mp4").exists()
    assert (clip_dir / "image.jpg").exists()
    poses = np.load(clip_dir / "poses.npy")
    actions = np.load(clip_dir / "action.npy")
    assert poses.shape == (4, 4, 4)
    assert actions.shape == (4, 4)
    assert float(np.abs(poses - poses[:1]).max()) > 0.0
    assert float(np.abs(actions).max()) > 0.0
