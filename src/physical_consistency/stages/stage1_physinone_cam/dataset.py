"""Processed PhysInOne camera-only dataset loader."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PhysInOneCamDataset(Dataset):
    """Load standardized PhysInOne camera clips for pure Stage-1 training."""

    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        num_frames: int = 81,
        height: int = 480,
        width: int = 480,
        repeat: int = 1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.repeat = int(repeat)

        csv_path = os.path.join(dataset_dir, f"metadata_{split}.csv")
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            self.samples = list(csv.DictReader(handle))
        if split == "train" and not self.samples:
            raise ValueError(f"No samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        import cv2

        sample = self.samples[index % len(self.samples)]
        clip_dir = os.path.join(self.dataset_dir, sample["clip_path"])
        video_path = os.path.join(clip_dir, "video.mp4")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode any frame from {video_path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        video_tensor = torch.stack(frames, dim=1)

        poses = np.load(os.path.join(clip_dir, "poses.npy"))
        intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))
        return {
            "clip_name": os.path.basename(clip_dir),
            "video": video_tensor,
            "prompt": sample["prompt"],
            "poses": torch.from_numpy(self._pad_or_truncate(poses)).float(),
            "intrinsics": torch.from_numpy(self._pad_or_truncate(intrinsics)).float(),
            "source_height": int(sample.get("source_height", self.height) or self.height),
            "source_width": int(sample.get("source_width", self.width) or self.width),
            "camera_id": sample.get("camera_id", ""),
            "trajectory_name": sample.get("trajectory_name", ""),
        }

    def _pad_or_truncate(self, array: np.ndarray) -> np.ndarray:
        if len(array) >= self.num_frames:
            return array[: self.num_frames]
        rep_shape = (self.num_frames - len(array),) + (1,) * (array.ndim - 1)
        pad = np.tile(array[-1:], rep_shape)
        return np.concatenate([array, pad], axis=0)
