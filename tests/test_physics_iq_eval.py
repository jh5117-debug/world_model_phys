from __future__ import annotations

import csv

import cv2
import numpy as np

from physical_consistency.eval.physics_iq import (
    build_physics_iq_input_csv,
    evaluate_video_pair,
)


def _write_test_video(path, *, fps=8, frame_count=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (4 + idx, 12), (20 + idx, 28), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_build_physics_iq_input_csv_dataset_clip_mode(tmp_path):
    dataset_dir = tmp_path / "Dataset" / "processed_csgo_v3"
    clip_dir = dataset_dir / "val" / "clips" / "clip_0001"
    _write_test_video(clip_dir / "video.mp4")

    manifest_csv = tmp_path / "metadata_val.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_path", "prompt"])
        writer.writeheader()
        writer.writerow({"clip_path": "val/clips/clip_0001", "prompt": "test prompt"})

    output_csv = tmp_path / "pairs.csv"
    build_physics_iq_input_csv(
        manifest_csv=manifest_csv,
        reference_source_root=dataset_dir,
        candidate_source_root=dataset_dir,
        output_csv=output_csv,
        reference_source_mode="dataset_clip",
        candidate_source_mode="dataset_clip",
    )

    content = output_csv.read_text(encoding="utf-8")
    assert "reference_videopath" in content
    assert str(clip_dir / "video.mp4") in content
    assert "test prompt" in content


def test_evaluate_video_pair_identity_score_is_high(tmp_path):
    video_path = tmp_path / "identity.mp4"
    _write_test_video(video_path)

    result = evaluate_video_pair(
        reference_videopath=video_path,
        candidate_videopath=video_path,
        compare_seconds=5.0,
        sample_frames=12,
        resize_divisor=2,
        mask_threshold=10,
    )

    assert result["compare_frame_count"] >= 2
    assert result["mse_mean"] == 0.0
    assert result["spatiotemporal_iou_mean"] == 1.0
    assert result["spatial_iou"] == 1.0
    assert result["weighted_spatial_iou"] == 1.0
    assert result["physics_iq_style_score"] >= 99.99
