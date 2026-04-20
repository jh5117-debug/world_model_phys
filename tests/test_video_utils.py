from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from physical_consistency.eval.lingbot_generate import _generated_manifest_row
from physical_consistency.eval.video_utils import (
    VideoValidationError,
    validate_video_readable,
    write_side_by_side_video,
)


def _write_tiny_video(path, *, color: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (16, 12),
    )
    assert writer.isOpened()
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:, :] = color
    for _ in range(3):
        writer.write(frame)
    writer.release()


def test_validate_video_readable_rejects_broken_mp4(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"0" * 48)

    with pytest.raises(VideoValidationError):
        validate_video_readable(broken, min_frames=1)


def test_write_side_by_side_video_outputs_readable_preview(tmp_path):
    reference = tmp_path / "reference.mp4"
    candidate = tmp_path / "candidate.mp4"
    output = tmp_path / "comparison.mp4"
    _write_tiny_video(reference, color=(255, 0, 0))
    _write_tiny_video(candidate, color=(0, 255, 0))

    write_side_by_side_video(
        reference_videopath=reference,
        candidate_videopath=candidate,
        output_path=output,
        max_frames=3,
        height=12,
        width=16,
    )

    assert output.exists()
    validate_video_readable(output, min_frames=1)


def test_generated_manifest_row_keeps_candidate_when_reference_is_missing(tmp_path):
    candidate = tmp_path / "candidate.mp4"
    comparison = tmp_path / "comparison.mp4"
    _write_tiny_video(candidate, color=(0, 255, 0))
    cfg = SimpleNamespace(
        dataset_dir=str(tmp_path / "dataset"),
        video_filename="video.mp4",
        frame_num=3,
        height=12,
        width=16,
    )

    row = _generated_manifest_row(
        cfg=cfg,
        row={"clip_path": "val/clips/missing_clip", "prompt": "test"},
        clip_name="missing_clip",
        candidate_videopath=candidate,
        comparison_path=comparison,
    )

    assert row["candidate_videopath"] == str(candidate.resolve())
    assert row["comparison_videopath"] == ""
    assert not comparison.exists()


def test_generated_manifest_row_writes_comparison_from_manifest_reference(tmp_path):
    reference = tmp_path / "reference.mp4"
    candidate = tmp_path / "candidate.mp4"
    comparison = tmp_path / "comparison.mp4"
    _write_tiny_video(reference, color=(255, 0, 0))
    _write_tiny_video(candidate, color=(0, 255, 0))
    cfg = SimpleNamespace(
        dataset_dir=str(tmp_path / "dataset"),
        video_filename="video.mp4",
        frame_num=3,
        height=12,
        width=16,
    )

    row = _generated_manifest_row(
        cfg=cfg,
        row={"clip_path": "val/clips/example", "prompt": "test", "reference_videopath": str(reference)},
        clip_name="example",
        candidate_videopath=candidate,
        comparison_path=comparison,
    )

    assert row["reference_videopath"] == str(reference.resolve())
    assert row["candidate_videopath"] == str(candidate.resolve())
    assert row["comparison_videopath"] == str(comparison.resolve())
    validate_video_readable(comparison, min_frames=1)
