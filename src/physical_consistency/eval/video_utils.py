"""Video validation and paired preview helpers for generated eval clips."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from physical_consistency.common.io import ensure_dir


class VideoValidationError(RuntimeError):
    """Raised when a generated video cannot be decoded enough for eval."""


def validate_video_readable(video_path: str | Path, *, min_frames: int = 1) -> int:
    """Validate that a video exists and yields at least ``min_frames`` frames."""
    path = Path(video_path)
    if not path.exists():
        raise VideoValidationError(f"Video does not exist: {path}")
    if path.stat().st_size <= 0:
        raise VideoValidationError(f"Video is empty: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoValidationError(f"Video cannot be opened: {path}")
        count = 0
        while count < max(1, min_frames):
            ok, _frame = cap.read()
            if not ok:
                break
            count += 1
        if count < max(1, min_frames):
            raise VideoValidationError(
                f"Video has only {count} readable frames; expected at least {min_frames}: {path}"
            )
        return count
    finally:
        cap.release()


def _resize_frame(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _safe_fps(cap: cv2.VideoCapture, *, fallback: float = 16.0) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        return fallback
    return fps


def write_side_by_side_video(
    *,
    reference_videopath: str | Path,
    candidate_videopath: str | Path,
    output_path: str | Path,
    max_frames: int,
    height: int,
    width: int,
) -> Path:
    """Write a left=GT/right=generated preview video."""
    reference_path = Path(reference_videopath)
    candidate_path = Path(candidate_videopath)
    validate_video_readable(reference_path, min_frames=1)
    validate_video_readable(candidate_path, min_frames=1)

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    ref_cap = cv2.VideoCapture(str(reference_path))
    gen_cap = cv2.VideoCapture(str(candidate_path))
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        fps = _safe_fps(gen_cap, fallback=_safe_fps(ref_cap))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height),
        )
        if not writer.isOpened():
            raise VideoValidationError(f"Could not open VideoWriter for {out_path}")

        while written < max(1, max_frames):
            ok_ref, ref_frame = ref_cap.read()
            ok_gen, gen_frame = gen_cap.read()
            if not ok_ref or not ok_gen:
                break
            ref_frame = _resize_frame(ref_frame, height=height, width=width)
            gen_frame = _resize_frame(gen_frame, height=height, width=width)
            writer.write(np.concatenate([ref_frame, gen_frame], axis=1))
            written += 1
    finally:
        ref_cap.release()
        gen_cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise VideoValidationError(
            f"Could not write side-by-side preview for {candidate_path}; no paired frames decoded."
        )
    validate_video_readable(out_path, min_frames=1)
    return out_path


def _draw_label(frame: np.ndarray, label: str, *, x_offset: int, header_height: int) -> None:
    cv2.putText(
        frame,
        label,
        (x_offset + 18, max(28, header_height - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_labeled_side_by_side_video(
    *,
    left_videopath: str | Path,
    right_videopath: str | Path,
    output_path: str | Path,
    left_label: str,
    right_label: str,
    max_frames: int,
    height: int,
    width: int,
) -> Path:
    """Write a labeled left/right preview video."""
    left_path = Path(left_videopath)
    right_path = Path(right_videopath)
    validate_video_readable(left_path, min_frames=1)
    validate_video_readable(right_path, min_frames=1)

    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    writer: cv2.VideoWriter | None = None
    written = 0
    header_height = 44
    try:
        fps = _safe_fps(right_cap, fallback=_safe_fps(left_cap))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2, height + header_height),
        )
        if not writer.isOpened():
            raise VideoValidationError(f"Could not open VideoWriter for {out_path}")

        while written < max(1, max_frames):
            ok_left, left_frame = left_cap.read()
            ok_right, right_frame = right_cap.read()
            if not ok_left or not ok_right:
                break
            left_frame = _resize_frame(left_frame, height=height, width=width)
            right_frame = _resize_frame(right_frame, height=height, width=width)
            body = np.concatenate([left_frame, right_frame], axis=1)
            header = np.full((header_height, width * 2, 3), 24, dtype=np.uint8)
            _draw_label(header, left_label, x_offset=0, header_height=header_height)
            _draw_label(header, right_label, x_offset=width, header_height=header_height)
            writer.write(np.concatenate([header, body], axis=0))
            written += 1
    finally:
        left_cap.release()
        right_cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise VideoValidationError(
            f"Could not write labeled side-by-side preview for {left_path} and {right_path}."
        )
    validate_video_readable(out_path, min_frames=1)
    return out_path
