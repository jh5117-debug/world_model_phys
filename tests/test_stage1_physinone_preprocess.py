from physical_consistency.stages.stage1_physinone_cam.preprocess import (
    _contiguous_window_starts,
    _slice_members_into_windows,
)


def test_contiguous_window_starts_cover_tail():
    starts = _contiguous_window_starts(frame_count=90, clip_frames=75, window_stride=75)
    assert starts == [0, 15]


def test_slice_members_into_windows_splits_long_clip_and_covers_tail():
    members = [f"frame_{idx:04d}.jpg" for idx in range(90)]

    windows = _slice_members_into_windows(
        members,
        clip_frames=75,
        sampling_mode="contiguous_windows",
        window_stride=75,
    )

    assert len(windows) == 2
    first_members, first_indices = windows[0]
    second_members, second_indices = windows[1]
    assert len(first_members) == 75
    assert first_indices[0] == 0
    assert first_indices[-1] == 74
    assert len(second_members) == 75
    assert second_indices[0] == 15
    assert second_indices[-1] == 89


def test_slice_members_into_windows_pads_short_clip_with_last_frame():
    members = [f"frame_{idx:04d}.jpg" for idx in range(3)]

    windows = _slice_members_into_windows(
        members,
        clip_frames=5,
        sampling_mode="contiguous_windows",
        window_stride=5,
    )

    assert len(windows) == 1
    sampled_members, sampled_indices = windows[0]
    assert sampled_indices == [0, 1, 2, 2, 2]
    assert sampled_members[-1] == members[-1]
