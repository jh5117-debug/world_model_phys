from __future__ import annotations

from pathlib import Path

from physical_consistency.datasets.test_subset import (
    build_csgo_test_subset,
    select_diverse_test_rows,
)
from physical_consistency.datasets.manifest_builder import materialize_dataset_view


def test_select_diverse_test_rows_preserves_sample_count_and_order():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "prompt": f"First-person view holding a {'AK-47' if idx % 2 else 'Knife'}.",
                "video": f"val/clips/clip_{idx:04d}/video.mp4",
                "clip_path": f"val/clips/clip_{idx:04d}",
                "map": "de_dust2",
                "episode_id": "28",
                "stem": f"stem_{idx//2}",
                "num_frames": "81",
            }
        )
    selected = select_diverse_test_rows(rows, sample_count=5, seed=0)
    assert len(selected) == 5
    original_positions = [rows.index(row) for row in selected]
    assert original_positions == sorted(original_positions)


def test_build_csgo_test_subset_rewrites_paths_and_links(tmp_path: Path):
    dataset_dir = tmp_path / "processed_csgo_v3"
    val_clips = dataset_dir / "val" / "clips"
    val_clips.mkdir(parents=True)

    rows = []
    for idx in range(4):
        clip_name = f"clip_{idx:04d}"
        clip_dir = val_clips / clip_name
        clip_dir.mkdir()
        (clip_dir / "video.mp4").write_text("video")
        (clip_dir / "image.jpg").write_text("image")
        rows.append(
            {
                "prompt": f"First-person view holding a {'AK-47' if idx % 2 else 'Knife'}.",
                "video": f"val/clips/{clip_name}/video.mp4",
                "clip_path": f"val/clips/{clip_name}",
                "map": "de_dust2",
                "episode_id": "28",
                "stem": f"stem_{idx}",
                "num_frames": "81",
            }
        )

    from physical_consistency.common.io import read_csv_rows, write_csv_rows

    write_csv_rows(dataset_dir / "metadata_val.csv", rows, list(rows[0].keys()))

    result = build_csgo_test_subset(dataset_dir, sample_count=2, seed=0, overwrite=True)

    assert result.selected_rows == 2
    metadata_rows = read_csv_rows(dataset_dir / "metadata_test.csv")
    assert len(metadata_rows) == 2
    assert all(row["clip_path"].startswith("test/clips/") for row in metadata_rows)
    assert all(row["video"].startswith("test/clips/") for row in metadata_rows)
    for row in metadata_rows:
        assert (dataset_dir / row["clip_path"]).is_symlink()


def test_materialize_dataset_view_keeps_test_split_symlink(tmp_path: Path):
    dataset_dir = tmp_path / "processed_csgo_v3"
    for split in ["train", "val", "test"]:
        (dataset_dir / split).mkdir(parents=True)
    rows = [
        {
            "prompt": "demo",
            "video": "test/clips/clip_0000/video.mp4",
            "clip_path": "test/clips/clip_0000",
            "map": "de_dust2",
            "episode_id": "28",
            "stem": "stem_0",
            "num_frames": "81",
        }
    ]
    from physical_consistency.common.io import write_csv_rows

    write_csv_rows(dataset_dir / "metadata_train.csv", rows, list(rows[0].keys()))
    write_csv_rows(dataset_dir / "metadata_test.csv", rows, list(rows[0].keys()))
    write_csv_rows(dataset_dir / "metadata_val.csv", rows, list(rows[0].keys()))
    write_csv_rows(tmp_path / "manifest.csv", rows, list(rows[0].keys()))

    out_dir = materialize_dataset_view(dataset_dir, tmp_path / "manifest.csv", tmp_path / "view")
    assert (out_dir / "test").is_symlink()
