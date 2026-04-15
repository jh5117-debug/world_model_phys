import csv
from pathlib import Path

from physical_consistency.common.io import read_csv_rows, write_csv_rows
from physical_consistency.datasets.manifest_builder import build_fixed_manifest, materialize_dataset_view


def test_build_fixed_manifest_is_deterministic(tmp_path):
    metadata = tmp_path / "metadata_val.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_path", "prompt"])
        writer.writeheader()
        for index in range(10):
            writer.writerow({"clip_path": f"val/clips/clip_{index}", "prompt": f"prompt {index}"})

    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    build_fixed_manifest(metadata, out_a, sample_count=4, seed=42)
    build_fixed_manifest(metadata, out_b, sample_count=4, seed=42)
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_materialize_dataset_view_keeps_direct_split_symlink_when_media_exists(tmp_path: Path):
    dataset_dir = tmp_path / "processed_csgo_v3"
    val_clip_dir = dataset_dir / "val" / "clips" / "clip_0000"
    val_clip_dir.mkdir(parents=True)
    (val_clip_dir / "image.jpg").write_text("image", encoding="utf-8")
    rows = [{"clip_path": "val/clips/clip_0000", "prompt": "prompt 0"}]
    write_csv_rows(dataset_dir / "metadata_train.csv", rows, list(rows[0].keys()))
    write_csv_rows(dataset_dir / "metadata_val.csv", rows, list(rows[0].keys()))
    manifest_path = tmp_path / "manifest.csv"
    write_csv_rows(manifest_path, rows, list(rows[0].keys()))

    out_dir = materialize_dataset_view(dataset_dir, manifest_path, tmp_path / "view")

    assert (out_dir / "val").is_symlink()
    assert (out_dir / "val" / "clips" / "clip_0000" / "image.jpg").exists()
    written_rows = read_csv_rows(out_dir / "metadata_val.csv")
    assert written_rows[0]["clip_path"] == "val/clips/clip_0000"


def test_materialize_dataset_view_falls_back_to_other_split_clip_media(tmp_path: Path):
    dataset_dir = tmp_path / "processed_csgo_v3"
    train_clip_dir = dataset_dir / "train" / "clips" / "clip_0000"
    train_clip_dir.mkdir(parents=True)
    (train_clip_dir / "image.jpg").write_text("image", encoding="utf-8")
    rows = [{"clip_path": "val/clips/clip_0000", "prompt": "prompt 0"}]
    write_csv_rows(dataset_dir / "metadata_train.csv", rows, list(rows[0].keys()))
    write_csv_rows(dataset_dir / "metadata_val.csv", rows, list(rows[0].keys()))
    manifest_path = tmp_path / "manifest.csv"
    write_csv_rows(manifest_path, rows, list(rows[0].keys()))

    out_dir = materialize_dataset_view(dataset_dir, manifest_path, tmp_path / "view")

    target_clip_dir = out_dir / "val" / "clips" / "clip_0000"
    assert not (out_dir / "val").is_symlink()
    assert target_clip_dir.is_symlink()
    assert target_clip_dir.resolve() == train_clip_dir.resolve()
    assert (target_clip_dir / "image.jpg").exists()
