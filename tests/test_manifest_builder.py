import csv

from physical_consistency.datasets.manifest_builder import build_fixed_manifest


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
