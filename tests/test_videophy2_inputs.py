from physical_consistency.eval.videophy2 import build_videophy2_input_csv


def test_build_videophy2_input_csv_dataset_clip_mode(tmp_path):
    dataset_dir = tmp_path / "Dataset" / "processed_csgo_v3"
    clip_dir = dataset_dir / "val" / "clips" / "clip_0001"
    clip_dir.mkdir(parents=True)
    (clip_dir / "video.mp4").write_text("x", encoding="utf-8")

    manifest_csv = tmp_path / "metadata_val.csv"
    manifest_csv.write_text(
        "clip_path,prompt\n"
        "val/clips/clip_0001,test prompt\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "videophy2.csv"
    build_videophy2_input_csv(
        manifest_csv=manifest_csv,
        video_source_root=dataset_dir,
        output_csv=output_csv,
        task="sa",
        source_mode="dataset_clip",
    )

    content = output_csv.read_text(encoding="utf-8")
    assert "videopath,caption" in content
    assert str(clip_dir / "video.mp4") in content
    assert "test prompt" in content
