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


def test_build_videophy2_input_csv_manifest_video_column_mode(tmp_path):
    candidate_video = tmp_path / "generated" / "clip_0001_gen.mp4"
    candidate_video.parent.mkdir(parents=True)
    candidate_video.write_text("x", encoding="utf-8")

    manifest_csv = tmp_path / "generated_videos.csv"
    manifest_csv.write_text(
        "candidate_videopath,prompt\n"
        f"{candidate_video},generated prompt\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "videophy2.csv"
    build_videophy2_input_csv(
        manifest_csv=manifest_csv,
        video_source_root=tmp_path,
        output_csv=output_csv,
        task="sa",
        source_mode="manifest_video_column",
        manifest_video_column="candidate_videopath",
        manifest_caption_column="prompt",
    )

    content = output_csv.read_text(encoding="utf-8")
    assert "videopath,caption" in content
    assert str(candidate_video) in content
    assert "generated prompt" in content
