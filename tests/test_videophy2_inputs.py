import sys

from physical_consistency.eval import videophy2 as videophy2_module
from physical_consistency.eval.videophy2 import build_videophy2_input_csv, run_videophy2_for_seed


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


def test_run_videophy2_for_seed_uses_current_python(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo" / "VIDEOPHY2"
    repo_dir.mkdir(parents=True)
    (repo_dir / "inference.py").write_text("print('ok')\n", encoding="utf-8")

    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    candidate_video = tmp_path / "videos" / "clip_0001_gen.mp4"
    candidate_video.parent.mkdir(parents=True)
    candidate_video.write_text("x", encoding="utf-8")

    manifest_csv = tmp_path / "generated_videos.csv"
    manifest_csv.write_text(
        "candidate_videopath,prompt\n"
        f"{candidate_video},generated prompt\n",
        encoding="utf-8",
    )

    calls = []

    def _fake_run_command(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(videophy2_module, "run_command", _fake_run_command)

    run_videophy2_for_seed(
        repo_dir=str(tmp_path / "repo"),
        checkpoint_dir=str(checkpoint_dir),
        manifest_csv=str(manifest_csv),
        video_source_root=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        batch_size=4,
        seed=0,
        task_modes=["sa"],
        source_mode="manifest_video_column",
        manifest_video_column="candidate_videopath",
        manifest_caption_column="prompt",
    )

    assert calls
    assert calls[0][0][0] == sys.executable
