from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_trd_snapshot_tests.py"
    spec = importlib.util.spec_from_file_location("run_trd_snapshot_tests", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materialize_clean_outputs_copies_videos_and_merges_videophy_scores(tmp_path):
    runner = _load_runner_module()
    result_root = tmp_path / "tets_result"
    result_dir = result_root / "20260417_000000__exp__epoch_1"
    experiment_name = "exp_epoch_1_20260417_000000"
    video_dir = result_dir / "runs" / "eval" / experiment_name / "seed_42" / "csgo_metrics" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "clip_a_gen.mp4").write_bytes(b"video-a")
    (video_dir / "clip_b_gen.mp4").write_bytes(b"video-b")

    seed_dir = result_dir / "runs" / "eval" / "videophy2" / experiment_name / "seed_42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "output_sa.csv").write_text(
        "videopath,score\n/path/clip_a_gen.mp4,4.5\n/path/clip_b_gen.mp4,3.0\n",
        encoding="utf-8",
    )
    (seed_dir / "output_pc.csv").write_text(
        "videopath,score\n/path/clip_a_gen.mp4,4.0\n/path/clip_b_gen.mp4,5.0\n",
        encoding="utf-8",
    )
    (result_dir / "videophy2_summary.md").write_text("summary\n", encoding="utf-8")

    clean_dirs = runner._materialize_clean_outputs(
        result_root=result_root,
        result_dir=result_dir,
        timestamp="20260417_000000",
        checkpoint_label="epoch_1",
        experiment_name=experiment_name,
        seeds=[42],
    )

    assert len(clean_dirs) == 2
    for clean_dir in [result_dir / "clean", result_root / "clean_results" / "20260417_000000" / "epoch_1"]:
        assert (clean_dir / "videos" / "clip_a_gen.mp4").read_bytes() == b"video-a"
        assert (clean_dir / "videos" / "clip_b_gen.mp4").read_bytes() == b"video-b"
        scores = (clean_dir / "videophy2_scores_seed_42.csv").read_text(encoding="utf-8")
        assert "clip_a,/path/clip_a_gen.mp4,4.5000,4.0000,1" in scores
        assert "clip_b,/path/clip_b_gen.mp4,3.0000,5.0000,0" in scores
        assert (clean_dir / "videophy2_summary.md").read_text(encoding="utf-8") == "summary\n"
