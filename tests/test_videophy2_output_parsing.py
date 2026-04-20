from pathlib import Path
import sys


VIDEOPHY_DIR = Path(__file__).resolve().parents[1] / "third_party" / "videophy" / "VIDEOPHY2"
if str(VIDEOPHY_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOPHY_DIR))

from output_parsing import extract_score_text_candidates, parse_score_from_output
from physical_consistency.eval.videophy2 import summarize_videophy2_outputs, write_videophy2_summary


def test_parse_videophy_scores_accepts_clean_single_scores():
    assert parse_score_from_output("1", task="sa") == 1
    assert parse_score_from_output("score: 4.", task="sa") == 4
    assert parse_score_from_output("one", task="pc") == 1
    assert parse_score_from_output("2", task="rule") == 2


def test_parse_videophy_scores_rejects_malformed_repeated_tokens():
    assert parse_score_from_output("100000", task="sa") is None
    assert parse_score_from_output("1.1.1.", task="pc") is None
    assert parse_score_from_output("", task="sa") is None


def test_parse_videophy_scores_respects_task_ranges():
    assert parse_score_from_output("0", task="sa") is None
    assert parse_score_from_output("0", task="rule") == 0
    assert parse_score_from_output("5", task="rule") is None


def test_parse_videophy_scores_uses_only_assistant_segment_when_present():
    output = (
        'Human: Does this video match the description? '
        'Please rate the video on a scale from 1 to 5.\n'
        'AI: 4'
    )
    assert parse_score_from_output(output, task="sa") == 4


def test_extract_score_text_candidates_prioritizes_assistant_suffix():
    candidates = extract_score_text_candidates("Human: score from 1 to 5\nAI: four\n")
    assert candidates[0] == "four"


def test_summarize_videophy2_outputs_prefers_choice_score(tmp_path):
    sa_csv = tmp_path / "output_sa.csv"
    pc_csv = tmp_path / "output_pc.csv"
    sa_csv.write_text(
        "videopath,score,choice_score\n"
        "a.mp4,4,5.0\n"
        "b.mp4,3,4.0\n",
        encoding="utf-8",
    )
    pc_csv.write_text(
        "videopath,score,choice_score\n"
        "a.mp4,3,4.0\n"
        "b.mp4,3,5.0\n",
        encoding="utf-8",
    )

    summary = summarize_videophy2_outputs(sa_csv, pc_csv)

    assert summary["sa_mean"] == 4.5
    assert summary["pc_mean"] == 4.5
    assert summary["joint"] == 1.0
    assert summary["count"] == 2


def test_write_videophy2_summary_counts_samples_not_seeds(tmp_path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()
    (seed_dir / "output_sa.csv").write_text(
        "videopath,choice_score\n"
        "a.mp4,5.0\n"
        "b.mp4,4.0\n",
        encoding="utf-8",
    )
    (seed_dir / "output_pc.csv").write_text(
        "videopath,choice_score\n"
        "a.mp4,4.0\n"
        "b.mp4,3.0\n",
        encoding="utf-8",
    )

    summary_path = write_videophy2_summary(tmp_path)
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))

    assert summary["seeds"][0]["count"] == 2
    assert summary["seeds"][0]["means"]["sa_mean"]["count"] == 2
    assert summary["means"]["sa_mean"]["mean"] == 4.5
    assert summary["means"]["sa_mean"]["count"] == 2
    assert summary["means"]["pc_mean"]["mean"] == 3.5
    assert summary["means"]["joint"]["mean"] == 0.5
