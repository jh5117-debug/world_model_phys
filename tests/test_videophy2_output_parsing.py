from pathlib import Path
import sys


VIDEOPHY_DIR = Path(__file__).resolve().parents[1] / "third_party" / "videophy" / "VIDEOPHY2"
if str(VIDEOPHY_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOPHY_DIR))

from output_parsing import extract_score_text_candidates, parse_score_from_output


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
