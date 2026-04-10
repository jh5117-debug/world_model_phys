import re


_WORD_TO_SCORE = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_WORD_SCORE_RE = re.compile(r"\b(zero|one|two|three|four|five)\b")
_DIGIT_SCORE_RE = re.compile(r"(?<!\d)([0-5])(?!\d)")


def _allowed_scores_for_task(task: str) -> set[int]:
    if task == "rule":
        return {0, 1, 2}
    return {1, 2, 3, 4, 5}


def parse_score_from_output(output: str, *, task: str) -> int | None:
    text = output.lower().strip()
    if not text:
        return None

    allowed_scores = _allowed_scores_for_task(task)

    word_matches = _WORD_SCORE_RE.findall(text)
    if word_matches:
        word_scores = [score for word, score in _WORD_TO_SCORE.items() if word in word_matches]
        word_scores = [score for score in word_scores if score in allowed_scores]
        return word_scores[0] if len(word_scores) == 1 and len(word_matches) == 1 else None

    digit_matches = [int(match) for match in _DIGIT_SCORE_RE.findall(text)]
    digit_matches = [score for score in digit_matches if score in allowed_scores]
    return digit_matches[0] if len(digit_matches) == 1 else None
