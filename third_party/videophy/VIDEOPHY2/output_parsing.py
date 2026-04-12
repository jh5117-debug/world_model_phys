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
_ASSISTANT_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:ai|assistant)\s*:\s*", re.IGNORECASE)
_LINE_SPLIT_RE = re.compile(r"[\r\n]+")


def _allowed_scores_for_task(task: str) -> set[int]:
    if task == "rule":
        return {0, 1, 2}
    return {1, 2, 3, 4, 5}


def extract_score_text_candidates(output: str) -> list[str]:
    text = output.strip()
    if not text:
        return []

    candidates: list[str] = []

    def _add(candidate: str) -> None:
        normalized = candidate.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    assistant_parts = _ASSISTANT_MARKER_RE.split(text)
    if len(assistant_parts) > 1:
        _add(assistant_parts[-1])
    _add(text)

    for candidate in list(candidates):
        first_line = _LINE_SPLIT_RE.split(candidate, maxsplit=1)[0].strip()
        _add(first_line)

    return candidates


def _parse_single_score(text: str, *, allowed_scores: set[int]) -> int | None:
    lowered = text.lower().strip()
    if not lowered:
        return None

    word_matches = [_WORD_TO_SCORE[word] for word in _WORD_SCORE_RE.findall(lowered)]
    word_matches = [score for score in word_matches if score in allowed_scores]
    if len(word_matches) == 1:
        return word_matches[0]
    if len(word_matches) > 1:
        return None

    digit_matches = [int(match) for match in _DIGIT_SCORE_RE.findall(lowered)]
    digit_matches = [score for score in digit_matches if score in allowed_scores]
    if len(digit_matches) == 1:
        return digit_matches[0]
    return None


def parse_score_from_output(output: str, *, task: str) -> int | None:
    text = output.strip()
    if not text:
        return None

    allowed_scores = _allowed_scores_for_task(task)
    for candidate in extract_score_text_candidates(text):
        score = _parse_single_score(candidate, allowed_scores=allowed_scores)
        if score is not None:
            return score
    return None
