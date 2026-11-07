"""Deterministic detection of clearly independent questions in a message."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_OPENING = re.compile(
    r"^(?:what\s+(?:is|are|was|were)|who\s+(?:is|are|was|were)|"
    r"when\s+(?:is|was|did)|where\s+(?:is|are|was|were)|"
    r"why\b|how\b|define\b|compare\b|differentiate\b|"
    r"distinguish\b|contrast\b|explain\b|describe\b|discuss\b|"
    r"elaborate\b|illustrate\b|list\b|enumerate\b|"
    r"give\b|write\b|state\b|mention\b|"
    r"can\s+you\b|could\s+you\b|tell\s+me\b)",
    re.IGNORECASE,
)
_MARKS_ONLY = re.compile(
    r"^(?:please\s+)?give\s+(?:a\s+)?\d+\s+marks?\s+answer\b(?!.*\bfor\s+\S+)",
    re.IGNORECASE,
)
_CONTEXTUAL_FOLLOW_UP = re.compile(
    r"^(?:explain|describe|show|give|make)\s+(?:it|this|that)\b|"
    r"^(?:explain|describe)\s+.+\b(?:it|this|that)\b|"
    r"^give\s+(?:me\s+)?an?\s+example\s+of\s+(?:it|this|that)\b",
    re.IGNORECASE,
)


def _split_candidates(message: str) -> List[str]:
    """Split at strong sentence boundaries, never at list items or conjunctions."""
    text = str(message or "").replace("\r\n", "\n").replace("\r", "\n")
    # Numbered and bullet lines are answer content, not question boundaries.
    text = re.sub(r"\n\s*(?=(?:\d+(?:\.\d+)*[.)]|[-*•])\s+)", " ", text)
    # A new question-shaped line is a strong boundary; wrapped prose is joined.
    text = re.sub(r"\n\s*(?=[A-Z])", "\n", text)
    pieces: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Question marks unambiguously end a question. Other sentence breaks
        # count only when the next clause has an independent request opener.
        parts = re.split(r"(?<=[?])\s+|(?<=[.!])\s+(?=[A-Z])", line)
        pieces.extend(p.strip() for p in parts if p.strip())
    return pieces


def decompose_questions(message: str, history: str = "") -> Dict[str, Any]:
    """Return independent question segments, retaining a lone request verbatim.

    History is accepted so callers can use a stable interface. In a multi-part
    message, a clearly separate pronoun follow-up becomes its own segment and
    records the preceding question as its local context.
    """
    del history
    original = str(message or "").strip()
    if not original:
        return {"is_multi_question": False, "questions": []}

    candidates = _split_candidates(original)
    # Discard a boundary unless both sides look like standalone requests.
    independent: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip(" \t\n")
        if not candidate:
            continue
        if not independent:
            independent.append(candidate)
            continue
        if _MARKS_ONLY.match(candidate):
            # A mark requirement without its own topic modifies the previous request.
            independent[-1] = f"{independent[-1]} {candidate}"
        elif _OPENING.match(candidate):
            independent.append(candidate)
        else:
            independent[-1] = f"{independent[-1]} {candidate}"

    if len(independent) < 2:
        return {"is_multi_question": False, "questions": [
            {"text": original, "order": 1}
        ]}
    questions = []
    for index, text in enumerate(independent, 1):
        item = {"text": text, "order": index}
        if _CONTEXTUAL_FOLLOW_UP.match(text) and index > 1:
            item["context_question"] = independent[index - 2]
        questions.append(item)
    return {"is_multi_question": True, "questions": questions}
