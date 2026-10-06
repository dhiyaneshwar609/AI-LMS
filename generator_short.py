# generator_short.py
# AI-LMS — Evidence-First Short Answer Generator
#
# Uses:
#   RAG V2 results
#   Hosted two-LLM generation + verification
#
# Main goal:
#   Generate an exam-style answer ONLY from retrieved course material.
#
# Public API:
#   generate_short_answer(question, results, history="")
#   generate_answer(question, results, history="")

from __future__ import annotations

import re
from typing import Any, Dict, List

from hosted_llm import generate_draft, verify_and_correct


# ============================================================
# CONFIG
# ============================================================

MAX_CONTEXT_CHARS = 7500
GROUNDING_THRESHOLD = 0.42


# ============================================================
# TEXT HELPERS
# ============================================================

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from",
    "into", "about", "what", "when", "where", "which",
    "their", "there", "then", "than", "have", "has",
    "are", "was", "were", "will", "would", "could",
    "should", "does", "how", "why", "can", "may",
    "also", "such", "using", "used", "use", "explain",
    "describe", "discuss", "detail", "details",
    "between", "differentiate", "difference", "differences",
    "compare", "comparison", "distinguish", "distinction",
    "contrast", "versus", "give", "list", "define",
    "mention", "state", "write", "briefly", "short",
}


def _text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _clean_text(text: str) -> str:
    text = _text(text)

    if not text:
        return ""

    text = text.replace("\x06", "[")
    text = text.replace("\x07", "]")

    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or ch.isprintable()
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _clean_answer(answer: str) -> str:

    answer = _clean_text(answer)

    if not answer:
        return ""

    answer = re.sub(
        r"^\s*(?:answer|final answer)\s*:\s*",
        "",
        answer,
        flags=re.I,
    )

    answer = re.sub(
        r"^\s*```(?:markdown|text|md)?\s*",
        "",
        answer,
        flags=re.I,
    )

    answer = re.sub(
        r"\s*```\s*$",
        "",
        answer,
    )

    return answer.strip()


def _words(text: str) -> set[str]:

    words = re.findall(
        r"[A-Za-z][A-Za-z0-9_-]{2,}",
        text or "",
    )

    return {
        w.lower()
        for w in words
        if w.lower() not in STOP_WORDS
    }


def _sentences(text: str) -> List[str]:

    text = _clean_text(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+|(?<=;)\s+",
        text,
    )

    result = []

    for part in parts:
        part = part.strip(" -•\t")

        if len(part) >= 12:
            result.append(part)

    return result


# ============================================================
# DEFINITION FOCUS
# ============================================================

def _definition_candidates(question: str, results: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Prefer explicit definition evidence over merely related facts."""
    terms = _words(question)
    source = _source_text(results)
    sentences = _sentences(source)
    if not terms or not sentences:
        return []

    markers = (
        " is a ", " is an ", " is the ", " refers to ",
        " defined as ", " can be defined as ", " is known as ",
        " is called ", " framework for ", " system for ",
        " platform for ", " environment for ",
    )

    ranked = []
    for sentence in sentences:
        low = " " + sentence.lower() + " "
        words = _words(sentence)
        term_hits = len(terms & words)
        if term_hits == 0:
            continue
        marker_hits = sum(1 for marker in markers if marker in low)
        score = (term_hits * 2.0) + (marker_hits * 6.0)
        if len(sentence) <= 240:
            score += 1.0
        ranked.append((score, sentence))

    ranked.sort(key=lambda item: (-item[0], len(item[1])))

    selected = []
    seen = set()
    for _, sentence in ranked:
        key = re.sub(r"\W+", "", sentence.lower())
        if not key or key in seen:
            continue
        selected.append(sentence)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _definition_focus_context(question: str, results: List[Dict[str, Any]]) -> str:
    candidates = _definition_candidates(question, results)
    if not candidates:
        return ""
    return (
        "DEFINITION-FOCUSED COURSE EVIDENCE:\n"
        "Use these sentences first when answering the definition.\n\n"
        + "\n".join(f"- {sentence}" for sentence in candidates)
    )


# ============================================================
# RAG CONTEXT
# ============================================================

def build_short_context(
    results: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:

    blocks = []
    seen = set()
    total = 0

    for i, item in enumerate(results or [], 1):

        text = _text(
            item.get("text")
            or item.get("chunk_text")
            or item.get("content")
        )

        if not text:
            continue

        text = _clean_text(text)

        key = re.sub(
            r"\W+",
            "",
            text.lower(),
        )

        if not key or key in seen:
            continue

        remaining = max_chars - total

        if remaining <= 0:
            break

        if len(text) > remaining:

            if blocks:
                break

            text = text[:remaining]

        topic = _text(item.get("topic"))
        page = item.get("page")

        header = f"[Evidence {i}]"

        if topic:
            header += f" Topic: {topic}"

        if page is not None:
            header += f" Page: {page}"

        block = f"{header}\n{text}"

        blocks.append(block)
        seen.add(key)

        total += len(block) + 2

    return "\n\n".join(blocks)


# ============================================================
# QUESTION FORMAT
# ============================================================

def detect_answer_format(question: str) -> str:

    q = question.lower().strip()

    if re.search(
        r"\b(differentiate|differentiate between|compare|comparison|"
        r"distinguish|difference between|contrast|versus|vs)\b",
        q,
    ):
        return "comparison"

    if re.search(
        r"\b(define|what is|what are|definition of)\b",
        q,
    ):
        return "definition"

    if re.search(r"\b(?:example|examples|instance)\b", q):
        return "example"

    if re.search(
        r"\b(advantages|disadvantages|limitations|merits|demerits)\b",
        q,
    ):
        return "advantages_limitations"

    if re.search(
        r"\b(steps|procedure|algorithm|algorithm for)\b",
        q,
    ):
        return "algorithm"

    if re.search(
        r"\b(classify|classification|types of|type of)\b",
        q,
    ):
        return "classification"

    if re.search(
        r"\b(why|purpose|importance|need for)\b",
        q,
    ):
        return "why_purpose"

    if re.search(
        r"\b(how does|how do|working|process|workflow)\b",
        q,
    ):
        return "process"

    if re.search(
        r"\b(explain|describe|discuss|elaborate|illustrate)\b",
        q,
    ):
        return "explanation"

    return "general"


# ============================================================
# FORMAT INSTRUCTIONS
# ============================================================

def _format_instruction(answer_format: str) -> str:

    if answer_format == "definition":

        return """
Give a direct definition first.

For a question of the form "What is X?" or "Define X":
- First identify the sentence in the course material that defines X.
- Prefer explicit definitions such as "X is ...", "X refers to ...", or "X is a/an ...".
- Do NOT answer with a related operation, component, process, example, implementation detail, or consequence merely because it mentions X.
- If the material contains a direct definition, use that definition before secondary details.

Then provide only the important explanation, properties, purpose, or example that is supported by the course material.
"""

    if answer_format == "example":
        return """
Start with a level-two heading containing the concept name, followed by a
concise explanation supported by the course material. Then use this structure:

## Course-Supported Example

### Input
[Copy the exact example input from the course material. If it is not
available in the extracted text, state that explicitly.]

### Output
[Copy the exact output explicitly shown in the course material, preserving
all code and formatting. Never execute, calculate, reconstruct, or guess an
output. If it is not explicitly present, say it is not available in the
extracted course material.]

### Explanation of the Example
[Explain only relationships and behavior supported by the course material.]

Do not omit Input or Output. Do not put explanations in place of source output.
"""

    if answer_format == "comparison":

        return """
Use a Markdown comparison table.

Use:

| Aspect | Concept 1 | Concept 2 |
|---|---|---|

Include only comparison points supported by
the course material.
"""

    if answer_format == "advantages_limitations":

        return """
Use clear headings:

### Advantages
- ...

### Limitations
- ...

Include only points supported by the material.
"""

    if answer_format == "algorithm":

        return """
Give the algorithm/procedure name first if supported.

Then:

### Steps

1. ...
2. ...
3. ...

Do not invent missing steps.
"""

    if answer_format == "classification":

        return """
Give the classification/types as a numbered list.

Explain each type only when the course material
contains the explanation.
"""

    if answer_format == "process":

        return """
Explain the process in logical ordered steps.

Use a flow representation only when the
course material supports the relationships.
"""

    if answer_format == "why_purpose":

        return """
Answer the reason/purpose directly.

Then give the supported reasons as concise points.
"""

    if answer_format == "explanation":

        return """
Give a clear academic explanation.

Use headings, bullets, steps, formulas or examples
only when supported by the course material.
"""

    return """
Use the most natural academic structure for
the student's exact question.
"""


# ============================================================
# PROMPT
# ============================================================

def _system_prompt(answer_format: str) -> str:

    return f"""
You are the AI Tutor inside a university Learning Management System.

Your task is to answer the student's question using ONLY the
COURSE MATERIAL supplied to you.

STRICT RULES:

1. COURSE MATERIAL is the only factual authority.

2. Do NOT use outside knowledge.

3. Do NOT complete missing information from your own knowledge.

4. Do NOT invent:
   - facts
   - definitions
   - examples
   - formulas
   - values
   - algorithms
   - steps
   - components
   - relationships
   - advantages
   - limitations
   - classifications
   - diagrams

5. Preserve the technical terminology used in the course material.

6. Answer exactly what the student asks.

7. If a requested point is not supported by the course material,
   do not guess.

8. Do not mention:
   - RAG
   - retrieval
   - chunks
   - embeddings
   - Gemma
   - Ollama
   - prompts
   - validation
   - internal reasoning

9. Output ONLY the student's final answer.

ANSWER FORMAT:

{_format_instruction(answer_format)}
""".strip()


# ============================================================
# OLLAMA
# ============================================================

def _call_generator(
    question: str,
    context: str,
    answer_format: str,
    history: str = "",
    retry: bool = False,
) -> str:
    retry_instruction = ""
    if retry:
        retry_instruction = """
CORRECTION REQUIRED:
The previous draft did not pass the grounding checks.
Remove unsupported claims and regenerate using ONLY the supplied course
material. Do not add outside knowledge.
""".strip()

    user_prompt = f"""
Student question:

{question}

Previous conversation:

{history or "(none)"}

COURSE MATERIAL:

----------------------------

{context}

----------------------------

{retry_instruction}

Now produce the final draft answer.
""".strip()

    return generate_draft(
        system_prompt=_system_prompt(answer_format),
        user_prompt=user_prompt,
    )


def _verify_answer(
    question: str,
    draft_answer: str,
    context: str,
    answer_format: str,
    history: str = "",
) -> str:
    return verify_and_correct(
        question=question,
        draft_answer=draft_answer,
        course_context=context,
        answer_format=answer_format,
        history=history,
        long_answer=False,
    )


# ============================================================
# GROUNDING
# ============================================================

def _source_text(results: List[Dict[str, Any]]) -> str:

    parts = []

    for item in results or []:

        text = _text(
            item.get("text")
            or item.get("chunk_text")
            or item.get("content")
        )

        if text:
            parts.append(text)

    return " ".join(parts)


def _sentence_grounding(
    sentence: str,
    source: str,
) -> float:

    sentence_words = _words(sentence)
    source_words = _words(source)

    if not sentence_words:
        return 1.0

    if not source_words:
        return 0.0

    overlap = (
        len(sentence_words & source_words)
        / len(sentence_words)
    )

    return overlap


def _unsupported_numbers(
    answer: str,
    source: str,
) -> bool:

    pattern = (
        r"(?<![A-Za-z])"
        r"(?:"
        r"\d+(?:\.\d+)%"
        r"|"
        r"\d+\.\d+"
        r"|"
        r"\b(?:19|20)\d{2}\b"
        r")"
    )

    answer_numbers = set(
        re.findall(pattern, answer)
    )

    source_numbers = set(
        re.findall(pattern, source)
    )

    return not answer_numbers.issubset(
        source_numbers
    )


def _looks_like_meta(answer: str) -> bool:

    bad_patterns = [
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\baccording to the retrieved\b",
        r"\bbased on the provided context\b",
        r"\bthe rag\b",
        r"\bthe model\b",
    ]

    return any(
        re.search(pattern, answer, re.I)
        for pattern in bad_patterns
    )


def _grounding_score(
    answer: str,
    results: List[Dict[str, Any]],
) -> float:

    source = _source_text(results)

    sentences = _sentences(answer)

    if not source or not sentences:
        return 0.0

    scores = []

    for sentence in sentences:

        # Headings are not factual claims.
        if sentence.startswith("#"):
            continue

        score = _sentence_grounding(
            sentence,
            source,
        )

        scores.append(score)

    if not scores:
        return 1.0

    return sum(scores) / len(scores)


def _definition_structure_is_valid(answer: str) -> bool:
    """Reject a related fact that does not actually define the term."""
    if not _sentences(answer):
        return False
    markers = (
        " is a ", " is an ", " is the ", " refers to ",
        " defined as ", " can be defined as ", " is known as ",
        " is called ",
    )
    text = " " + answer.lower() + " "
    return any(marker in text for marker in markers)


def _is_valid_answer(
    question: str,
    answer: str,
    results: List[Dict[str, Any]],
) -> bool:

    answer = _clean_answer(answer)

    if not answer:
        return False

    if _looks_like_meta(answer):
        return False

    if detect_answer_format(question) == "definition":
        if not _definition_structure_is_valid(answer):
            return False

    source = _source_text(results)

    if not source:
        return False

    score = _grounding_score(
        answer,
        results,
    )

    if score < GROUNDING_THRESHOLD:
        return False

    if _unsupported_numbers(
        answer,
        source,
    ):
        return False

    return True


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def _fallback_answer(
    question: str,
    results: List[Dict[str, Any]],
) -> str:

    source = _source_text(results)
    if not source:
        return ""

    answer_format = detect_answer_format(question)

    # For definitions, never use ordinary word overlap because it can select
    # a related fact instead of the actual definition.
    if answer_format == "definition":
        candidates = _definition_candidates(question, results, limit=3)
        return candidates[0] if candidates else ""

    question_words = _words(question)
    sentences = _sentences(source)
    ranked = []

    for sentence in sentences:
        words = _words(sentence)
        if not words:
            continue
        overlap = len(question_words & words)
        ranked.append((overlap, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for _, sentence in ranked:
        if sentence in selected:
            continue
        selected.append(sentence)
        if len(selected) >= 5:
            break

    return "\n\n".join(selected) if selected else ""


# ============================================================
# PUBLIC API
# ============================================================

def generate_short_answer(
    question: str,
    results: List[Dict[str, Any]],
    history: str = "",
) -> str:

    question = _clean_text(question)

    if not question:
        return ""

    if not results:
        return ""

    context = build_short_context(results)

    if not context:
        return ""

    answer_format = detect_answer_format(question)

    # For definitions, put explicit definition evidence first.
    if answer_format == "definition":
        focus = _definition_focus_context(question, results)
        if focus:
            context = focus + "\n\nGENERAL COURSE EVIDENCE:\n" + context

    # ------------------------------------------------------------
    # Attempt 1: generator -> verifier
    # ------------------------------------------------------------
    draft = _call_generator(
        question=question,
        context=context,
        answer_format=answer_format,
        history=history,
        retry=False,
    )

    verified = _verify_answer(
        question=question,
        draft_answer=draft,
        context=context,
        answer_format=answer_format,
        history=history,
    )

    if _is_valid_answer(question, verified, results):
        return verified

    # If the verifier is temporarily unavailable, a valid generator
    # answer may still be used. It remains protected by the grounding gate.
    if _is_valid_answer(question, draft, results):
        return draft

    print(
        "[generator_short] First generator/verifier result rejected."
    )

    # ------------------------------------------------------------
    # Attempt 2: corrected generator -> verifier
    # ------------------------------------------------------------
    draft = _call_generator(
        question=question,
        context=context,
        answer_format=answer_format,
        history=history,
        retry=True,
    )

    verified = _verify_answer(
        question=question,
        draft_answer=draft,
        context=context,
        answer_format=answer_format,
        history=history,
    )

    if _is_valid_answer(question, verified, results):
        return verified

    if _is_valid_answer(question, draft, results):
        return draft

    print(
        "[generator_short] Second result rejected. Using evidence fallback."
    )

    return _fallback_answer(
        question,
        results,
    )


# Compatibility
generate_answer = generate_short_answer