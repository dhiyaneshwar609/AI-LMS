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
import logging
import time
from typing import Any, Dict, List

from hosted_llm import generate_draft, verify_and_correct

logger = logging.getLogger(__name__)


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
    "mention", "state", "write", "briefly", "short", "mark", "marks",
    "answer", "answers", "exam",
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

def _relevant_candidates(question: str, results: List[Dict[str, Any]]) -> List[tuple]:
    """Rank sentences per chunk while retaining each chunk's metadata."""
    terms = _words(question)
    if not terms:
        return []
    markers = (
        " is a ", " is an ", " is the ", " refers to ", " defined as ",
        " can be defined as ", " is known as ", " is called ",
        " framework for ", " system for ", " platform for ", " environment for ",
    )
    ranked = []
    for chunk_index, item in enumerate(results or []):
        chunk = _text(item.get("text") or item.get("chunk_text") or item.get("content"))
        for sentence_index, sentence in enumerate(_sentences(chunk)):
            words = _words(sentence)
            hits = len(terms & words)
            if not hits:
                continue
            low = " " + sentence.lower() + " "
            marker_hits = sum(1 for marker in markers if marker in low)
            score = (hits / max(1, len(terms))) * 4.0 + marker_hits * 3.0
            score -= max(0, len(sentence) - 220) / 220.0
            # Retain the full result record alongside the candidate so source
            # title/URL/page metadata remains available during selection.
            ranked.append((score, hits, marker_hits, chunk_index, sentence, item, sentence_index))
    ranked.sort(key=lambda row: (-row[0], -row[1], len(row[4])))
    return ranked


def _definition_candidates(question: str, results: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    """Choose one direct definition and a few relevant points from its chunk."""
    ranked = _relevant_candidates(question, results)
    markers = (" is a ", " is an ", " is the ", " refers to ", " defined as ",
               " can be defined as ", " is known as ", " is called ")
    definitions = [row for row in ranked if any(m in (" " + row[4].lower() + " ") for m in markers)]
    if not definitions:
        return []
    primary = definitions[0]
    selected = [primary]
    chunk_sentences = _sentences(_text(
        primary[5].get("text") or primary[5].get("chunk_text") or primary[5].get("content")
    ))
    for sentence in chunk_sentences[primary[6] + 1:primary[6] + 3]:
        if len(selected) >= min(limit, 3):
            break
        if re.match(r"^(?:it|this|these|they|the same|in addition)\b", sentence, re.IGNORECASE):
            selected.append((0, 0, 0, primary[3], sentence, primary[5], primary[6] + 1))
    for row in ranked:
        if len(selected) >= min(limit, 3):
            break
        if row[3] != primary[3] or row is primary:
            continue
        if any(m in (" " + row[4].lower() + " ") for m in markers):
            continue
        selected.append(row)
    return [row[4] for row in selected]


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
Copy the exact example input from the course material only when present.

### Output
[Copy the exact output explicitly shown in the course material, preserving
all code and formatting. Never execute, calculate, reconstruct, or guess an
output. Omit this subsection when no explicit output is present.

### Explanation of the Example
[Explain only relationships and behavior supported by the course material.]

Omit any subsection that has no supporting course evidence. Do not add an
example section when the course material contains no relevant example.
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


def _requirements_instruction(requirements_context=None) -> str:
    if not isinstance(requirements_context, dict):
        return ""

    requirements = requirements_context.get("requirements_context")
    if not isinstance(requirements, dict):
        requirements = requirements_context

    plan = requirements_context.get("plan")
    if not isinstance(plan, list):
        plan = []
    sections = plan or requirements.get("required_sections", [])
    sections = [str(item).replace("_", " ")[:100] for item in sections[:12]]

    intent = requirements_context.get("intent") or requirements.get("intent")
    answer_type = requirements_context.get("answer_type") or requirements.get("answer_type")
    detail_level = requirements_context.get("detail_level") or requirements.get("detail_level")
    answer_format = requirements_context.get("answer_format") or requirements.get("answer_format")
    exam_marks = requirements_context.get("exam_marks")
    if exam_marks is None:
        exam_marks = requirements.get("exam_marks")
    topic = requirements_context.get("topic") or requirements.get("topic")

    lines = [
        "ANSWER PRESENTATION REQUIREMENTS (format only; not factual evidence):",
        "Use COURSE MATERIAL as the only factual source. Conversation history may resolve references or preferences only, never facts. Skip optional sections when unsupported; never invent content.",
    ]
    for label, value in (
        ("Intent", intent),
        ("Answer type", answer_type),
        ("Detail level", detail_level),
        ("Answer format", answer_format),
    ):
        if value:
            lines.append(f"{label}: {str(value)[:80]}.")
    if isinstance(exam_marks, int):
        lines.append(f"Exam marks: {exam_marks}; keep the answer appropriately concise or detailed.")
        if exam_marks <= 2:
            lines.append(
                "For 2 marks, use at most two concise sentences. Give the direct answer or definition first, then add one distinct, relevant supporting point only when the course evidence supports one. Do not restate the first claim in different words; if the evidence supports only one point, give only that point."
            )
    if sections:
        lines.append("Structure in order: " + "; ".join(sections) + ".")
    if requirements.get("language_style") == "simple":
        lines.append("Use simple, clear language while preserving supported technical meaning.")
    optional_requests = []
    for field, label in (
        ("include_example", "example"),
        ("include_diagram", "diagram"),
        ("include_pseudocode", "pseudocode or flow"),
        ("include_complexity", "complexity details"),
    ):
        if requirements.get(field):
            optional_requests.append(label)
    if optional_requests:
        lines.append(
            "Requested optional content: " + ", ".join(optional_requests)
            + "; include each only when supported by course evidence and appropriate."
        )
    if topic:
        lines.append(
            "Topic identifier for resolving references only (not factual evidence): "
            + str(topic)[:120]
            + "."
        )
    return "\n".join(lines)


def _log_generation_result(kind, requirements_context, answer_format, answer) -> None:
    requirements = {}
    if isinstance(requirements_context, dict):
        nested = requirements_context.get("requirements_context")
        requirements = nested if isinstance(nested, dict) else requirements_context
    fields_used = [
        field for field in (
            "intent", "answer_type", "detail_level", "exam_marks", "answer_format",
            "required_sections", "include_example", "include_diagram",
            "include_pseudocode", "include_complexity", "language_style", "topic",
        ) if field in requirements
    ]
    logger.info(
        "phase=generator_requirements kind=%s supplied=%s fields_used=%s requested_format=%s requested_detail=%s exam_marks=%s answer_length_chars=%d",
        kind,
        bool(requirements_context),
        ",".join(fields_used),
        str(requirements.get("answer_format") or answer_format or "")[:80],
        str(requirements.get("detail_level") or "")[:80],
        requirements.get("exam_marks"),
        len(str(answer or "")),
    )


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
    requirements_context=None,
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

    requirement_instruction = _requirements_instruction(requirements_context)
    if requirement_instruction:
        user_prompt += "\n\n" + requirement_instruction

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
    requirements_context=None,
) -> str:
    requirement_instruction = _requirements_instruction(requirements_context)
    verifier_history = (
        f"{history}\n\n{requirement_instruction}"
        if requirement_instruction
        else history
    )
    return verify_and_correct(
        question=question,
        draft_answer=draft_answer,
        course_context=context,
        answer_format=answer_format,
        history=verifier_history,
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
    requirements_context=None,
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

    requirements = requirements_context if isinstance(requirements_context, dict) else {}
    nested = requirements.get("requirements_context")
    if isinstance(nested, dict):
        requirements = nested
    marks = requirements.get("exam_marks")
    if isinstance(marks, int) and marks <= 2:
        sentences = [s for s in _sentences(answer) if not s.startswith("#")]
        if len(sentences) > 2 or _has_repeated_claim(sentences):
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


def _has_repeated_claim(sentences: List[str]) -> bool:
    """Catch close paraphrases without requiring exact sentence matches."""
    content_sets = []
    for sentence in sentences:
        words = {
            re.sub(r"(ing|ed|es|s)$", "", word)
            for word in _words(sentence)
            if len(word) > 2 and word not in STOP_WORDS
        }
        if words:
            content_sets.append(words)

    for index, words in enumerate(content_sets):
        for other in content_sets[index + 1:]:
            shared = words & other
            # A shared subject alone is insufficient. Require several shared
            # content concepts and substantial overlap with the shorter claim.
            if len(shared) >= 4 and len(shared) / min(len(words), len(other)) >= 0.5:
                return True
    return False


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def _fallback_answer(
    question: str,
    results: List[Dict[str, Any]],
    requirements_context=None,
) -> str:
    answer_format = detect_answer_format(question)

    # For definitions, never use ordinary word overlap because it can select
    # a related fact instead of the actual definition.
    if answer_format == "definition":
        candidates = _definition_candidates(question, results, limit=3)
        return "\n\n".join(candidates) if candidates else ""

    ranked = _relevant_candidates(question, results)
    requirements = requirements_context if isinstance(requirements_context, dict) else {}
    nested = requirements.get("requirements_context")
    if isinstance(nested, dict):
        requirements = nested
    marks = requirements.get("exam_marks")
    limit = 2 if isinstance(marks, int) and marks <= 2 else 3
    selected = []
    seen = set()
    for row in ranked:
        sentence = row[4]
        key = re.sub(r"\W+", "", sentence.lower())
        if not key or key in seen:
            continue
        if len(selected) >= 1 and _has_repeated_claim([*selected, sentence]):
            continue
        selected.append(sentence)
        seen.add(key)
        if len(selected) >= limit:
            break

    return "\n\n".join(selected) if selected else ""


# ============================================================
# PUBLIC API
# ============================================================

def generate_short_answer(
    question: str,
    results: List[Dict[str, Any]],
    history: str = "",
    requirements_context=None,
) -> str:

    question = _clean_text(question)

    if not question:
        _log_generation_result("short", requirements_context, "", "")
        return ""

    if not results:
        _log_generation_result("short", requirements_context, "", "")
        return ""

    context = build_short_context(results)

    if not context:
        _log_generation_result("short", requirements_context, "", "")
        return ""

    answer_format = detect_answer_format(question)
    logger.info('phase=generator_started kind=short')

    def finish(answer: str) -> str:
        _log_generation_result("short", requirements_context, answer_format, answer)
        return answer

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
        requirements_context=requirements_context,
    )

    verified = _verify_answer(
        question=question,
        draft_answer=draft,
        context=context,
        answer_format=answer_format,
        history=history,
        requirements_context=requirements_context,
    )

    validation_started = time.perf_counter()
    verified_valid = _is_valid_answer(question, verified, results, requirements_context)
    logger.info(
        'phase=generator_validation kind=short attempt=1 candidate=verified accepted=%s elapsed_s=%.3f',
        verified_valid,
        time.perf_counter() - validation_started,
    )
    if verified_valid:
        return finish(verified)

    # If the verifier is temporarily unavailable, a valid generator
    # answer may still be used. It remains protected by the grounding gate.
    validation_started = time.perf_counter()
    draft_valid = _is_valid_answer(question, draft, results, requirements_context)
    logger.info(
        'phase=generator_validation kind=short attempt=1 candidate=draft accepted=%s elapsed_s=%.3f',
        draft_valid,
        time.perf_counter() - validation_started,
    )
    if draft_valid:
        return finish(draft)

    print(
        "[generator_short] First generator/verifier result rejected."
    )

    # ------------------------------------------------------------
    # Attempt 2: corrected generator -> verifier
    # ------------------------------------------------------------
    logger.info('phase=generator_retry kind=short attempt=2')
    draft = _call_generator(
        question=question,
        context=context,
        answer_format=answer_format,
        history=history,
        retry=True,
        requirements_context=requirements_context,
    )

    verified = _verify_answer(
        question=question,
        draft_answer=draft,
        context=context,
        answer_format=answer_format,
        history=history,
        requirements_context=requirements_context,
    )

    validation_started = time.perf_counter()
    verified_valid = _is_valid_answer(question, verified, results, requirements_context)
    logger.info(
        'phase=generator_validation kind=short attempt=2 candidate=verified accepted=%s elapsed_s=%.3f',
        verified_valid,
        time.perf_counter() - validation_started,
    )
    if verified_valid:
        return finish(verified)

    validation_started = time.perf_counter()
    draft_valid = _is_valid_answer(question, draft, results, requirements_context)
    logger.info(
        'phase=generator_validation kind=short attempt=2 candidate=draft accepted=%s elapsed_s=%.3f',
        draft_valid,
        time.perf_counter() - validation_started,
    )
    if draft_valid:
        return finish(draft)

    print(
        "[generator_short] Second result rejected. Using evidence fallback."
    )

    return finish(_fallback_answer(question, results, requirements_context))


# Compatibility
generate_answer = generate_short_answer
