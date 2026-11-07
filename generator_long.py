# AI-LMS Long Answer Generator V2 — grounded, structure-aware, RAG-only
"""
AI-LMS Long Answer Generator
----------------------------
Generic, course-material-grounded generator for long/detailed university
questions.

Uses a hosted two-LLM generator + verifier.
Answers are generated only from retrieved RAG evidence.

The answer structure adapts to the student's wording rather than forcing one
fixed template. It supports explanations, algorithms, processes, architecture,
comparisons, classifications, advantages/limitations, formulas/numericals,
and other detailed academic questions.
"""

from __future__ import annotations

import re
import logging
import time
import numpy as np
from hosted_llm import generate_draft, verify_and_correct
from rag_v2_search import _load_model
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONTEXT_CHARS = 10000


# ============================================================
# TEXT HELPERS
# ============================================================

def _text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _clean_text(text: str) -> str:
    text = _text(text)

    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return text.strip()


def _clean_answer(text: str) -> str:
    """
    Clean Hosted model output while preserving academic and technical notation.

    In particular, repair control-character corruption that can occur in
    expressions such as dependent[k][i], then remove any remaining
    non-printable characters.
    """

    import unicodedata

    text = _text(text)

    if not text:
        return ""

    # Normalize Unicode without changing ordinary academic notation.
    text = unicodedata.normalize("NFKC", text)

    # Some model/OCR pipelines can turn square brackets into control
    # characters. Restore the common mappings before removing controls.
    text = text.replace("\x06", "[")
    text = text.replace("\x07", "]")

    # Remove remaining ASCII control characters while preserving
    # newline and tab.
    text = "".join(
        ch
        for ch in text
        if ch in ("\n", "\t") or ch.isprintable()
    )

    # Remove accidental code fences.
    text = re.sub(
        r"^\s*```(?:markdown|text|md|html)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    text = text.replace("```", "")

    # Remove accidental answer/meta prefixes.
    text = re.sub(
        r"^\s*(?:answer|final\s+answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^(?:Based on|According to|Using|Drawing upon)"
        r"[^.\n]{0,180}"
        r"(?:course material|provided material|retrieved material)"
        r"[.:]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove internal/meta-only lines.
    bad_line_patterns = (
        r"^\s*(?:analysis|reasoning|plan|approach|strategy)\s*:?\s*$",

        r"^\s*(?:proposed\s+answer\s+structure|answer\s+structure)"
        r"\s*:?\s*$",

        r"^\s*(?:student\s+question|course\s+material)\s*:\s*$",

        r"^\s*(?:i\s+will|i\s+would|let'?s|we\s+should|we\s+need)\b",

        r"^\s*(?:according\s+to\s+the|based\s+on\s+the)\s+"
        r"(?:provided|given|supplied)\b",
    )

    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if any(
            re.search(
                pattern,
                stripped,
                re.IGNORECASE,
            )
            for pattern in bad_line_patterns
        ):
            continue

        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines).strip()

    # Keep technical expressions compact:
    #
    # P [ i ]  -> P[i]
    # P [ i ] [ j ] -> P[i][j]
    #
    text = re.sub(
        r"\[\s+",
        "[",
        text,
    )

    text = re.sub(
        r"\s+\]",
        "]",
        text,
    )

    # Prevent huge blank gaps.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Remove duplicate paragraphs without altering their content.
    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text,
        )
        if p.strip()
    ]

    unique = []
    seen = set()

    for paragraph in paragraphs:
        key = re.sub(
            r"\W+",
            "",
            paragraph.lower(),
        )

        if key and key not in seen:
            unique.append(paragraph)
            seen.add(key)

    return "\n\n".join(unique).strip()


def _words(text: str) -> List[str]:
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )


def _meaningful_words(text: str) -> List[str]:

    stopwords = {
        "what",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "from",
        "by",
        "explain",
        "describe",
        "discuss",
        "elaborate",
        "give",
        "details",
        "about",
        "how",
        "why",
        "does",
        "do",
        "its",
        "their",
        "this",
        "that",
        "these",
        "those",
        "compare",
        "comparison",
        "difference",
        "differences",
        "differentiate",
        "distinguish",
        "contrast",
        "versus",
        "vs",
        "list",
        "mention",
        "define",
        "write",
        "briefly",
        "short",
    }

    return [
        word
        for word in _words(text)
        if len(word) >= 3
        and word not in stopwords
    ]


# ============================================================
# GENERIC ANSWER-FORMAT DETECTION
# ============================================================

def _is_comparison_question(question: str) -> bool:

    q = _clean_text(question).lower()

    if not q:
        return False

    patterns = (
        r"\bdifferentiate\s+between\b",
        r"\bdifferentiate\b",
        r"\bdifference\s+between\b",
        r"\bdifferences\s+between\b",
        r"\bcompare\b",
        r"\bcomparison\s+between\b",
        r"\bdistinguish\s+between\b",
        r"\bdistinguish\b",
        r"\bcontrast\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bvs\b",
    )

    return any(
        re.search(
            pattern,
            q,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


def detect_answer_format(question: str) -> str:
    """
    Infer the requested academic structure from the student's wording.

    This function contains no topic-specific knowledge.
    """

    q = _clean_text(question).lower()

    if not q:
        return "general"

    if _is_comparison_question(q):
        return "comparison"

    if re.search(
        r"\b(advantage|advantages|benefit|benefits|"
        r"limitation|limitations|disadvantage|"
        r"disadvantages|pros and cons)\b",
        q,
    ):
        return "advantages_limitations"

    if re.search(
        r"\b(solve|calculate|compute|find|derive|evaluate)\b",
        q,
    ) or re.search(
        r"\b(numerical|formula|equation|calculation)\b",
        q,
    ):
        return "numerical"

    if re.search(
        r"\b(algorithm|algorithms|procedure|pseudocode)\b",
        q,
    ) or re.search(
        r"\b(steps to|steps for)\b",
        q,
    ):
        return "algorithm"

    if re.search(
        r"\b(draw|show|give|explain|describe)?\s*"
        r"(diagram|block diagram|flowchart|flow chart|figure|"
        r"architecture diagram|system diagram)\b",
        q,
    ):
        return "diagram"

    if re.search(
        r"\b(architecture|architectures|system architecture|"
        r"components and working|components of)\b",
        q,
    ):
        return "architecture"

    if re.search(
        r"\b(types of|type of|classification|classify|"
        r"categories|categorize)\b",
        q,
    ):
        return "classification"

    if re.search(
        r"\b(how does|how do|how is|how are|working of|"
        r"works|process of|workflow|steps involved)\b",
        q,
    ):
        return "process"

    if re.search(
        r"\b(why|purpose of|importance of|need for|reason for)\b",
        q,
    ):
        return "why_purpose"

    if re.search(
        r"\b(list|enumerate|mention|name|state the types|"
        r"give the types)\b",
        q,
    ):
        return "list"

    if re.search(
        r"^\s*(what is|what are|define|definition of)\b",
        q,
    ):
        return "definition"

    if re.search(r"\b(?:example|examples|instance)\b", q):
        return "example"

    if re.search(
        r"\b(explain|describe|discuss|elaborate|illustrate)\b",
        q,
    ):
        return "explanation"

    return "general"


def _format_instruction(answer_format: str) -> str:

    instructions = {

        "definition": """
FORMAT:
- Start with a precise definition.
- Then add the important supported explanation, properties, roles, or example
  only when present in the course material.
- Do not add unrelated sections just to make the answer longer.
""",

        "example": """
FORMAT:
Start with a level-two heading containing the concept name, followed by a
concise explanation supported by the course material. Then use this structure:

## Course-Supported Example

### Input
Copy the exact example input from the course material only when present.

### Output
Copy the exact output explicitly present in the course material, preserving
code and formatting. Never execute, calculate, reconstruct, or guess output.
Omit this subsection when no explicit output is present.

### Explanation of the Example
Explain only what the course evidence supports. Omit any subsection that has
no supporting evidence, and omit the example section when no relevant example
is present in the course material.
""",

        "list": """
FORMAT:
- Present the requested items as a clear numbered/bulleted list.
- Explain each item to the depth supported by the course material.
""",

        "comparison": """
FORMAT:
- Use a Markdown comparison table for the comparison itself.
- Add a short introductory statement only when useful.
- Use only supported comparison aspects; never invent differences.
""",

        "advantages_limitations": """
FORMAT:
- Use clear headings such as "### Advantages" and "### Limitations" when
  those parts are requested and supported.
- Use numbered/bulleted points with brief explanations.
""",

        "numerical": """
FORMAT:
- If the evidence contains the required data, use:
  ### Given
  ### Formula
  ### Substitution / Calculation
  ### Final Answer
- Show the actual calculation only when the supplied material provides enough
  information.
- Never invent values, formulas, assumptions, or results.
""",

        "algorithm": """
FORMAT:
- Start with a clear definition or purpose of the algorithm/procedure.
- Use "### Steps" or an equivalent clear heading and present the steps in correct logical order.
- Explain each important step sufficiently for the reader to understand its role.
- Include important conditions and working details when supported by the retrieved course evidence.
- Include pseudocode or a flow only when supported by the evidence and appropriate to the question.
- Include complexity or an example only when supported by the evidence and relevant to the question.
- Write at least 60 words when the retrieved evidence supports that amount of detail.
""",

        "diagram": """
FORMAT:
- Start with a simple ASCII/text diagram using boxes/arrows when the source
  explicitly supports the components and relationships.
- Follow it with a detailed explanation if requested.
- Every component and relationship shown must be supported by the material.
- Never invent a missing component or connection.
""",
        "architecture": """
FORMAT:
- Use sections such as "### Architecture", "### Components",
  "### Working", and "### Relationships" only when supported.
- A diagram or flow representation may be included only if the retrieved
  course material explicitly supports the relationships/components shown.
- Never invent a diagram or unsupported component.
""",

        "classification": """
FORMAT:
- Use a clear "### Classification" or "### Types" section.
- Explain each supported type/category separately.
""",

        "process": """
FORMAT:
- Explain the process/workflow in logical ordered steps.
- Use headings when they improve readability.
- A flow representation should be used only when supported by the material.
""",

        "why_purpose": """
FORMAT:
- State the reason/purpose directly.
- Follow with the supported reasons, roles, or significance.
""",

        "explanation": """
FORMAT:
- Give a logically organized explanation.
- Choose headings based on the actual content instead of using a fixed set.
- Include examples, formulas, steps, or relationships only when supported.
""",

        "general": """
FORMAT:
- Choose the most natural detailed academic structure for the exact question.
- Use headings, bullets, tables, formulas, or steps only when they improve
  the answer and are supported by the material.
""",
    }

    return instructions.get(
        answer_format,
        instructions["general"],
    ).strip()


# ============================================================
# RAG CONTEXT
# ============================================================

def build_long_context(
    results: List[Dict[str, Any]]
) -> str:
    """
    Build a clean context from RAG V2 results.

    The generator receives only retrieved course evidence.
    """

    blocks = []
    total_chars = 0

    for i, result in enumerate(
        results or [],
        start=1,
    ):

        if not isinstance(result, dict):
            continue

        text = _clean_text(
            result.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        filename = _text(
            result.get(
                "filename",
                "",
            )
        )

        page = result.get(
            "page",
            "",
        )

        section = _text(
            result.get(
                "section",
                "",
            )
        )

        header_parts = []

        if filename:
            header_parts.append(
                f"Source: {filename}"
            )

        if page != "":
            header_parts.append(
                f"Page: {page}"
            )

        if section:
            header_parts.append(
                f"Section: {section}"
            )

        header = " | ".join(
            header_parts
        )

        block = f"[Evidence {i}]"

        if header:
            block += f"\n{header}"

        block += f"\n{text}"

        remaining = (
            MAX_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        if len(block) > remaining:
            block = block[:remaining]

        blocks.append(block)

        total_chars += (
            len(block)
            + 2
        )

    return "\n\n".join(
        blocks
    ).strip()


# ============================================================
# SYSTEM PROMPT
# ============================================================

BASE_SYSTEM_PROMPT = """
You are the long-answer AI Tutor inside an academic Learning Management System.

Your job is to answer the student's question using ONLY the retrieved course
material supplied in the user message.

IMPORTANT RULES:

1. Use the retrieved course material as the factual source.

2. Do not use outside knowledge to introduce unsupported facts.

3. Do not invent examples, statistics, names, components, advantages,
   disadvantages, formulas, steps, diagrams, or technical details.

4. Preserve important technical terminology from the course material.

5. Answer the actual student question directly.

6. Adapt the structure to the wording and type of question.

7. Do not force a fixed template when the question does not require it.

8. If the course material does not provide enough information for a requested
   point, say that there is not enough information in the available course
   material rather than guessing.

9. Do not mention prompts, RAG, retrieval, agents, evidence, chunks, context,
   embeddings, Ollama, Gemma, or internal processing.

10. Do not say "according to the provided material" or similar meta phrases.

11. Do not output a list of retrieved evidence.

12. Write a synthesized academic answer, not a copy of the evidence blocks.

13. Never emit ASCII control characters or corrupted technical symbols.

14. Preserve technical notation using ordinary readable characters. For example,
    use dependent[k][i], P_i, P_j, P_k, and (i, j, k) when such notation is
    supported by the course material. Do not replace brackets or subscripts
    with hidden/control characters.

15. Use clean Markdown headings, numbered steps, bullets, tables, formulas,
    and paragraphs according to the actual question type. Do not force a
    fixed template when it is not appropriate.

16. For algorithms, keep the algorithm name, purpose, conditions, variables,
    and steps in their correct logical order.

17. For architecture/process questions, explain supported components and
    relationships in a logical sequence.

18. For comparisons, use a Markdown table only when a genuine comparison
    is requested and enough supported attributes are available.

19. Do not output retrieved evidence as a numbered source list.

20. Keep mathematical and technical notation readable. Examples:
    dependent[k][i]
    dependent[i][j]
    P_i
    P_j
    P_k
    (i, j, k)

21. Never convert square brackets into control characters.

22. Never replace variable names with visually corrupted symbols.

23. If the course material contains a variable such as dependent[k][i],
    reproduce it exactly in readable form.

24. Do not unnecessarily rewrite the technical notation from the source.

25. Give the answer in an exam-friendly academic format.

26. For a long-answer question, provide sufficient explanation rather than
    only a few sentences.

27. Do not repeat the same point in multiple sections.

28. Do not create empty headings.

29. Do not add a "Sources" section inside the generated answer. The LMS UI
    handles source display separately.

30. Do not mention that you were given retrieved context.

31. Do not mention confidence scores or retrieval scores.

32. Do not mention internal validation or answer-generation steps.

33. If only part of a requested answer is supported, answer only the supported
    part and clearly state the limitation without guessing.

34. Preserve the terminology and naming used in the course material.

35. When the question asks for steps, present the steps in the correct order.

36. When the question asks for a definition, give the definition first.

37. When the question asks for comparison, make the comparison easy to read.

38. When the question asks for architecture, separate components and working
    when the source supports those distinctions.

39. When the question asks for a formula or numerical calculation, show the
    formula and calculation clearly when supported.

40. Never output hidden control characters.
""".strip()


# ============================================================
# USER PROMPT BUILDER
# ============================================================

def _build_user_prompt(
    question: str,
    context: str,
    answer_format: str,
    retry: bool = False,
    history: str = "",
    requirements_context=None,
) -> str:

    extra = ""

    if retry:
        extra = """
IMPORTANT CORRECTION:

Write the actual answer to the student's question now.

Do not output evidence lists or source descriptions.

Every factual statement must be supported by the supplied course material.

Follow the requested answer structure below.

Make all technical notation readable.

Do not output control characters.
""".strip()

    requirement_instruction = _requirements_instruction(requirements_context)
    requirement_block = "\n\n" + requirement_instruction if requirement_instruction else ""

    return f"""
Student question:

{question}

Detected answer type:

{answer_format}

Required structure guidance:

{_format_instruction(answer_format)}

Previous conversation (ONLY for resolving references such as "this", "that",
or "the above"; it is NOT a factual source):

{history or "(none)"}

Retrieved course material — the ONLY factual source:

{context}

{extra}{requirement_block}

Now write at the resolved detail level. Use a moderately detailed academic answer for medium or structured requirements, and a complete structured academic answer only for detailed or high-mark requirements.

Important:
- Return only the answer.
- Do not describe your reasoning.
- Do not describe the retrieved material.
- Do not mention RAG.
- Do not mention sources as evidence blocks.
- Do not include internal instructions.
- Keep technical notation readable.
""".strip()


# ============================================================
# OLLAMA CALL
# ============================================================

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
        "Use COURSE MATERIAL as the only factual source. Skip optional sections when unsupported; never invent content.",
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
        lines.append(f"Exam marks: {exam_marks}; use an appropriately complete structure.")
        if exam_marks <= 2:
            lines.append(
                "For 2 marks, use at most two concise sentences. Give the direct answer or definition first, then add one distinct, relevant supporting point only when the course evidence supports one. Do not restate the first claim in different words; if the evidence supports only one point, give only that point."
            )
    if sections:
        lines.append("Structure in order: " + "; ".join(sections) + ".")
    if requirements.get("language_style") == "simple":
        lines.append("Use simple, clear language while preserving supported technical meaning.")
    if answer_format == "architecture":
        if detail_level in {"structured", "medium"} and not (
            isinstance(exam_marks, int) and exam_marks >= 8
        ):
            lines.append(
                "For a normal architecture explanation, combine each core component with its role, then explain only the key supported relationships. Keep the answer moderately detailed; omit peripheral ecosystem tools unless the question asks for them."
            )
        elif detail_level == "detailed" or (
            isinstance(exam_marks, int) and exam_marks >= 8
        ):
            lines.append(
                "Use the complete planned structure appropriate to the requested detailed or exam-mark answer. Keep every component, role, and relationship grounded in course evidence."
            )
    if answer_format in {"architecture", "process", "explanation", "general"}:
        lines.append(
            "Preserve the actor and action stated in the evidence. If the evidence says MapReduce schedules mappers near HDFS blocks, describe MapReduce as working with HDFS; do not attribute that scheduling action to HDFS alone."
        )
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

def _call_generator(
    question: str,
    context: str,
    answer_format: str = "general",
    retry: bool = False,
    history: str = "",
    requirements_context=None,
) -> str:

    if not context:
        return ""

    extra = ""

    if retry:
        extra = """
IMPORTANT CORRECTION:
The previous draft did not pass the grounding checks.
Remove unsupported claims and regenerate using ONLY the supplied course
material. Do not add outside knowledge.
""".strip()

    user_prompt = _build_user_prompt(
        question=question,
        context=context,
        answer_format=answer_format,
        retry=retry,
        history=history,
        requirements_context=requirements_context,
    )

    if extra:
        user_prompt += "\n\n" + extra

    return generate_draft(
        system_prompt=BASE_SYSTEM_PROMPT,
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
        long_answer=True,
    )


# ============================================================
# ANSWER QUALITY CHECKS
# ============================================================

def _looks_like_evidence_list(
    answer: str
) -> bool:

    answer = _text(answer)

    if not answer:
        return False

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    ]

    if len(lines) < 3:
        return False

    numbered = sum(
        bool(
            re.match(
                r"^\d+\.\s+",
                line,
            )
        )
        for line in lines
    )

    evidence_words = sum(
        bool(
            re.search(
                r"\b(source|evidence|retrieved|"
                r"course material|page)\b",
                line,
                re.IGNORECASE,
            )
        )
        for line in lines
    )

    return (
        numbered >= 3
        and evidence_words >= 2
    )


def _looks_like_meta_answer(
    answer: str
) -> bool:

    answer = _text(answer)

    if not answer:
        return True

    bad_patterns = [
        r"\bas an ai\b",
        r"\bi cannot answer\b",
        r"\bi can't answer\b",
        r"\bmy instructions\b",
        r"\bsystem prompt\b",
        r"\bdeveloper message\b",
        r"\bthe prompt asks\b",
        r"\bthe user asks\b",
        r"\binternal reasoning\b",
    ]

    return any(
        re.search(
            pattern,
            answer,
            flags=re.IGNORECASE,
        )
        for pattern in bad_patterns
    )


def _answer_overlap(
    question: str,
    answer: str,
) -> float:

    q_words = set(
        _meaningful_words(question)
    )

    a_words = set(
        _meaningful_words(answer)
    )

    if not q_words or not a_words:
        return 0.0

    overlap = q_words.intersection(
        a_words
    )

    return len(overlap) / max(
        1,
        len(q_words),
    )


def _has_reasonable_structure(
    answer: str,
    answer_format: str,
    results: List[Dict[str, Any]] | None = None,
) -> bool:

    answer = _text(answer)

    if not answer:
        return False

    words = _words(answer)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if answer_format == "comparison":

        lines = [
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        ]

        table_lines = [
            line
            for line in lines
            if line.startswith("|")
            and line.endswith("|")
        ]

        return (
            len(table_lines) >= 2
            and any(
                re.search(
                    r"\|\s*:?-{2,}:?\s*"
                    r"(?:\|\s*:?-{2,}:?\s*)+\|?",
                    line,
                )
                for line in table_lines
            )
        )

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    if answer_format == "numerical":

        return len(words) >= 20

    # --------------------------------------------------------
    # Lists / Algorithms / Classification
    # --------------------------------------------------------

    if answer_format == "algorithm":

        if len(words) < 60:
            return False

        numbered_steps = bool(
            re.search(
                r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
                answer,
            )
        )

        headings = bool(
            re.search(
                r"(?m)^(?:#{1,4}\s+|\*\*[^*\n]+\*\*)",
                answer,
            )
        )

        paragraphs = [
            p.strip()
            for p in re.split(
                r"\n\s*\n",
                answer,
            )
            if p.strip()
        ]

        # Accept a properly explained algorithm even when the local
        # model uses paragraphs instead of numbered steps.
        return (
            numbered_steps
            or headings
            or len(paragraphs) >= 2
        )

    if answer_format in {
        "list",
        "classification",
    }:

        return (
            len(words) >= 60
            and bool(
                re.search(
                    r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
                    answer,
                )
            )
        )

    # --------------------------------------------------------
    # Diagram
    # --------------------------------------------------------

    if answer_format == "diagram":
        return (
            len(words) >= 20
            and bool(re.search(r"(?:->|→|↓|←|\|)", answer))
        )

    # --------------------------------------------------------
    # Detailed answer types
    # --------------------------------------------------------

    if answer_format in {
        "process",
        "architecture",
        "explanation",
        "advantages_limitations",
        "general",
    }:

        paragraphs = [
            p.strip()
            for p in re.split(
                r"\n\s*\n",
                answer,
            )
            if p.strip()
        ]

        headings = re.findall(
            r"(?m)^(?:#{1,4}\s+|\*\*[^*\n]+\*\*)",
            answer,
        )

        if (
            len(words) >= 120
            and len(paragraphs) >= 2
        ):
            return True

        if (
            len(words) >= 80
            and headings
        ):
            return True

        return len(words) >= 60

    return len(words) >= 60



def _source_sentences(results: List[Dict[str, Any]]) -> List[str]:
    """Return clean factual sentences from the retrieved evidence."""
    out = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        text = _clean_text(result.get("text", ""))
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip(" -•\t")
            if len(_meaningful_words(sentence)) >= 4:
                out.append(sentence)
    return out


def _source_grounding_score(answer: str, results: List[Dict[str, Any]]) -> float:
    """
    Combine lexical overlap with semantic claim-to-evidence matching.

    Semantic matching reuses the cached embedding model used by RAG retrieval.
    It supplements, rather than replaces, lexical grounding.
    """
    answer_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", _text(answer))
        if len(_meaningful_words(s)) >= 4
    ]
    source_words = set()
    for sentence in _source_sentences(results):
        source_words.update(_meaningful_words(sentence))

    if not answer_sentences or not source_words:
        return 0.0

    evidence_sentences = _source_sentences(results)
    if not evidence_sentences:
        return 0.0

    lexical_support = []
    total = 0

    for sentence in answer_sentences:
        words = set(_meaningful_words(sentence))
        if not words:
            continue
        total += 1

        overlap = len(words & source_words) / max(1, len(words))
        lexical_support.append(overlap >= 0.30)

    if not total:
        return 0.0

    try:
        model = _load_model()
        embeddings = model.encode(
            answer_sentences + evidence_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        answer_embeddings = embeddings[:len(answer_sentences)]
        evidence_embeddings = embeddings[len(answer_sentences):]
        semantic_support = np.max(
            answer_embeddings @ evidence_embeddings.T,
            axis=1,
        ) >= 0.42
        supported = sum(
            lexical or semantic
            for lexical, semantic in zip(lexical_support, semantic_support)
        )
    except Exception:
        supported = sum(lexical_support)

    return supported / total


def _has_unsupported_specifics(answer: str, results: List[Dict[str, Any]]) -> bool:
    """
    Catch obvious fabricated numeric specifics without treating numbered
    headings/steps as factual numeric claims.
    """
    source = " ".join(
        _clean_text(r.get("text", ""))
        for r in results or []
        if isinstance(r, dict)
    ).lower()

    if not source:
        return True

    # Only check percentages, decimals, and four-digit years. Plain list
    # numbering (1., 2., 3.) is not evidence of a fabricated fact.
    answer_numbers = re.findall(
        r"(?<![A-Za-z])(?:\d+(?:\.\d+)%|\d+\.\d+%?|\b(?:19|20)\d{2}\b)",
        answer,
    )
    source_numbers = set(
        re.findall(
            r"(?<![A-Za-z])(?:\d+(?:\.\d+)%|\d+\.\d+%?|\b(?:19|20)\d{2}\b)",
            source,
        )
    )

    return any(number not in source_numbers for number in answer_numbers)


def _has_incomplete_ending(answer: str) -> bool:
    """Reject obvious dangling prose while allowing short complete answers."""
    text = _clean_answer(answer).rstrip()
    if not text:
        return True

    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    # Tables and code/diagram blocks can validly end without prose punctuation.
    if last_line.startswith("|") or re.fullmatch(r"[`~]+", last_line):
        return False

    ending = re.sub(r"[.!?;:,\s]+$", "", last_line).lower()
    dangling = re.compile(
        r"(?:\b(?:and|or|but|to|with|for|of|in|on|by|as|that|which|because|while|including|such as)"
        r"|\b(?:execute|perform|provide|ensure|allow|include|consist|depend|enable|support|contain))$"
    )
    return bool(dangling.search(ending))




def _is_usable_answer(
    question: str,
    answer: str,
    answer_format: str,
    results: List[Dict[str, Any]],
    attempt: int | str = "unknown",
    candidate: str = "unknown",
    draft_empty: bool | None = None,
    verifier_called: bool | None = None,
) -> bool:

    answer = _clean_answer(answer)
    word_count = len(_words(answer))
    log_context = (
        attempt,
        candidate,
        draft_empty,
        verifier_called,
    )

    if not answer:
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=empty word_count=0 draft_empty=%s verifier_called=%s',
            *log_context,
        )
        return False

    if _has_incomplete_ending(answer):
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=incomplete_ending word_count=%d draft_empty=%s verifier_called=%s',
            *log_context[:2], word_count, *log_context[2:],
        )
        return False

    if _looks_like_evidence_list(answer):
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=raw_evidence word_count=%d draft_empty=%s verifier_called=%s',
            *log_context[:2], word_count, *log_context[2:],
        )
        return False

    if _looks_like_meta_answer(answer):
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=meta_answer word_count=%d draft_empty=%s verifier_called=%s',
            *log_context[:2], word_count, *log_context[2:],
        )
        return False

    overlap = _answer_overlap(
        question,
        answer,
    )

    # The question-overlap check is only a sanity check. The evidence check
    # below is the important grounding gate because a correct academic answer
    # may naturally use terminology not present verbatim in the question.
    if overlap < 0.04:
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=question_overlap overlap=%.4f threshold=0.0400 word_count=%d '
            'draft_empty=%s verifier_called=%s',
            *log_context[:2], overlap, word_count, *log_context[2:],
        )
        return False

    if not _has_reasonable_structure(
        answer,
        answer_format,
        results,
    ):
        structure_thresholds = {
            "comparison": "table_lines>=2_and_separator",
            "numerical": "word_count>=20",
            "algorithm": "word_count>=60_and_steps_or_heading_or_two_paragraphs",
            "list": "word_count>=60_and_numbered_or_bulleted",
            "classification": "word_count>=60_and_numbered_or_bulleted",
            "diagram": "word_count>=20_and_arrow_or_pipe",
            "process": "word_count>=60_or_(word_count>=80_and_heading)_or_(word_count>=120_and_two_paragraphs)",
            "architecture": "word_count>=60_or_(word_count>=80_and_heading)_or_(word_count>=120_and_two_paragraphs)",
            "explanation": "word_count>=60_or_(word_count>=80_and_heading)_or_(word_count>=120_and_two_paragraphs)",
            "advantages_limitations": "word_count>=60_or_(word_count>=80_and_heading)_or_(word_count>=120_and_two_paragraphs)",
            "general": "word_count>=60_or_(word_count>=80_and_heading)_or_(word_count>=120_and_two_paragraphs)",
        }.get(answer_format, "word_count>=60")
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=structure_length answer_format=%s word_count=%d thresholds=%s '
            'draft_empty=%s verifier_called=%s',
            *log_context[:2], answer_format, word_count, structure_thresholds,
            *log_context[2:],
        )
        return False

    # Reject answers that are largely unrelated to the retrieved evidence.
    grounding_score = _source_grounding_score(answer, results)
    if grounding_score < 0.28:
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=source_grounding grounding_score=%.4f threshold=0.2800 '
            'word_count=%d draft_empty=%s verifier_called=%s',
            *log_context[:2], grounding_score, word_count, *log_context[2:],
        )
        return False

    if _has_unsupported_specifics(answer, results):
        logger.info(
            'phase=generator_candidate_rejected kind=long attempt=%s candidate=%s '
            'reason=unsupported_specifics word_count=%d draft_empty=%s verifier_called=%s',
            *log_context[:2], word_count, *log_context[2:],
        )
        return False

    return True


# ============================================================
# PUBLIC GENERATOR
# ============================================================

def generate_long_answer(
    question: str,
    results: List[Dict[str, Any]],
    history: str = "",
    requirements_context=None,
) -> str:
    """
    Generate a detailed answer from RAG V2 results.

    Public interface intentionally matches the existing AI Tutor.
    """

    question = _clean_text(question)

    if not question:
        _log_generation_result("long", requirements_context, "", "")
        return ""

    if not results:
        _log_generation_result("long", requirements_context, "", "")
        return ""

    context = build_long_context(results)

    if not context:
        _log_generation_result("long", requirements_context, "", "")
        return ""

    answer_format = detect_answer_format(question)
    logger.info('phase=generator_started kind=long')
    logger.info('phase=generator_attempt_started kind=long attempt=1')

    def finish(answer: str) -> str:
        _log_generation_result("long", requirements_context, answer_format, answer)
        return answer

    # --------------------------------------------------------
    # Attempt 1: generator -> verifier
    # --------------------------------------------------------
    draft = _call_generator(
        question=question,
        context=context,
        answer_format=answer_format,
        retry=False,
        history=history,
        requirements_context=requirements_context,
    )
    logger.info(
        'phase=generator_draft_generated kind=long attempt=1 empty=%s word_count=%d',
        not bool(_clean_answer(draft)),
        len(_words(_clean_answer(draft))),
    )

    verifier_called = bool(draft and context)
    verified = _verify_answer(
        question=question,
        draft_answer=draft,
        context=context,
        answer_format=answer_format,
        history=history,
        requirements_context=requirements_context,
    )
    logger.info(
        'phase=generator_verifier_completed kind=long attempt=1 '
        'verifier_called=%s result_empty=%s',
        verifier_called,
        not bool(_clean_answer(verified)),
    )

    validation_started = time.perf_counter()
    verified_usable = _is_usable_answer(
        question,
        verified,
        answer_format,
        results,
        attempt=1,
        candidate="verified",
        draft_empty=not bool(_clean_answer(draft)),
        verifier_called=verifier_called,
    )
    logger.info(
        'phase=generator_validation kind=long attempt=1 candidate=verified accepted=%s '
        'draft_empty=%s verifier_called=%s elapsed_s=%.3f',
        verified_usable,
        not bool(_clean_answer(draft)),
        verifier_called,
        time.perf_counter() - validation_started,
    )
    if verified_usable:
        return finish(_clean_answer(verified))

    # If the verifier is temporarily unavailable, keep the existing
    # grounding/structure gate as a safe fallback for a valid draft.
    validation_started = time.perf_counter()
    draft_usable = _is_usable_answer(
        question,
        draft,
        answer_format,
        results,
        attempt=1,
        candidate="draft",
        draft_empty=not bool(_clean_answer(draft)),
        verifier_called=verifier_called,
    )
    logger.info(
        'phase=generator_validation kind=long attempt=1 candidate=draft accepted=%s '
        'draft_empty=%s verifier_called=%s elapsed_s=%.3f',
        draft_usable,
        not bool(_clean_answer(draft)),
        verifier_called,
        time.perf_counter() - validation_started,
    )
    if draft_usable:
        return finish(_clean_answer(draft))

    # --------------------------------------------------------
    # Attempt 2: corrected generator -> verifier
    # --------------------------------------------------------
    logger.info('phase=generator_retry kind=long attempt=2')
    logger.info('phase=generator_attempt_started kind=long attempt=2')
    retry_draft = _call_generator(
        question=question,
        context=context,
        answer_format=answer_format,
        retry=True,
        history=history,
        requirements_context=requirements_context,
    )
    logger.info(
        'phase=generator_draft_generated kind=long attempt=2 empty=%s word_count=%d',
        not bool(_clean_answer(retry_draft)),
        len(_words(_clean_answer(retry_draft))),
    )

    retry_verifier_called = bool(retry_draft and context)
    retry_verified = _verify_answer(
        question=question,
        draft_answer=retry_draft,
        context=context,
        answer_format=answer_format,
        history=history,
        requirements_context=requirements_context,
    )
    logger.info(
        'phase=generator_verifier_completed kind=long attempt=2 '
        'verifier_called=%s result_empty=%s',
        retry_verifier_called,
        not bool(_clean_answer(retry_verified)),
    )

    validation_started = time.perf_counter()
    retry_verified_usable = _is_usable_answer(
        question,
        retry_verified,
        answer_format,
        results,
        attempt=2,
        candidate="verified",
        draft_empty=not bool(_clean_answer(retry_draft)),
        verifier_called=retry_verifier_called,
    )
    logger.info(
        'phase=generator_validation kind=long attempt=2 candidate=verified accepted=%s '
        'draft_empty=%s verifier_called=%s elapsed_s=%.3f',
        retry_verified_usable,
        not bool(_clean_answer(retry_draft)),
        retry_verifier_called,
        time.perf_counter() - validation_started,
    )
    if retry_verified_usable:
        return finish(_clean_answer(retry_verified))

    validation_started = time.perf_counter()
    retry_draft_usable = _is_usable_answer(
        question,
        retry_draft,
        answer_format,
        results,
        attempt=2,
        candidate="draft",
        draft_empty=not bool(_clean_answer(retry_draft)),
        verifier_called=retry_verifier_called,
    )
    logger.info(
        'phase=generator_validation kind=long attempt=2 candidate=draft accepted=%s '
        'draft_empty=%s verifier_called=%s elapsed_s=%.3f',
        retry_draft_usable,
        not bool(_clean_answer(retry_draft)),
        retry_verifier_called,
        time.perf_counter() - validation_started,
    )
    if retry_draft_usable:
        return finish(_clean_answer(retry_draft))

    # Do not return raw retrieved evidence as if it were an answer.
    return finish("")


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def generate_answer(
    question: str,
    results: List[Dict[str, Any]],
    history: str = "",
    requirements_context=None,
) -> str:
    """
    Compatibility wrapper in case another part of the LMS imports
    generate_answer from this module.
    """

    return generate_long_answer(
        question=question,
        results=results,
        history=history,
        requirements_context=requirements_context,
    )
