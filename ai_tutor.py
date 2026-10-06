"""
AI-LMS Agentic AI Tutor

Architecture:

    Student Question
          |
          v
    Query Understanding
          |
          v
    Context-Aware Query Understanding
          |
          v
    Course Detection
          |
          v
    Existing RAG V2
      FAISS + Lexical
          |
          v
    Generic Reranking + Evidence Role Priority
          |
          v
    Evidence Validation
          |
       sufficient?
        /      \
      NO        YES
      |          |
    Retry        |
      |          |
      +----------+
          |
          v
    Grounded Answer Generation
          |
          +--> Grounded Diagram Agent
          |
          v
    Grounding Check
          |
          v
    Final Tutor Answer

Design goals:
- Topic agnostic
- No subject-specific hard-coding
- Course-aware retrieval
- Conversation-aware follow-up questions
- Existing RAG V2 remains the retrieval authority
- Better definition retrieval
- Better evidence selection
- Grounded answers
- Source metadata preserved
- Existing generate_answer() interface preserved
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rag_v2_search import search as rag_v2_search

# Route directly to the grounded short/long generators.
from generator_short import generate_short_answer
from generator_long import generate_long_answer


# ============================================================
# COMMON HELPERS
# ============================================================

_STOP_WORDS = {
    "what", "is", "are", "was", "were",
    "the", "a", "an",
    "of", "in", "on", "for", "to",
    "and", "or", "with",
    "from", "by", "about",
    "explain", "describe", "discuss",
    "define", "meaning",
    "how", "why",
    "can", "could", "would", "should",
    "please", "give",
    "write", "briefly",
    "tell", "me",
    "does", "do", "did",
    "its", "their",
    "this", "that",
    "these", "those",
    "be", "being", "been",
    "into", "through",
    "using", "used", "use",
    "again",
}


_DEFINITION_OPENERS = (
    "is",
    "are",
    "was",
    "were",
    "refers to",
    "means",
    "denotes",
    "defined as",
    "known as",
    "called",
    "consists of",
    "comprises",
    "represents",
    "provides",
    "offers",
    "enables",
    "allows",
    "supports",
)


def _text(result: Dict[str, Any]) -> str:
    return str(
        result.get("text")
        or result.get("chunk_text")
        or result.get("content")
        or ""
    ).strip()


def _semantic_score(result: Dict[str, Any]) -> float:
    try:

        if result.get("semantic_score") is not None:
            return float(result["semantic_score"])

        if result.get("original_score") is not None:
            return float(result["original_score"])

        return float(
            result.get("score", 0.0)
        )

    except Exception:
        return 0.0



def _evidence_type(result: Dict[str, Any]) -> str:
    """Return the retrieval evidence role assigned by RAG V2."""
    value = str(result.get("evidence_type") or "primary").strip().lower()
    return value if value in {"primary", "supporting_context"} else "primary"


def _is_primary_evidence(result: Dict[str, Any]) -> bool:
    return _evidence_type(result) == "primary"


def _is_supporting_context(result: Dict[str, Any]) -> bool:
    return _evidence_type(result) == "supporting_context"


def _evidence_priority(result: Dict[str, Any]) -> float:
    """Primary evidence is authoritative; supporting context is secondary."""
    return 1.0 if _is_primary_evidence(result) else 0.35


def _ordered_evidence(
    results: List[Dict[str, Any]],
    max_primary: int = 6,
    max_supporting: int = 3,
) -> List[Dict[str, Any]]:
    """
    Keep direct evidence ahead of supporting context.

    Supporting context can enrich an answer but is never allowed to displace
    all direct evidence when direct evidence is available.
    """
    primary = [r for r in results if _is_primary_evidence(r)]
    supporting = [r for r in results if _is_supporting_context(r)]

    primary.sort(
        key=lambda r: (
            float(r.get("rerank_score", r.get("score", 0.0)) or 0.0),
            float(r.get("semantic_score", r.get("original_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    supporting.sort(
        key=lambda r: (
            float(r.get("rerank_score", r.get("score", 0.0)) or 0.0),
            float(r.get("context_score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    if not primary:
        return supporting[:max_supporting]

    return primary[:max_primary] + supporting[:max_supporting]


def _primary_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in results if _is_primary_evidence(r)]


def _clean_space(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()


def _words(text: str) -> List[str]:

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        str(text or "").lower()
    )

    return [
        token
        for token in tokens
        if token not in _STOP_WORDS
        and len(token) >= 2
    ]


def _word_set(text: str) -> set[str]:
    return set(
        _words(text)
    )


# ============================================================
# QUESTION TOPIC
# ============================================================

def _question_topic(question: str) -> str:
    """
    Extract the actual concept from common definition questions.

    Examples:

        What is Hadoop?
        -> Hadoop

        What is Lamport's algorithm?
        -> Lamport's algorithm

        Define mutual exclusion.
        -> mutual exclusion
    """

    q = _clean_space(
        question
    ).strip(" ?.!")


    patterns = (

        r"^\s*what\s+(?:is|are)\s+(.+?)\s*$",

        r"^\s*define\s+(.+?)\s*$",

        r"^\s*definition\s+of\s+(.+?)\s*$",

        r"^\s*meaning\s+of\s+(.+?)\s*$",

    )


    for pattern in patterns:

        match = re.match(
            pattern,
            q,
            flags=re.IGNORECASE
        )

        if match:

            return _clean_space(
                match.group(1)
            ).strip(" ?.!")


    return ""


# ============================================================
# QUESTION INTENT
# ============================================================

def _question_intent(question: str) -> str:
    q = _clean_space(question).lower()

    if re.search(
        r"\b(?:compare|comparison|differences?|differentiate|distinguish|"
        r"contrast|versus|vs\.?)\b",
        q,
    ):
        return "comparison"

    if re.search(
        r"\b(?:advantages?|disadvantages?|limitations?|benefits?|"
        r"merits|demerits|pros and cons)\b",
        q,
    ):
        return "advantages"

    if re.search(r"\b(?:algorithm|pseudocode)\b", q):
        return "algorithm"

    if re.search(
        r"\b(?:architecture|components?|system design)\b",
        q,
    ):
        return "architecture"

    if re.search(
        r"\b(?:list|enumerate|name|mention|classify|classification|"
        r"types? of|categories of)\b",
        q,
    ):
        return "list"

    if re.search(r"\b(?:example|examples|instance)\b", q):
        return "example"

    if re.search(
        r"\b(?:how|working|steps?|procedure|process|workflow)\b",
        q,
    ):
        return "how"

    if re.search(r"\b(?:why|reason|purpose)\b", q):
        return "why"

    if re.search(
        r"\b(?:what\s+(?:is|are)|define|definition\s+of|meaning\s+of)\b",
        q,
    ):
        return "definition"

    if re.search(
        r"\b(?:explain|describe|discuss|elaborate|illustrate)\b",
        q,
    ):
        return "explanation"

    return "general"


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def _split_sentences(text: str) -> List[str]:
    """
    Split real sentences without treating PDF line-wrapping as a
    sentence boundary.

    Many course PDFs contain line breaks in the middle of a sentence,
    for example:

        Hadoop is a software framework for writing distributed
        applications.

    The old splitter treated those two PDF lines as two separate
    sentences and could return only the first fragment.  We preserve
    genuine bullet/list boundaries, but join ordinary wrapped lines.
    """

    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")

    # Preserve common bullet/list boundaries before joining PDF
    # line-wrapped text.
    bullet_break = re.compile(
        r"\n(?=\s*(?:[•*¢©]|e\s+|[-–]\s+|\d+(?:\.\d+)*\s+))",
        flags=re.IGNORECASE,
    )

    text = bullet_break.sub("<<SENTENCE_BREAK>>", text)

    # A newline in the middle of a PDF sentence is only formatting.
    text = re.sub(r"\s*\n\s*", " ", text)

    # Restore list boundaries as a real separator.
    text = text.replace("<<SENTENCE_BREAK>>", "\n")

    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    result = []

    for piece in pieces:
        piece = _clean_space(piece)
        if piece:
            result.append(piece)

    return result


# ============================================================
# FOLLOW-UP QUESTION DETECTION
# ============================================================

_FOLLOW_UP_PATTERNS = [

    r"^\s*why\s*\??$",

    r"^\s*how\s*\??$",

    r"^\s*what\s+about\s+(?:this|that)\s*\??$",

    r"^\s*explain\s+(?:this|that)\s*(?:again)?\s*$",

    r"^\s*explain\s+again\s*$",

    r"^\s*what\s+does\s+(?:this|that)\s+mean\s*\??$",

    r"^\s*what\s+happens\s+next\s*\??$",

    r"^\s*give\s+(?:me\s+)?an?\s+example\s*$",

    r"^\s*give\s+an?\s+example\s+of\s+(?:this|that)\s*$",

    r"^\s*why\s+is\s+(?:this|that)\s+needed\s*\??$",

    r"^\s*how\s+does\s+(?:this|that)\s+work\s*\??$",

    r"^\s*can\s+you\s+explain\s+(?:this|that)\s*$",

]


def _is_follow_up(
    question: str
) -> bool:

    q = _clean_space(
        question
    ).lower()


    for pattern in _FOLLOW_UP_PATTERNS:

        if re.match(
            pattern,
            q
        ):

            return True


    meaningful_words = _words(
        q
    )


    # Very short questions are usually contextual.
    if len(meaningful_words) <= 3:

        return True


    return False


# ============================================================
# HISTORY EXTRACTION
# ============================================================

def _extract_history_text(
    history: str
) -> str:

    history = _clean_space(
        history
    )

    if not history:
        return ""


    # Keep only the most recent portion.
    #
    # This prevents extremely long conversations from polluting
    # the retrieval query.
    if len(history) > 5000:

        history = history[-5000:]


    return history


# ============================================================
# EXTRACT IMPORTANT CONCEPTS FROM HISTORY
# ============================================================

def _extract_history_concepts(
    history: str
) -> List[str]:

    history = _extract_history_text(
        history
    )

    if not history:
        return []


    concepts = []


    # Prefer phrases from previous questions.
    question_matches = re.findall(
        r"(?:user|student|question)\s*[:\-]\s*(.+?)(?=(?:assistant|ai|answer)\s*[:\-]|$)",
        history,
        flags=re.IGNORECASE,
    )


    candidate_texts = (
        question_matches
        if question_matches
        else [history]
    )


    for candidate in candidate_texts:

        candidate = _clean_space(
            candidate
        )

        topic = _question_topic(
            candidate
        )

        if topic:

            concepts.append(
                topic
            )

            continue


        tokens = _words(
            candidate
        )

        concepts.extend(
            tokens
        )


    # Remove duplicates.
    unique = []

    seen = set()

    for concept in concepts:

        concept = _clean_space(
            concept
        ).strip(
            ".,!?;:"
        )

        if not concept:
            continue


        key = concept.lower()


        if key in seen:
            continue


        seen.add(
            key
        )

        unique.append(
            concept
        )


    return unique[-12:]


# ============================================================
# BUILD CONTEXT-AWARE SEARCH QUERY
# ============================================================

def _build_contextual_query(
    question: str,
    history: str = "",
) -> str:

    question = _clean_space(
        question
    )

    if not question:
        return ""


    if not _is_follow_up(
        question
    ):

        # IMPORTANT:
        # Normal questions are sent unchanged to RAG.
        return question


    history = _extract_history_text(
        history
    )


    if not history:

        return question


    concepts = _extract_history_concepts(
        history
    )


    if not concepts:

        return question


    # Keep the student's actual question first.
    #
    # The previous concepts are only retrieval context.
    contextual_query = (
        f"{question}. "
        f"Previous conversation concepts: "
        f"{', '.join(concepts)}"
    )


    return contextual_query


# ============================================================
# DEFINITION EVIDENCE SCORING
# ============================================================

def _definition_sentence_quality(
    sentence: str,
    topic: str,
) -> float:

    sentence = _clean_space(
        sentence
    )

    topic = _clean_space(
        topic
    )


    if not sentence or not topic:
        return 0.0


    low = sentence.lower()
    topic_low = topic.lower()


    if topic_low not in low:
        return 0.0


    score = 0.0


    # --------------------------------------------------------
    # Direct subject
    # --------------------------------------------------------

    direct_subject = re.search(
        rf"^\s*(?:the\s+)?"
        rf"{re.escape(topic_low)}\s+"
        rf"(?:is|are|was|were|"
        rf"refers\s+to|means|denotes|"
        rf"is\s+defined\s+as|"
        rf"are\s+defined\s+as|"
        rf"is\s+known\s+as|"
        rf"are\s+known\s+as|"
        rf"is\s+called|"
        rf"are\s+called|"
        rf"consists\s+of|comprises|"
        rf"represents|provides|"
        rf"offers|enables|"
        rf"allows|supports)\b",
        low,
    )


    if direct_subject:

        score += 0.75


    # --------------------------------------------------------
    # OCR-friendly subject
    # --------------------------------------------------------

    near_subject = re.search(
        rf"^\s*"
        rf"(?:\d+(?:\.\d+)*\s+|"
        rf"[a-z]\)\s+|"
        rf"[-•]\s+)?"
        rf"(?:the\s+)?"
        rf"{re.escape(topic_low)}\s+"
        rf"(?:is|are|was|were|"
        rf"refers\s+to|means|denotes|"
        rf"defined\s+as|known\s+as|"
        rf"called|consists\s+of|"
        rf"comprises|represents|"
        rf"provides|offers|"
        rf"enables|allows|supports)\b",
        low,
    )


    if near_subject:

        score += 0.65


    # --------------------------------------------------------
    # Concept near definition verb
    # --------------------------------------------------------

    if re.search(
        rf"\b{re.escape(topic_low)}\b"
        rf".{{0,70}}"
        rf"\b(?:is|are|was|were|"
        rf"refers\s+to|means|denotes|"
        rf"defined\s+as|known\s+as|"
        rf"called|consists\s+of|"
        rf"comprises|represents|"
        rf"provides|offers|"
        rf"enables|allows|supports)\b",
        low,
    ):

        score += 0.18


    # --------------------------------------------------------
    # Penalize another concept being defined first
    # --------------------------------------------------------

    another_concept_definition = re.match(
        r"^\s*(?:the\s+)?"
        r"[a-z][a-z0-9_-]{1,40}\s+"
        r"(?:is|are|was|were|"
        r"refers\s+to|means|"
        r"defined\s+as|known\s+as|"
        r"called|consists\s+of|"
        r"comprises|represents)\b",
        low,
    )


    topic_is_subject = re.match(
        rf"^\s*(?:the\s+)?"
        rf"{re.escape(topic_low)}\b",
        low,
    )


    if (
        another_concept_definition
        and not topic_is_subject
    ):

        score -= 0.45


    # --------------------------------------------------------
    # Reasonable sentence length
    # --------------------------------------------------------

    word_count = len(
        sentence.split()
    )


    if word_count >= 8:

        score += 0.08


    if word_count >= 14:

        score += 0.04


    # --------------------------------------------------------
    # Dangling OCR fragment
    # --------------------------------------------------------

    if re.search(
        r"\b(?:and|or|the|a|an|"
        r"of|to|from|with|for|"
        r"is|are|called|known)\s*$",
        low,
    ):

        score -= 0.35


    return max(
        0.0,
        min(
            score,
            1.0
        )
    )


# ============================================================
# DIRECT DEFINITION EXTRACTION
# ============================================================

def _best_direct_definition(
    question: str,
    results: List[Dict[str, Any]],
) -> str:

    topic = _question_topic(
        question
    )


    if not topic:
        return ""


    candidates = []


    for result_index, result in enumerate(
        results
    ):

        text = _text(
            result
        )


        sentences = _split_sentences(
            text
        )


        for sentence_index, sentence in enumerate(
            sentences
        ):

            sentence = _clean_space(
                sentence
            )


            quality = _definition_sentence_quality(
                sentence,
                topic,
            )


            if quality <= 0:
                continue


            # The requested concept should actually begin the
            # definition. This prevents related concepts from
            # being selected accidentally.
            starts_with_topic = re.match(
                rf"^\s*"
                rf"(?:[•*e¢©]\s*)?"
                rf"(?:\d+(?:\.\d+)*\s+)?"
                rf"(?:the\s+)?"
                rf"{re.escape(topic.lower())}\b",
                sentence.lower(),
            )


            if not starts_with_topic:
                continue


            candidates.append(
                (
                    quality,
                    _semantic_score(
                        result
                    ),
                    -result_index,
                    -sentence_index,
                    sentence,
                )
            )


    if not candidates:

        return ""


    candidates.sort(
        reverse=True
    )


    return candidates[0][-1]


# ============================================================
# FOCUSED DEFINITION RESULTS
# ============================================================

def _definition_focused_results(
    question: str,
    results: List[Dict[str, Any]],
    limit: int = 6,
) -> List[Dict[str, Any]]:

    topic = _question_topic(
        question
    )


    if not topic:

        return results[:limit]


    focused = []


    for result in results:

        text = _text(
            result
        )


        scored = []


        for index, sentence in enumerate(
            _split_sentences(text)
        ):

            quality = _definition_sentence_quality(
                sentence,
                topic,
            )


            if quality > 0:

                scored.append(
                    (
                        quality,
                        index,
                        sentence,
                    )
                )


        scored.sort(
            reverse=True,
            key=lambda item: (
                item[0],
                -item[1],
            ),
        )


        if scored:

            quality, _, sentence = scored[0]


            item = dict(
                result
            )


            item["text"] = sentence


            item[
                "_definition_evidence_score"
            ] = round(
                quality,
                6,
            )


            focused.append(
                item
            )


    if focused:

        focused.sort(
            key=lambda item: (
                _evidence_priority(item),
                float(
                    item.get(
                        "_definition_evidence_score",
                        0.0,
                    )
                ),
                _semantic_score(item),
            ),
            reverse=True,
        )

        return _ordered_evidence(
            focused,
            max_primary=limit,
            max_supporting=0,
        )[:limit]


    return _ordered_evidence(
        results,
        max_primary=limit,
        max_supporting=0,
    )[:limit]


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    output = []

    seen = set()


    for result in results:

        if not isinstance(
            result,
            dict
        ):

            continue


        key = (
            str(
                result.get(
                    "course_id",
                    ""
                )
            ),
            str(
                result.get(
                    "material_id",
                    ""
                )
            ),
            str(
                result.get(
                    "page",
                    ""
                )
            ),
            _clean_space(
                _text(result)
            )[:500].lower(),
        )


        if key in seen:

            continue


        seen.add(
            key
        )


        output.append(
            result
        )


    return output


# ============================================================
# QUERY UNDERSTANDING AGENT
# ============================================================

class QueryUnderstandingAgent:

    name = "Query Understanding Agent"


    def run(
        self,
        question: str
    ) -> Dict[str, Any]:

        question = _clean_space(
            question
        )


        if not question:

            return {
                "question": "",
                "answer_type": "short",
                "intent": "general",
                "retrieval_k": 10,
                "retry_k": 16,
                "search_query": "",
            }


        q = question.lower()


        intent = _question_intent(
            question
        )


        # ----------------------------------------------------
        # Answer size
        # ----------------------------------------------------
        if re.search(
            r"\b(?:8|10|12|13|15|16)\s*[- ]?marks?\b|"
            r"\b(?:in\s+detail|in\s+depth|detailed|deeply|elaborate|"
            r"long\s+answer)\b",
            q,
        ):

            answer_type = "detailed"
            retrieval_k = 12
            retry_k = 18

        elif re.search(
            r"\b(?:2)\s*[- ]?marks?\b|\b(?:short\s+answer|briefly)\b",
            q,
        ):

            answer_type = "short"
            retrieval_k = 10
            retry_k = 16

        elif re.search(r"\b5\s*[- ]?marks?\b", q) or intent in {
            "explanation",
            "comparison",
            "how",
            "algorithm",
            "architecture",
            "list",
            "advantages",
            "why",
            "example",
        }:

            answer_type = "medium"
            retrieval_k = 10
            retry_k = 16

        else:

            answer_type = "short"
            retrieval_k = 10
            retry_k = 16


        return {
            "question": question,
            "answer_type": answer_type,
            "intent": intent,
            "retrieval_k": retrieval_k,
            "retry_k": retry_k,
            "search_query": question,
        }


    def improve_query(
        self,
        question: str
    ) -> str:

        topic = _question_topic(
            question
        )


        if topic and _question_intent(question) == "definition":
            return f"{topic} definition"

        cleaned = _words(question)
        if cleaned:
            return " ".join(cleaned)

        return _clean_space(question)


# ============================================================
# COURSE DETECTION AGENT
# ============================================================

class CourseDetectionAgent:

    name = "Course Detection Agent"

    def run(
        self,
        question: str
    ) -> Optional[Dict[str, Any]]:

        try:

            results = rag_v2_search(
                query=question,
                course_id=None,
                material_id=None,
                top_k=16,
            )

        except Exception:

            return None


        if not results:

            return None


        course_scores: Dict[str, float] = {}


        for result in results:

            course_id = str(
                result.get(
                    "course_id",
                    ""
                )
            ).strip()


            if not course_id:

                continue


            score = _semantic_score(
                result
            )


            # Generic word overlap.
            q_words = _word_set(
                question
            )

            r_words = _word_set(
                _text(result)
            )


            if q_words:

                overlap = (
                    len(
                        q_words & r_words
                    )
                    / max(
                        1,
                        len(q_words)
                    )
                )


                score += min(
                    0.12,
                    overlap * 0.12
                )


            course_scores[
                course_id
            ] = max(
                course_scores.get(
                    course_id,
                    0.0,
                ),
                score,
            )


        if not course_scores:

            return None


        best_course = max(
            course_scores,
            key=course_scores.get,
        )


        best_score = course_scores[
            best_course
        ]


        if best_score < 0.30:

            return None


        return {
            "course_id": best_course,
            "score": float(
                best_score
            ),
            "course_scores": course_scores,
        }


# ============================================================
# RETRIEVAL AGENT
# ============================================================

class RetrievalAgent:

    name = "Retrieval Agent"


    def run(
        self,
        query: str,
        course_id: Optional[str] = None,
        material_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:

        if not query:

            return []


        try:

            results = rag_v2_search(
                query=query,
                course_id=course_id,
                material_id=material_id,
                top_k=top_k,
            )

        except Exception:

            return []


        cleaned = []


        for result in results or []:

            if (
                isinstance(result, dict)
                and _text(result)
            ):

                cleaned.append(
                    dict(result)
                )


        return _deduplicate_results(
            cleaned
        )


# ============================================================
# RERANKING AGENT
# ============================================================

class RerankingAgent:

    name = "Reranking Agent"


    def run(
        self,
        question: str,
        results: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:

        if not results:

            return []


        q_words = _word_set(
            question
        )


        q_lower = _clean_space(
            question
        ).lower()


        topic = _question_topic(
            question
        )


        intent = _question_intent(
            question
        )


        reranked = []


        for index, result in enumerate(
            results
        ):

            item = dict(
                result
            )


            text = _text(
                item
            )


            text_lower = text.lower()


            semantic = _semantic_score(
                result
            )


            # ------------------------------------------------
            # Word overlap
            # ------------------------------------------------

            text_words = _word_set(
                text
            )


            overlap = (
                len(
                    q_words & text_words
                )
                / max(
                    1,
                    len(q_words)
                )
                if q_words
                else 0.0
            )


            # ------------------------------------------------
            # Exact phrase
            # ------------------------------------------------

            phrase_match = 0.0


            if (
                topic
                and topic.lower() in text_lower
            ):

                phrase_match = 1.0


            elif (
                q_lower
                and q_lower in text_lower
            ):

                phrase_match = 1.0


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = " ".join(
                str(
                    item.get(
                        key,
                        ""
                    )
                    or ""
                )
                for key in (
                    "topic",
                    "topics",
                    "section",
                    "filename",
                )
            ).lower()


            metadata_overlap = (
                len(
                    q_words
                    & _word_set(
                        metadata
                    )
                )
                / max(
                    1,
                    len(q_words)
                )
                if q_words
                else 0.0
            )


            # ------------------------------------------------
            # Base score
            # ------------------------------------------------

            score = (
                semantic * 0.58
                + overlap * 0.20
                + phrase_match * 0.12
                + metadata_overlap * 0.10
            )


            # ------------------------------------------------
            # Evidence role
            # ------------------------------------------------
            # Direct evidence remains authoritative. Supporting context
            # enriches the answer but must not outrank direct evidence
            # simply because it has lexical similarity.
            evidence_type = _evidence_type(item)
            if evidence_type == "primary":
                score += 0.08
            else:
                score -= 0.03
                score += min(
                    0.05,
                    max(
                        0.0,
                        float(item.get("context_score", 0.0) or 0.0),
                    ) * 0.05,
                )


            # ------------------------------------------------
            # Definition relevance
            # ------------------------------------------------

            definition_score = 0.0


            if (
                intent == "definition"
                and topic
            ):

                for sentence in _split_sentences(
                    text
                ):

                    definition_score = max(
                        definition_score,
                        _definition_sentence_quality(
                            sentence,
                            topic,
                        ),
                    )


                score += (
                    definition_score
                    * 0.65
                )


                # Penalize chunks that only mention the
                # concept without actually defining it.
                if (
                    topic.lower()
                    in text_lower
                    and definition_score < 0.25
                ):

                    score -= 0.08


            # ------------------------------------------------
            # Intent-specific signals
            # ------------------------------------------------

            if intent == "comparison":

                if re.search(
                    r"\b(?:difference|similar|"
                    r"whereas|while|versus|vs)\b",
                    text_lower,
                ):

                    score += 0.08


            elif intent in {"how", "algorithm"}:

                if re.search(
                    r"\b(?:step|process|procedure|"
                    r"working|algorithm|phase)\b",
                    text_lower,
                ):

                    score += 0.06


            elif intent == "architecture":

                if re.search(
                    r"\b(?:component|role|relationship|architecture|"
                    r"structure|layer|module)\b",
                    text_lower,
                ):

                    score += 0.06


            elif intent == "list":

                if re.search(
                    r"\b(?:types?|categories|classification|include|"
                    r"following|consists of)\b",
                    text_lower,
                ):

                    score += 0.06


            elif intent == "advantages":

                if re.search(
                    r"\b(?:advantage|benefit|"
                    r"limitation|disadvantage)\b",
                    text_lower,
                ):

                    score += 0.06


            elif intent == "why":

                if re.search(
                    r"\b(?:because|reason|"
                    r"purpose|allows|enables)\b",
                    text_lower,
                ):

                    score += 0.05


            elif intent == "example":

                if re.search(
                    r"\b(?:example|instance|such as)\b",
                    text_lower,
                ):

                    score += 0.06


            item[
                "original_score"
            ] = semantic


            item[
                "rerank_score"
            ] = round(
                score,
                6,
            )


            item[
                "_definition_evidence_score"
            ] = round(
                definition_score,
                6,
            )


            item[
                "_rerank_index"
            ] = index


            reranked.append(
                item
            )


        # ----------------------------------------------------
        # Normal ordering
        # ----------------------------------------------------

        reranked.sort(
            key=lambda x: (
                x.get(
                    "rerank_score",
                    0.0,
                ),
                x.get(
                    "original_score",
                    0.0,
                ),
                -x.get(
                    "_rerank_index",
                    0,
                ),
            ),
            reverse=True,
        )


        # ----------------------------------------------------
        # Definition ordering
        # ----------------------------------------------------

        if (
            intent == "definition"
            and topic
        ):

            reranked.sort(
                key=lambda x: (
                    x.get(
                        "_definition_evidence_score",
                        0.0,
                    ),
                    x.get(
                        "rerank_score",
                        0.0,
                    ),
                    x.get(
                        "original_score",
                        0.0,
                    ),
                    -x.get(
                        "_rerank_index",
                        0,
                    ),
                ),
                reverse=True,
            )


        return reranked[
            :max(
                1,
                int(top_k)
            )
        ]


# ============================================================
# VALIDATION AGENT
# ============================================================

class ValidationAgent:

    name = "Validation Agent"


    def __init__(
        self,
        minimum_semantic_score: float = 0.45,
        minimum_results: int = 1,
    ):

        self.minimum_semantic_score = (
            minimum_semantic_score
        )

        self.minimum_results = (
            minimum_results
        )


    def run(
        self,
        question: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not results:

            return {
                "sufficient": False,
                "reason": (
                    "No relevant course evidence "
                    "was retrieved."
                ),
                "best_score": 0.0,
                "evidence_count": 0,
            }


        semantic_scores = [
            _semantic_score(item)
            for item in results
        ]

        primary_results = _primary_results(results)
        primary_scores = [
            _semantic_score(item)
            for item in primary_results
        ]

        # Supporting context alone must not make weak retrieval appear
        # sufficient. Validation therefore prefers direct evidence.
        best_score = max(
            primary_scores if primary_scores else semantic_scores,
            default=0.0,
        )


        intent = _question_intent(
            question
        )


        topic = _question_topic(
            question
        )


        # ----------------------------------------------------
        # Definition validation
        # ----------------------------------------------------

        if (
            intent == "definition"
            and topic
        ):

            definition_pool = primary_results or results

            definition_scores = [
                float(
                    item.get(
                        "_definition_evidence_score",
                        0.0,
                    )
                )
                for item in definition_pool
            ]


            best_definition = max(
                definition_scores,
                default=0.0,
            )


            sufficient = (
                len(results)
                >= self.minimum_results
                and (
                    best_score
                    >= self.minimum_semantic_score
                    or best_definition
                    >= 0.45
                )
            )


        else:

            sufficient = (
                len(results)
                >= self.minimum_results
                and best_score
                >= self.minimum_semantic_score
            )


        return {
            "sufficient": bool(
                sufficient
            ),
            "reason": (
                "Relevant course evidence found."
                if sufficient
                else
                "Retrieved evidence is too weak "
                "to reliably answer the question."
            ),
            "best_score": float(
                best_score
            ),
            "evidence_count": len(
                results
            ),
        }




# ============================================================
# GROUNDED DIAGRAM AGENT
# ============================================================


def _diagram_requested(question: str) -> bool:
    q = _clean_space(question).lower()
    return bool(re.search(
        r"\b(?:diagram|flowchart|flow chart|architecture diagram|block diagram|"
        r"sequence diagram|draw|sketch|illustrate a diagram|"
        r"graphical representation|visualize|visualise)\b",
        q,
        flags=re.IGNORECASE,
    ))


def _diagram_label(value: str, max_words: int = 8) -> str:
    value = _clean_space(value)
    value = re.sub(r'[\[\]{}<>#"`]', '', value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;-–—")
    words = value.split()
    if len(words) > max_words:
        value = " ".join(words[:max_words]) + "…"
    return value[:90].strip()


def _unique_preserve(values: List[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        value = _diagram_label(value)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


class GroundedDiagramAgent:
    """Build a Mermaid diagram only from relationships stated in retrieved evidence."""

    name = "Grounded Diagram Agent"

    _RELATION_PATTERNS = (
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:is|are)\s+followed\s+by\s+(?P<b>[A-Za-z][A-Za-z0-9 /_-]{1,60})", "followed by"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:sends?|passes?)\s+(?:data|information|request|requests)?\s*(?:to)\s+(?P<b>[A-Za-z][A-Za-z0-9 /_-]{1,60})", "sends to"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:communicates|connects)\s+(?:with|to)\s+(?P<b>[A-Za-z][A-Za-z0-9 /_-]{1,60})", "communicates with"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:contains|includes|comprises)\s+(?P<b>[A-Za-z][A-Za-z0-9 /_,-]{1,80})", "contains"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:consists of)\s+(?P<b>[A-Za-z][A-Za-z0-9 /_,-]{1,80})", "consists of"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:is|are)\s+divided\s+into\s+(?P<b>[A-Za-z][A-Za-z0-9 /_,-]{1,80})", "divided into"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:stores?|stores data in)\s+(?P<b>[A-Za-z][A-Za-z0-9 /_-]{1,60})", "stores in"),
        (r"(?P<a>[A-Za-z][A-Za-z0-9 /_-]{1,60})\s+(?:processed|handled)\s+by\s+(?P<b>[A-Za-z][A-Za-z0-9 /_-]{1,60})", "processed by"),
    )

    _ENUMERATED_LAYER = re.compile(
        r"\((?P<ordinal>iv|iii|ii|i|[1-9]\d*)[)}]\s*"
        r"(?P<label>[^.;\n]{1,140}?\blayer\b)",
        re.IGNORECASE,
    )

    @staticmethod
    def _layer_label(raw_label: str, evidence_text: str) -> str:
        label = re.sub(r"\blayer\b.*$", "", raw_label, flags=re.IGNORECASE)
        label = re.sub(r"^\s*(?:APIs?|interfaces?)\s+at\s+", "", label, flags=re.IGNORECASE)
        label = _clean_space(label).strip(" ,:-")
        low = label.lower()
        evidence = evidence_text.lower()

        if "application" in low and "support" in low:
            return "Application Support Layer"
        if "processing" in low and "framework" in low:
            component = " - MapReduce" if re.search(r"\bmapreduce\b", evidence) else ""
            return f"Processing Framework{component} Layer"
        if "resource" in low and "manager" in low:
            component = " - YARN" if re.search(r"\byarn\b", evidence) else ""
            return f"Resource Manager{component} Layer"
        if "distributed" in low and "storage" in low:
            component = " - HDFS" if re.search(r"\bhdfs\b", evidence) else ""
            return f"Distributed Storage{component} Layer"

        return _diagram_label(f"{label.title()} Layer", max_words=8)

    def _derived_layer_diagram(
        self,
        evidence_results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        evidence_text = "\n".join(_text(result) for result in evidence_results)
        layer_list = re.search(
            r"\b(?P<count>three|four|five|six|3|4|5|6)\s+layers?\b"
            r"[\s\S]{0,100}?\bas follows\b(?P<items>[\s\S]{0,700})",
            evidence_text,
            re.IGNORECASE,
        )
        if not layer_list:
            return None

        listed_text = layer_list.group("items")
        expected_count = {
            "three": 3, "3": 3, "four": 4, "4": 4,
            "five": 5, "5": 5, "six": 6, "6": 6,
        }[layer_list.group("count").lower()]
        ordered_layers = []
        for match in self._ENUMERATED_LAYER.finditer(listed_text):
            label = self._layer_label(match.group("label"), evidence_text)
            if label and label.lower() not in {value.lower() for value in ordered_layers}:
                ordered_layers.append(label)
            if len(ordered_layers) >= expected_count:
                break

        if len(ordered_layers) < 3 or len(ordered_layers) != expected_count:
            return None

        top_down_layers = list(reversed(ordered_layers))
        has_client_application_relation = bool(re.search(
            r"\bclient\b.{0,100}\bhosts?\b.{0,100}\bapplications?\b",
            evidence_text,
            flags=re.IGNORECASE | re.DOTALL,
        ))
        nodes = []
        if has_client_application_relation:
            nodes.append("Client")
        nodes.extend(top_down_layers)

        edges = [
            {"from": source, "to": target, "relation": "listed architecture layer order"}
            for source, target in zip(nodes, nodes[1:])
        ]

        node_ids = {node: f"N{index}" for index, node in enumerate(nodes, 1)}
        lines = ["flowchart TD"]
        for node in nodes:
            lines.append(f'    {node_ids[node]}["{node}"]')
        for edge in edges:
            lines.append(f"    {node_ids[edge['from']]} --> {node_ids[edge['to']]}")

        cross_layer_notes = []
        if re.search(r"\bAVRO\b.{0,100}\bserialization\b.{0,100}\blayers?\b", evidence_text, re.IGNORECASE | re.DOTALL):
            cross_layer_notes.append("AVRO: data serialization between layers")
        if re.search(r"\bZookeeper\b.{0,100}\bcoordination\b.{0,100}\blayer components\b", evidence_text, re.IGNORECASE | re.DOTALL):
            cross_layer_notes.append("Zookeeper: coordination among layer components")
        if cross_layer_notes:
            lines.append('    subgraph CrossLayer["Cross-layer functions"]')
            for index, note in enumerate(cross_layer_notes, len(nodes) + 1):
                note_id = f"N{index}"
                nodes.append(note)
                lines.append(f'        {note_id}["{note}"]')
            lines.append("    end")

        return {
            "requested": True,
            "grounded": True,
            "derived": True,
            "label": "Diagram derived from the course material",
            "type": "derived_flowchart",
            "mermaid": "\n".join(lines),
            "nodes": nodes,
            "edges": edges,
            "evidence_count": len(evidence_results),
            "reason": "The layer sequence and relationships were derived only from retrieved course evidence.",
        }

    def run(self, question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not _diagram_requested(question):
            return {"requested": False, "grounded": False, "type": None,
                    "mermaid": "", "nodes": [], "edges": [],
                    "evidence_count": 0, "reason": "No diagram was requested."}

        evidence_results = [r for r in results if _evidence_type(r) == "primary"] or list(results)
        derived_layer_diagram = self._derived_layer_diagram(evidence_results)
        if derived_layer_diagram:
            return derived_layer_diagram

        edges = []
        nodes = []
        seen_edges = set()

        for result in evidence_results[:8]:
            for sentence in _split_sentences(_text(result)):
                sentence = _clean_space(sentence)
                if not sentence:
                    continue
                for pattern, relation in self._RELATION_PATTERNS:
                    match = re.search(pattern, sentence, flags=re.IGNORECASE)
                    if not match:
                        continue
                    a = _diagram_label(match.group("a"))
                    b = _diagram_label(match.group("b"))
                    if not a or not b or a.lower() == b.lower():
                        continue
                    key = (a.lower(), b.lower(), relation)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append({"from": a, "to": b, "relation": relation,
                                  "page": result.get("page"),
                                  "filename": result.get("filename")})
                    nodes.extend([a, b])
                    if len(edges) >= 8:
                        break
                if len(edges) >= 8:
                    break
            if len(edges) >= 8:
                break

        nodes = _unique_preserve(nodes)[:12]
        if not edges:
            return {"requested": True, "grounded": False, "type": "diagram",
                    "mermaid": "", "nodes": nodes, "edges": [],
                    "evidence_count": len(evidence_results),
                    "reason": "The retrieved course evidence does not contain enough explicit relationships to build a grounded diagram safely."}

        node_ids = {node.lower(): f"N{i}" for i, node in enumerate(nodes, 1)}
        lines = ["flowchart TD"]
        for node in nodes:
            safe_label = node.replace('"', "'")
            lines.append(f'    {node_ids[node.lower()]}["{safe_label}"]')
        for edge in edges:
            a_id = node_ids.get(edge["from"].lower())
            b_id = node_ids.get(edge["to"].lower())
            if a_id and b_id:
                relation = edge["relation"].replace("|", "/")
                lines.append(f"    {a_id} -->|{relation}| {b_id}")

        return {"requested": True, "grounded": True, "derived": True,
            "label": "Diagram derived from the course material", "type": "derived_flowchart",
                "mermaid": "\n".join(lines), "nodes": nodes, "edges": edges,
                "evidence_count": len(evidence_results),
            "reason": "Diagram relationships were derived only from retrieved course evidence."}


def _diagram_grounded_summary(question: str, diagram: Dict[str, Any]) -> str:
    title = _clean_space(question)
    title = re.sub(
        r"^\s*(?:please\s+)?(?:draw|show|create|sketch|illustrate)\s+"
        r"(?:(?:and|to)\s+)?(?:explain\s+)?",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"^\s*(?:a|an|the)\s+(?:diagram|flowchart|figure)\s+(?:of\s+)?",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+(?:diagram|flowchart|figure)\s*$", "", title, flags=re.IGNORECASE)
    title = title.strip(" .?!").title() or "Architecture"

    nodes = [str(node) for node in diagram.get("nodes", [])]
    layers = [node for node in nodes if re.search(r"\blayer\b$", node, re.IGNORECASE)]
    facts = []

    if layers:
        facts.append(
            "The course material describes these layers: "
            + ", ".join(layers)
            + "."
        )

    if any(
        edge.get("from") == "Client"
        and "Application Support Layer" in str(edge.get("to", ""))
        for edge in diagram.get("edges", [])
    ):
        facts.append("The Client hosts applications using Hadoop ecosystem projects.")

    for node in nodes:
        if node.startswith("AVRO:"):
            facts.append("AVRO enables " + node.partition(":")[2].strip() + ".")
        elif node.startswith("Zookeeper:"):
            facts.append("Zookeeper enables " + node.partition(":")[2].strip() + ".")

    if not facts:
        for edge in diagram.get("edges", []):
            relation = edge.get("relation")
            if relation and relation != "listed architecture layer order":
                facts.append(f"{edge['from']} {relation} {edge['to']}.")

    if not facts:
        return ""

    return f"### {title}\n\n" + "\n\n".join(facts)


# ============================================================
# ANSWER AGENT
# ============================================================


class AnswerPlanner:
    """Plans answer structure without generating factual content."""

    name = "Answer Planning Agent"

    def run(self, question: str, answer_type: str = "short", results=None) -> Dict[str, Any]:
        q = _clean_space(question).lower()
        intent = _question_intent(question)
        topic = _question_topic(question)
        plan = []

        if intent == "definition":
            plan = ["concise_definition_first", "relevant_key_points"]
            if re.search(r"\b(?:example|examples|illustrate)\b", q):
                plan.append("course_supported_example")
        elif intent == "comparison":
            plan = ["direct_introduction", "comparison_table_when_appropriate", "key_differences"]
        elif intent == "algorithm":
            plan = ["purpose_or_concept", "ordered_steps", "pseudocode_or_flow_when_supported"]
        elif intent == "architecture":
            plan = ["components", "roles", "relationships", "diagram_only_when_supported"]
        elif intent == "list":
            plan = ["direct_numbered_or_bulleted_list", "brief_explanations_when_supported"]
        elif intent == "how":
            plan = ["brief_introduction", "steps_in_correct_sequence", "working_or_flow_when_supported"]
        elif intent == "explanation":
            plan = ["introduction", "structured_explanation", "relevant_details"]
            if answer_type == "detailed":
                plan.append("conclusion")
        elif intent == "why":
            plan = ["direct_reason", "explanation"]
        elif intent == "advantages":
            asks_advantages = bool(re.search(r"\b(?:advantages?|benefits?|merits|pros)\b", q))
            asks_disadvantages = bool(re.search(r"\b(?:disadvantages?|limitations?|demerits|cons)\b", q))
            requested_points = []
            if asks_advantages or not asks_disadvantages:
                requested_points.append("advantages")
            if asks_disadvantages:
                requested_points.append("disadvantages_or_limitations")
            plan = ["brief_introduction", *requested_points, "clear_numbered_or_bulleted_points"]
        elif intent == "example":
            plan = [
                "concept",
                "course_supported_example",
                "example_input_from_evidence",
                "example_output_exactly_from_evidence_or_unavailable",
                "explain_the_example_from_evidence",
            ]
        else:
            plan = ["direct_answer", "structured_relevant_explanation"]

        if answer_type == "short":
            depth_instruction = "concise_exam_oriented_response"
        elif answer_type == "detailed":
            depth_instruction = "complete_structured_exam_answer"
        else:
            depth_instruction = "moderately_detailed_response"

        if re.search(r"\b(?:diagram|flowchart|block diagram|draw)\b", q):
            plan.append("grounded_diagram")

        # Remove duplicates while preserving order.
        plan = list(dict.fromkeys(plan))

        return {
            "intent": intent,
            "topic": topic,
            "answer_type": answer_type,
            "response_depth": depth_instruction,
            "plan": plan,
            "evidence_priority": ["primary", "supporting_context"],
        }


# ============================================================
# ADAPTIVE LEARNING AGENT
# ============================================================

class AdaptiveLearningAgent:
    """
    Chooses how the tutor should teach the current concept.

    This agent does not create factual content. It only determines
    teaching mode from the student's wording, recent conversation,
    answer intent, and requested depth.
    """

    name = "Adaptive Learning Agent"

    def run(
        self,
        question: str,
        history: str = "",
        answer_type: str = "short",
        context_aware: bool = False,
        answer_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        q = _clean_space(question).lower()
        history = _extract_history_text(history)

        intent = _question_intent(question)
        follow_up = _is_follow_up(question)

        if re.search(r"\b(?:explain\s+again|again|simpler|simple|easy|easier|in simple words|eli5)\b", q):
            teaching_mode = "simplify"
            explanation_level = "simple"
        elif re.search(r"\b(?:in detail|detailed|deeply|deep|elaborate|more detail)\b", q):
            teaching_mode = "deep_explanation"
            explanation_level = "detailed"
        elif re.search(r"\b(?:example|examples|real[- ]world|illustrate)\b", q):
            teaching_mode = "example_focused"
            explanation_level = "normal"
        elif re.search(r"\b(?:diagram|flowchart|architecture|block diagram|draw)\b", q):
            teaching_mode = "visual_explanation"
            explanation_level = "normal"
        elif re.search(r"\b(?:5|10|13|15|16)\s*marks?|\b(?:exam|university|long answer)\b", q):
            teaching_mode = "exam_focused"
            explanation_level = "detailed"
        elif intent == "why":
            teaching_mode = "reason_focused"
            explanation_level = "normal"
        elif intent == "how":
            teaching_mode = "process_focused"
            explanation_level = "normal"
        elif intent == "algorithm":
            teaching_mode = "algorithm_focused"
            explanation_level = "normal"
        elif intent == "architecture":
            teaching_mode = "architecture_focused"
            explanation_level = "normal"
        elif intent == "list":
            teaching_mode = "list_focused"
            explanation_level = "normal"
        elif intent == "advantages":
            teaching_mode = "points_focused"
            explanation_level = "normal"
        elif intent == "comparison":
            teaching_mode = "comparison_focused"
            explanation_level = "normal"
        elif follow_up and history:
            teaching_mode = "continue_concept"
            explanation_level = "normal"
        elif answer_type == "detailed":
            teaching_mode = "deep_explanation"
            explanation_level = "detailed"
        elif answer_type == "medium":
            teaching_mode = "guided_explanation"
            explanation_level = "normal"
        else:
            teaching_mode = "direct_teaching"
            explanation_level = "normal"

        if follow_up and history:
            context_instruction = (
                "Resolve references such as this, that, it, again, next, or the previous step "
                "from the recent conversation before answering. Continue the current concept "
                "unless the student clearly introduces a new concept."
            )
        else:
            context_instruction = (
                "Treat this as a new request unless the available conversation clearly indicates "
                "that it is a continuation."
            )

        plan_steps = []
        if answer_plan:
            plan_steps = list(answer_plan.get("plan", []))

        return {
            "teaching_mode": teaching_mode,
            "explanation_level": explanation_level,
            "intent": intent,
            "follow_up": bool(follow_up and history),
            "context_instruction": context_instruction,
            "plan": plan_steps,
        }

    def instruction(self, profile: Dict[str, Any]) -> str:
        mode = profile.get("teaching_mode", "direct_teaching")
        level = profile.get("explanation_level", "normal")
        context_instruction = profile.get("context_instruction", "")

        mode_rules = {
            "simplify": "Explain the same concept more simply. Prefer short sentences and intuitive wording. Do not remove important course facts.",
            "deep_explanation": "Give a deeper explanation using the retrieved course evidence. Explain relationships and working clearly without adding outside facts.",
            "example_focused": (
                "Use the retrieved evidence to identify the example input and output. "
                "Copy any explicit source output exactly; never calculate or infer a "
                "missing output. If no explicit output is present, state that it is "
                "not available in the extracted course material. Explain only what "
                "the evidence supports."
            ),
            "visual_explanation": "Use the grounded diagram supplied by the tutor when available. Do not invent components or relationships. If no grounded diagram is available, explain the concept in text instead of inventing a visual structure.",
            "exam_focused": "Write an academically structured answer suitable for the requested marks. Cover the planned sections using only supported evidence.",
            "reason_focused": "Answer the reason directly first, then explain why using the retrieved evidence.",
            "process_focused": "Explain the process or working in a clear sequence. Do not invent steps that are absent from the evidence.",
            "algorithm_focused": "Explain the algorithm's purpose, then its supported steps in order. Include pseudocode or flow only when the course evidence supports it.",
            "architecture_focused": "Describe supported components, their roles, and relationships. Include a diagram only when the evidence supports it.",
            "list_focused": "Answer with a direct numbered or bulleted list. Explain items only to the extent supported by the course evidence.",
            "points_focused": "Present only the requested advantages or disadvantages as clear numbered or bulleted points, using course evidence.",
            "comparison_focused": "Compare the requested concepts using only evidence-supported attributes. A Markdown table is appropriate when useful.",
            "continue_concept": "Continue teaching the current concept from the conversation. Address exactly the student's follow-up instead of restarting unnecessarily.",
            "guided_explanation": "Give a clear, moderately detailed explanation with the most useful key points.",
            "direct_teaching": "Answer directly and clearly, then add only the explanation needed by the question.",
        }

        return (
            "ADAPTIVE TEACHING INSTRUCTIONS: "
            + mode_rules.get(mode, mode_rules["direct_teaching"])
            + f" Explanation level: {level}. "
            + context_instruction
            + " The retrieved course evidence remains the sole factual authority. "
              "Primary evidence has priority over supporting context. "
              "Never turn the adaptive instruction into factual content. "
        )


class AnswerAgent:

    name = "Answer Agent"


    def run(
        self,
        question: str,
        results: List[Dict[str, Any]],
        history: str = "",
        answer_type: str = "short",
        answer_plan: Optional[Dict[str, Any]] = None,
    ) -> str:

        if not results:

            return (
                "I could not find enough relevant information "
                "in the uploaded course material to answer "
                "this question accurately."
            )


        intent = _question_intent(
            question
        )


        # ----------------------------------------------------
        # Direct definition first
        # ----------------------------------------------------

        if intent == "definition":
            answer_results = (
                _definition_focused_results(
                    question,
                    results,
                    limit=6,
                )
            )

        else:
            # Use direct evidence first, then a small amount of supporting
            # context. This prevents context expansion from flooding the LLM.
            answer_results = _ordered_evidence(
                results,
                max_primary=5 if answer_type == "short" else 6,
                max_supporting=1 if answer_type == "short" else 3,
            )


        # ----------------------------------------------------
        # Generate using the correct grounded generator.
        # short    -> generator_short.py
        # medium   -> generator_long.py
        # detailed -> generator_long.py
        # ----------------------------------------------------

        try:
            plan_text = ""
            if answer_plan:
                plan_text = (
                    "\n\nANSWER STRUCTURE (follow only when supported by the evidence): "
                    + " -> ".join(answer_plan.get("plan", []))
                    + f"\nResponse depth: {answer_plan.get('response_depth', 'moderately_detailed_response')}. "
                    + "\nEvidence priority: primary evidence first; supporting context only when relevant. "
                    "Do not introduce facts that are absent from the retrieved evidence. "
                    "Synthesize a tutor response; do not return a retrieved sentence verbatim by itself."
                )

            planned_history = f"{history}{plan_text}" if plan_text else history
            answer_generator = (
                generate_short_answer
                if answer_type == "short"
                else generate_long_answer
            )
            answer = answer_generator(
                question,
                answer_results,
                history=planned_history,
            )

        except Exception:
            return (
                "I could not generate an answer from "
                "the uploaded course material."
            )


        answer = str(
            answer or ""
        ).strip()

        def is_raw_evidence_sentence(candidate: str) -> bool:
            candidate_key = re.sub(r"\W+", "", _clean_space(candidate).lower())
            return bool(candidate_key) and any(
                candidate_key
                == re.sub(r"\W+", "", _clean_space(sentence).lower())
                for result in answer_results
                for sentence in _split_sentences(_text(result))
            )

        if answer and is_raw_evidence_sentence(answer):
            revision_history = (
                f"{planned_history}\n\n"
                "REVISION REQUIRED: Your previous output copied one course sentence verbatim. "
                "Rewrite it as a concise, clear tutor explanation using only the same evidence. "
                "Add only relevant supported points, and do not quote or copy the sentence."
            )
            try:
                answer = str(
                    answer_generator(
                        question,
                        answer_results,
                        history=revision_history,
                    )
                    or ""
                ).strip()
            except Exception:
                answer = ""

            if answer and is_raw_evidence_sentence(answer):
                return (
                    "I could not generate a sufficiently synthesized answer "
                    "from the uploaded course material."
                )


        if not answer:

            return (
                "I could not generate a reliable answer "
                "from the uploaded course material."
            )


        # Remove accidental code fences.
        answer = re.sub(
            r"```(?:markdown|text)?",
            "",
            answer,
            flags=re.IGNORECASE,
        )


        answer = answer.replace(
            "```",
            ""
        ).strip()

        return answer


def _example_io_from_results(
    results: List[Dict[str, Any]],
    question: str = "",
) -> Tuple[str, str]:
    """Extract only source-explicit example input and printed output."""
    query_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z]{4,}", question)
        if word.lower() not in {
            "with", "from", "that", "this", "what", "when", "where",
            "which", "about", "explain", "example", "examples",
        }
    }
    candidates = []

    def resolve_literal_input(input_lines: List[str]) -> str:
        if not input_lines:
            return ""

        assignment = re.match(
            r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$",
            input_lines[-1],
        )
        if not assignment:
            return "\n".join(input_lines).strip()

        rhs = assignment.group(2)
        referenced_names = re.findall(r"\b[A-Za-z_]\w*\b", rhs)
        attribute_names = set(re.findall(r"\.([A-Za-z_]\w*)", rhs))
        for name in referenced_names:
            if name in attribute_names:
                continue
            for result in results:
                for line in _text(result).splitlines():
                    source_assignment = re.match(
                        rf"^\s*{re.escape(name)}\s*=\s*(.+)$",
                        line,
                    )
                    if source_assignment and re.match(
                        r"""(?:['"`].*['"`]|[-+]?\d|True\b|False\b|None\b|[\[{])""",
                        source_assignment.group(1).strip(),
                    ):
                        return line.strip()

        return "\n".join(input_lines).strip()

    labeled_input = ""
    for result in results:
        text = _text(result)
        lines = text.splitlines()

        for index, line in enumerate(lines):
            output_label = re.match(r"^\s*output\s*:\s*(.*)$", line, re.IGNORECASE)
            input_label = re.match(r"^\s*input\s*:\s*(.*)$", line, re.IGNORECASE)
            if output_label:
                output = output_label.group(1).strip()
                if not output and index + 1 < len(lines):
                    output = lines[index + 1].strip()
                if output:
                    input_text = ""
                    for prior in range(index - 1, -1, -1):
                        match = re.match(r"^\s*input\s*:\s*(.*)$", lines[prior], re.IGNORECASE)
                        if match:
                            input_text = match.group(1).strip()
                            if not input_text and prior + 1 < index:
                                input_text = lines[prior + 1].strip()
                            break
                    return input_text or labeled_input, output

            if input_label:
                input_text = input_label.group(1).strip()
                if not input_text and index + 1 < len(lines):
                    input_text = lines[index + 1].strip()
                labeled_input = input_text
                for following in range(index + 1, len(lines)):
                    output_match = re.match(
                        r"^\s*output\s*:\s*(.*)$",
                        lines[following],
                        re.IGNORECASE,
                    )
                    if output_match:
                        output = output_match.group(1).strip()
                        if not output and following + 1 < len(lines):
                            output = lines[following + 1].strip()
                        if output:
                            return input_text, output
                    if re.match(r"^\s*input\s*:", lines[following], re.IGNORECASE):
                        break

        print_lines = [
            index for index, line in enumerate(lines)
            if re.search(r"\bprint\s*\(", line)
        ]
        if not print_lines:
            continue

        input_lines = []
        for index in range(print_lines[0] - 1, -1, -1):
            candidate = lines[index].strip()
            if not candidate:
                if input_lines:
                    break
                continue
            if re.match(
                r"^(?:import|from|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*=|"
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\()",
                candidate,
            ):
                input_lines.insert(0, lines[index].rstrip())
                continue
            if input_lines:
                break

        output_lines = []
        for index in print_lines:
            following = index + 1
            while following < len(lines) and not lines[following].strip():
                following += 1
            if following >= len(lines):
                continue

            candidate = lines[following].rstrip()
            if not re.match(
                r"^(?:[\[{'\"(]|[-+]?\d|True\b|False\b|None\b|"
                r"tensor\s*\(|torch\.Size\s*\(|Traceback\b|"
                r"(?:Value|Type|Index|Key|Name|"
                r"Attribute|Runtime)Error\b)",
                candidate.strip(),
                re.IGNORECASE,
            ):
                continue

            output_lines.append(candidate)
            opening = sum(candidate.count(char) for char in "[{(")
            closing = sum(candidate.count(char) for char in "]})")
            cursor = following + 1
            while opening > closing and cursor < len(lines):
                continuation = lines[cursor].rstrip()
                output_lines.append(continuation)
                opening += sum(continuation.count(char) for char in "[{(")
                closing += sum(continuation.count(char) for char in "]})")
                cursor += 1

        if output_lines:
            context_start = max(0, print_lines[0] - 6)
            context_end = min(
                len(lines),
                print_lines[-1] + len(output_lines) + 2,
            )
            context = " ".join(lines[context_start:context_end]).lower()
            score = sum(
                2 if term in context else 1
                for term in query_terms
                if term in context
                or (len(term) >= 6 and term[:6] in context)
            )
            if query_terms and score < 2:
                continue
            candidates.append((
                score,
                "\n".join(input_lines).strip(),
                "\n".join(output_lines).strip(),
            ))

    if not candidates:
        return labeled_input, ""

    _, input_text, output_text = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    input_text = resolve_literal_input(input_text.splitlines())
    return input_text, output_text


def _enforce_example_io(
    answer: str,
    results: List[Dict[str, Any]],
    question: str = "",
) -> str:
    input_text, output_text = _example_io_from_results(results, question)
    if input_text:
        input_content = f"```text\n{input_text}\n```"
    else:
        input_content = (
            "The course material provides the example, but the exact input "
            "is not available in the extracted text."
        )

    if output_text:
        output_content = f"```text\n{output_text}\n```"
    else:
        output_content = (
            "The course material provides the example, but the exact "
            "output is not available in the extracted text."
        )

    if not re.search(r"(?im)^##\s+Course-Supported Example\s*$", answer):
        answer = answer.rstrip() + "\n\n## Course-Supported Example\n"

    example_headings = list(re.finditer(
        r"(?im)^##\s+Course-Supported Example\s*$",
        answer,
    ))
    for duplicate in reversed(example_headings[1:]):
        following_section = re.search(
            r"(?m)^##\s+",
            answer[duplicate.end():],
        )
        duplicate_end = (
            duplicate.end() + following_section.start()
            if following_section
            else len(answer)
        )
        answer = answer[:duplicate.start()] + answer[duplicate_end:]

    section = re.search(
        r"(?im)^##\s+Course-Supported Example\s*$",
        answer,
    )
    if not section:
        return answer.strip()
    next_section = re.search(r"(?m)^##\s+", answer[section.end():])
    section_end = (
        section.end() + next_section.start()
        if next_section
        else len(answer)
    )
    section_body = answer[section.end():section_end]
    explanation_match = re.search(
        r"(?im)^#{1,6}\s+Explanation of the Example\s*$",
        section_body,
    )
    explanation = ""
    if explanation_match:
        explanation = section_body[explanation_match.end():].strip()
        following_heading = re.search(r"(?m)^#{1,6}\s+", explanation)
        if following_heading:
            explanation = explanation[:following_heading.start()].strip()
    if not explanation:
        explanation = (
            "The retrieved course material does not provide additional "
            "explanation for this example."
        )

    example_section = (
        "\n\n## Course-Supported Example\n\n"
        f"### Input\n\n{input_content}\n\n"
        f"### Output\n\n{output_content}\n\n"
        f"### Explanation of the Example\n\n{explanation}\n"
    )
    return (
        answer[:section.start()]
        + example_section
        + answer[section_end:]
    ).strip()


# ============================================================
# ANSWER GROUNDING AGENT
# ============================================================

class AnswerGroundingAgent:

    name = "Answer Grounding Agent"

    def __init__(
        self,
        minimum_sentence_score: float = 0.35,
        minimum_coverage: float = 0.50,
        primary_threshold: float = 0.38,
        supporting_threshold: float = 0.48,
    ):
        self.minimum_sentence_score = minimum_sentence_score
        self.minimum_coverage = minimum_coverage
        self.primary_threshold = primary_threshold
        self.supporting_threshold = supporting_threshold

    @staticmethod
    def _claim_score(claim: str, evidence: str) -> float:
        claim_words = _word_set(claim)
        evidence_words = _word_set(evidence)

        if not claim_words:
            return 1.0

        overlap = len(claim_words & evidence_words) / max(1, len(claim_words))

        # Prefer exact multi-word phrases when they occur in the evidence.
        claim_clean = _clean_space(claim).lower()
        evidence_clean = _clean_space(evidence).lower()
        phrase_bonus = 0.12 if len(claim_clean.split()) >= 3 and claim_clean in evidence_clean else 0.0

        return min(1.0, overlap + phrase_bonus)

    @staticmethod
    def _evidence_buckets(
        results: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        primary = []
        supporting = []

        for result in results:
            text = _text(result)
            if not text:
                continue

            evidence_type = str(
                result.get("evidence_type", "primary")
            ).lower()

            if evidence_type == "supporting_context":
                supporting.append(text)
            else:
                primary.append(text)

        return " ".join(primary), " ".join(supporting)

    @staticmethod
    def _claim_sentences(answer: str) -> List[str]:
        sentences = _split_sentences(answer)

        # Headings and very short fragments are not factual claims by
        # themselves and should not cause a grounding failure.
        claims = []
        for sentence in sentences:
            cleaned = _clean_space(sentence).strip(" -*#:")
            if not cleaned:
                continue

            words = _word_set(cleaned)
            if len(words) < 3:
                continue

            # Markdown table separator rows are formatting, not claims.
            if re.fullmatch(r"[\s|:_-]+", cleaned):
                continue

            claims.append(cleaned)

        return claims

    def run(
        self,
        answer: str,
        results: List[Dict[str, Any]],
        question: str = "",
    ) -> Dict[str, Any]:

        answer = str(answer or "").strip()

        if not answer:
            return {
                "grounded": False,
                "coverage": 0.0,
                "checked_sentences": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "claims": [],
                "needs_revision": True,
            }

        primary_text, supporting_text = self._evidence_buckets(results)
        all_evidence = f"{primary_text} {supporting_text}".strip()

        claims = self._claim_sentences(answer)

        if not claims:
            return {
                "grounded": False,
                "coverage": 0.0,
                "checked_sentences": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "claims": [],
                "needs_revision": True,
            }

        claim_records = []
        supported = 0

        for claim in claims:
            primary_score = self._claim_score(
                claim,
                primary_text,
            ) if primary_text else 0.0

            supporting_score = self._claim_score(
                claim,
                supporting_text,
            ) if supporting_text else 0.0

            overall_score = self._claim_score(
                claim,
                all_evidence,
            )

            if primary_score >= self.primary_threshold:
                status = "supported_primary"
                supported += 1
            elif supporting_score >= self.supporting_threshold:
                status = "supported_context"
                supported += 1
            elif overall_score >= self.minimum_sentence_score:
                # Evidence exists, but is not strong enough to call it
                # directly supported. Mark for revision rather than
                # accepting a weak lexical match.
                status = "weak_support"
            else:
                status = "unsupported"

            claim_records.append({
                "claim": claim,
                "primary_score": round(primary_score, 4),
                "supporting_score": round(supporting_score, 4),
                "overall_score": round(overall_score, 4),
                "status": status,
            })

        unsupported = sum(
            1
            for item in claim_records
            if item["status"] in {"unsupported", "weak_support"}
        )

        coverage = supported / max(1, len(claim_records))

        # Definition concept check.
        concept_present = True
        if _question_intent(question) == "definition":
            topic = _question_topic(question)
            concept_present = bool(
                topic
                and re.search(
                    rf"\b{re.escape(topic.lower())}\b",
                    answer.lower(),
                )
            )

        # A good answer must have high claim coverage. A single unsupported
        # minor sentence is tolerated, but a large fraction of unsupported
        # claims triggers revision.
        grounded = bool(
            coverage >= self.minimum_coverage
            and concept_present
            and unsupported <= max(1, int(len(claim_records) * 0.25))
        )

        return {
            "grounded": grounded,
            "coverage": round(coverage, 4),
            "checked_sentences": len(claims),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "claims": claim_records,
            "needs_revision": not grounded,
        }


# ============================================================
# AI TUTOR ORCHESTRATOR
# ============================================================

class AITutor:

    name = "AI Tutor"


    def __init__(self):

        self.query_agent = (
            QueryUnderstandingAgent()
        )


        self.course_agent = (
            CourseDetectionAgent()
        )


        self.retrieval_agent = (
            RetrievalAgent()
        )


        self.reranking_agent = (
            RerankingAgent()
        )


        self.validation_agent = (
            ValidationAgent(
                minimum_semantic_score=0.45
            )
        )


        self.answer_planner = (
            AnswerPlanner()
        )

        self.adaptive_learning_agent = (
            AdaptiveLearningAgent()
        )

        self.grounded_diagram_agent = (
            GroundedDiagramAgent()
        )

        self.answer_agent = (
            AnswerAgent()
        )


        self.answer_grounding_agent = (
            AnswerGroundingAgent()
        )


    # ========================================================
    # EMPTY RESPONSE
    # ========================================================

    def _empty_response(
        self,
        message: str
    ) -> Dict[str, Any]:

        return {

            "answer": message,

            "sources": [],

            "search_query": "",

            "answer_type": "short",

            "validation": {
                "sufficient": False,
                "reason": message,
                "best_score": 0.0,
                "evidence_count": 0,
            },

            "answer_grounding": {
                "grounded": False,
                "coverage": 0.0,
                "checked_sentences": 0,
            },

            "attempts": 0,

            "context_aware": False,

            "diagram": {
                "requested": False,
                "grounded": False,
                "type": None,
                "mermaid": "",
                "nodes": [],
                "edges": [],
            },

        }


    # ========================================================
    # SOURCE RECORDS
    # ========================================================

    def _source_records(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        sources = []

        seen = set()


        for result in results:

            filename = result.get(
                "filename"
            )


            page = result.get(
                "page"
            )


            material_id = result.get(
                "material_id"
            )


            key = (
                str(filename or ""),
                str(page or ""),
                str(material_id or ""),
            )


            if key in seen:

                continue


            seen.add(
                key
            )


            sources.append({

                "page": page,

                "filename": filename,

                "material_id": material_id,

                "course_id": result.get(
                    "course_id"
                ),

                "section": result.get(
                    "section"
                ),

                "topic": (
                    result.get("topic")
                    or result.get("topics")
                ),

                "image_path": result.get(
                    "image_path"
                ),

                # Preserve the retrieval role so the UI/debug layer can
                # distinguish direct evidence from supporting context.
                "evidence_type": _evidence_type(result),
                "context_score": result.get("context_score"),
                "parent_result_rank": result.get("parent_result_rank"),
                "context_rank": result.get("context_rank"),

            })


        return sources


    # ========================================================
    # MAIN ASK
    # ========================================================

    def ask(
        self,
        question: str,
        course_id: Optional[str] = None,
        material_id: Optional[str] = None,
        history: str = "",
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # Original question
        # ----------------------------------------------------

        question = _clean_space(
            question
        )


        if not question:

            return self._empty_response(
                "Please enter a question."
            )


        # ----------------------------------------------------
        # Keep history bounded
        # ----------------------------------------------------

        history = _extract_history_text(
            history
        )


        # ----------------------------------------------------
        # Query understanding
        # ----------------------------------------------------

        query_info = (
            self.query_agent.run(
                question
            )
        )


        answer_type = query_info[
            "answer_type"
        ]


        initial_k = query_info[
            "retrieval_k"
        ]


        retry_k = query_info[
            "retry_k"
        ]


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The original student question remains untouched.
        #
        # The contextual query is ONLY used for retrieval.
        # ----------------------------------------------------

        contextual_query = (
            _build_contextual_query(
                question=question,
                history=history,
            )
        )


        context_aware = (
            contextual_query != question
        )


        search_query = contextual_query


        # ----------------------------------------------------
        # Course detection
        #
        # Use the contextual query only when it is actually
        # a follow-up. For normal questions use the original.
        # ----------------------------------------------------

        detected_course = None


        if course_id is None:

            course_detection_query = (
                contextual_query
                if context_aware
                else question
            )


            detected_course = (
                self.course_agent.run(
                    course_detection_query
                )
            )


            if detected_course:

                course_id = (
                    detected_course[
                        "course_id"
                    ]
                )


        # ----------------------------------------------------
        # FIRST RETRIEVAL
        # ----------------------------------------------------

        results = (
            self.retrieval_agent.run(
                query=search_query,
                course_id=course_id,
                material_id=material_id,
                top_k=initial_k,
            )
        )


        # ----------------------------------------------------
        # RERANK
        #
        # IMPORTANT:
        # Reranking uses ORIGINAL student question.
        # ----------------------------------------------------

        results = (
            self.reranking_agent.run(
                question=question,
                results=results,
                top_k=min(
                    initial_k,
                    8,
                ),
            )
        )

        # Keep primary evidence visible before supporting context.
        results = _ordered_evidence(
            results,
            max_primary=min(initial_k, 8),
            max_supporting=4,
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation = (
            self.validation_agent.run(
                question=question,
                results=results,
            )
        )


        attempts = 1


        # ----------------------------------------------------
        # DEFINITION RETRY
        # ----------------------------------------------------

        needs_definition_retry = (

            _question_intent(
                question
            ) == "definition"

            and bool(
                _question_topic(
                    question
                )
            )

            and not any(

                float(
                    result.get(
                        "_definition_evidence_score",
                        0.0,
                    )
                ) >= 0.45

                for result in results

            )

        )


        # ----------------------------------------------------
        # RETRY WHEN EVIDENCE IS WEAK
        # ----------------------------------------------------

        if (
            not validation["sufficient"]
            or needs_definition_retry
        ):

            improved_query = (
                self.query_agent.improve_query(
                    question
                )
            )


            # For a contextual follow-up, preserve the
            # conversation context during retry.
            if context_aware:

                improved_query = (
                    f"{improved_query}. "
                    f"Previous conversation concepts: "
                    f"{', '.join(_extract_history_concepts(history))}"
                )


            retry_results = (
                self.retrieval_agent.run(
                    query=improved_query,
                    course_id=course_id,
                    material_id=material_id,
                    top_k=retry_k,
                )
            )


            retry_results = (
                self.reranking_agent.run(
                    question=question,
                    results=retry_results,
                    top_k=min(
                        retry_k,
                        10,
                    ),
                )
            )

            retry_results = _ordered_evidence(
                retry_results,
                max_primary=min(retry_k, 10),
                max_supporting=5,
            )


            retry_validation = (
                self.validation_agent.run(
                    question=question,
                    results=retry_results,
                )
            )


            attempts = 2


            current_best = max(
                (
                    _semantic_score(
                        result
                    )
                    for result in results
                ),
                default=0.0,
            )


            retry_best = max(
                (
                    _semantic_score(
                        result
                    )
                    for result in retry_results
                ),
                default=0.0,
            )


            current_definition = max(
                (
                    float(
                        result.get(
                            "_definition_evidence_score",
                            0.0,
                        )
                    )
                    for result in results
                ),
                default=0.0,
            )


            retry_definition = max(
                (
                    float(
                        result.get(
                            "_definition_evidence_score",
                            0.0,
                        )
                    )
                    for result in retry_results
                ),
                default=0.0,
            )


            if (
                retry_validation[
                    "sufficient"
                ]

                or retry_best
                > current_best

                or retry_definition
                > current_definition
            ):

                results = retry_results

                validation = retry_validation

                search_query = improved_query

        # Final evidence ordering before validation/answer generation.
        results = _ordered_evidence(
            results,
            max_primary=8,
            max_supporting=4,
        )


        # ----------------------------------------------------
        # NO SUFFICIENT EVIDENCE
        # ----------------------------------------------------

        if (
            not results
            or not validation["sufficient"]
        ):

            return {

                "answer": (
                    "I could not find enough relevant "
                    "information in the uploaded course "
                    "material to answer this question accurately."
                ),

                "sources": self._source_records(
                    results
                ),

                "search_query": search_query,

                "answer_type": answer_type,

                "validation": validation,

                "answer_grounding": {
                    "grounded": False,
                    "coverage": 0.0,
                    "checked_sentences": 0,
                },

                "attempts": attempts,

                "course_id": course_id,

                "course_detection": detected_course,

                "context_aware": context_aware,

            }


        # ----------------------------------------------------
        # ANSWER GENERATION
        #
        # ORIGINAL QUESTION IS USED HERE.
        #
        # This is very important.
        #
        # We do NOT send the contextual retrieval query as
        # the student's question.
        # ----------------------------------------------------

        answer_plan = self.answer_planner.run(
            question=question,
            answer_type=answer_type,
            results=results,
        )

        adaptive_profile = self.adaptive_learning_agent.run(
            question=question,
            history=history,
            answer_type=answer_type,
            context_aware=context_aware,
            answer_plan=answer_plan,
        )

        adaptive_instruction = self.adaptive_learning_agent.instruction(
            adaptive_profile
        )

        adaptive_history = (
            f"{history}\n\n{adaptive_instruction}"
            if history
            else adaptive_instruction
        )

        answer = (
            self.answer_agent.run(
                question=question,
                results=results,
                history=adaptive_history,
                answer_type=answer_type,
                answer_plan=answer_plan,
            )
        )


        # ----------------------------------------------------
        # CLAIM-LEVEL GROUNDING CHECK
        # ----------------------------------------------------

        grounding = (
            self.answer_grounding_agent.run(
                answer=answer,
                results=results,
                question=question,
            )
        )

        # ----------------------------------------------------
        # ONE STRICT REGENERATION PASS
        #
        # If the generated answer contains unsupported claims,
        # ask the same generator to revise using ONLY the retrieved
        # evidence. This is deliberately limited to one pass so the
        # tutor does not enter a generation loop.
        # ----------------------------------------------------

        if grounding.get("needs_revision"):
            revision_instruction = (
                "\n\nGROUNDING REVISION REQUIRED. "
                "Review your previous answer against the retrieved course evidence. "
                "Keep only claims supported by the evidence. "
                "Remove or rewrite unsupported claims. "
                "Do not add outside knowledge, assumptions, or invented facts. "
                "Preserve the requested answer structure where possible. "
                "Return only the corrected final answer."
            )

            revision_history = f"{adaptive_history}{revision_instruction}"

            try:
                revision_results = results[:6]

                if answer_type == "short":
                    revised_answer = generate_short_answer(
                        question,
                        revision_results,
                        history=revision_history,
                    )
                else:
                    revised_answer = generate_long_answer(
                        question,
                        revision_results,
                        history=revision_history,
                    )

                revised_answer = str(
                    revised_answer or ""
                ).strip()

                revised_answer = re.sub(
                    r"```(?:markdown|text)?",
                    "",
                    revised_answer,
                    flags=re.IGNORECASE,
                ).replace("```", "").strip()

                if revised_answer:
                    revised_grounding = (
                        self.answer_grounding_agent.run(
                            answer=revised_answer,
                            results=results,
                            question=question,
                        )
                    )

                    if (
                        revised_grounding.get("grounded")
                        or revised_grounding.get("coverage", 0.0)
                        > grounding.get("coverage", 0.0)
                    ):
                        answer = revised_answer
                        grounding = revised_grounding
                        grounding["revision_applied"] = True
                    else:
                        grounding["revision_applied"] = False

            except Exception:
                grounding["revision_applied"] = False

        # ----------------------------------------------------
        # GROUNDED DIAGRAM
        # ----------------------------------------------------

        diagram = self.grounded_diagram_agent.run(
            question=question,
            results=results,
        )

        if (
            not grounding.get("grounded")
            and diagram.get("grounded")
            and diagram.get("mermaid")
        ):
            diagram_answer = _diagram_grounded_summary(question, diagram)
            if diagram_answer:
                answer = diagram_answer
                grounding = self.answer_grounding_agent.run(
                    answer=answer,
                    results=results,
                    question=question,
                )
                grounding["diagram_backed_response"] = True

        if _question_intent(question) == "example":
            answer = _enforce_example_io(
                answer,
                results,
                question,
            )

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "answer": answer,

            "sources": self._source_records(
                results
            ),

            "search_query": search_query,

            "answer_type": answer_type,

            "answer_plan": answer_plan,

            "adaptive_learning": adaptive_profile,

            "diagram": diagram,

            "validation": validation,

            "answer_grounding": grounding,

            "verification": {
                "grounded": bool(grounding.get("grounded", False)),
                "coverage": grounding.get("coverage", 0.0),
                "supported_claims": grounding.get("supported_claims", 0),
                "unsupported_claims": grounding.get("unsupported_claims", 0),
                "revision_applied": grounding.get("revision_applied", False),
            },

            "attempts": attempts,

            "course_id": course_id,

            "course_detection": detected_course,

            "context_aware": context_aware,

        }


# ============================================================
# PUBLIC HELPERS
# ============================================================

_DEFAULT_TUTOR: Optional[AITutor] = None


def get_ai_tutor() -> AITutor:

    global _DEFAULT_TUTOR


    if _DEFAULT_TUTOR is None:

        _DEFAULT_TUTOR = AITutor()


    return _DEFAULT_TUTOR


def ask_ai_tutor(
    question: str,
    course_id: Optional[str] = None,
    material_id: Optional[str] = None,
    history: str = "",
) -> Dict[str, Any]:

    return get_ai_tutor().ask(

        question=question,

        course_id=course_id,

        material_id=material_id,

        history=history,

    )