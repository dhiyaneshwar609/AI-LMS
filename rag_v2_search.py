"""
RAG V2 Search Layer for AI-LMS

Loads the pre-built RAG V2 index from:
    rag_data_v2/index.pkl

Usage:
    from rag_v2_search import search
    results = search("Explain Cassandra architecture", course_id="Big_Data_Analytics")
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "rag_data_v2" / "index.pkl"

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_index_data = None
_index_mtime = None


def _load_index():
    global _index_data, _index_mtime

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"RAG V2 index not found: {INDEX_PATH}\n"
            "Run rag_v2_build_embeddings.py first."
        )

    current_mtime = INDEX_PATH.stat().st_mtime

    # Reload automatically when a teacher uploads new material and the
    # index is rebuilt. This keeps future uploads available without
    # restarting Streamlit.
    if _index_data is None or _index_mtime != current_mtime:
        with open(INDEX_PATH, "rb") as f:
            _index_data = pickle.load(f)
        _index_mtime = current_mtime

    return _index_data


def _load_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)

    return _model


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_text(item: Dict[str, Any]) -> str:
    for key in ("text", "chunk_text", "content"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _get_topic(item: Dict[str, Any]) -> str:
    for key in ("topic", "topic_name", "section", "title"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _get_course(item: Dict[str, Any]) -> str:
    for key in ("course_id", "course", "course_name"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _get_material(item: Dict[str, Any]) -> str:
    for key in ("material_id", "material", "material_name"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _get_page(item: Dict[str, Any]) -> Optional[int]:
    for key in ("page", "page_number", "page_num"):
        value = item.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def _get_image_path(item: Dict[str, Any]) -> Optional[str]:
    for key in ("image_path", "page_image", "figure_path"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _get_chunk_index(item: Dict[str, Any]) -> Optional[int]:
    """Return a stable chunk/position index when the builder stored one."""
    for key in ("chunk_id", "chunk_index", "index", "position", "chunk_no", "chunk_number"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _same_group(primary: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    """Check whether two chunks belong to the same material/topic context."""
    p_material = _get_material(primary).strip().lower()
    c_material = _get_material(candidate).strip().lower()
    p_course = _get_course(primary).strip().lower()
    c_course = _get_course(candidate).strip().lower()
    p_topic = _get_topic(primary).strip().lower()
    c_topic = _get_topic(candidate).strip().lower()

    if p_material and c_material and p_material != c_material:
        return False
    if p_course and c_course and p_course != c_course:
        return False

    # Topic matching is intentionally soft. Some builders store a section on
    # one chunk and leave it blank on adjacent chunks.
    if p_topic and c_topic:
        if p_topic == c_topic:
            return True
        p_words = set(re.findall(r"[a-z0-9]+", p_topic))
        c_words = set(re.findall(r"[a-z0-9]+", c_topic))
        if p_words and c_words and len(p_words & c_words) >= max(1, min(2, len(p_words))):
            return True
        return False

    return True


def _context_expansion_score(primary: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """Score how useful a candidate is as supporting context for a primary hit."""
    if not _same_group(primary, candidate):
        return -1.0

    score = 0.0
    p_page = _get_page(primary)
    c_page = _get_page(candidate)
    if p_page is not None and c_page is not None:
        distance = abs(p_page - c_page)
        if distance == 0:
            score += 0.45
        elif distance == 1:
            score += 0.30
        elif distance == 2:
            score += 0.15
        else:
            score -= min(0.15, distance * 0.01)

    p_topic = _get_topic(primary).strip().lower()
    c_topic = _get_topic(candidate).strip().lower()
    if p_topic and c_topic and p_topic == c_topic:
        score += 0.35
    elif p_topic and c_topic:
        p_words = set(re.findall(r"[a-z0-9]+", p_topic))
        c_words = set(re.findall(r"[a-z0-9]+", c_topic))
        overlap = len(p_words & c_words)
        if overlap:
            score += min(0.25, overlap * 0.08)

    return score


def _expand_with_related_context(
    selected: List[Dict[str, Any]],
    chunks: List[Any],
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    max_context_per_hit: int = 2,
) -> List[Dict[str, Any]]:
    """Add nearby same-section chunks without replacing the precise hits.

    This is a compatibility-safe form of parent/child retrieval because the
    current V2 index does not expose a formal parent_id/child_id hierarchy.
    Precise chunks remain the primary evidence; expanded chunks are marked as
    supporting context and receive a lower score.
    """
    if not selected:
        return selected

    existing = set()
    for item in selected:
        existing.add((
            _get_material(item).strip().lower(),
            _get_page(item),
            _get_chunk_index(item),
            _get_text(item).strip(),
        ))

    expanded: List[Dict[str, Any]] = []

    for primary_rank, primary in enumerate(selected, start=1):
        related = []
        p_topic = _get_topic(primary)
        p_material = _get_material(primary)
        p_course = _get_course(primary)

        for i, raw in enumerate(chunks):
            if not isinstance(raw, dict):
                continue
            text = _get_text(raw)
            if not text.strip():
                continue

            if p_material and _get_material(raw).strip().lower() != p_material.strip().lower():
                continue
            if p_course and _get_course(raw).strip().lower() != p_course.strip().lower():
                continue
            if not _same_group(primary, raw):
                continue

            key = (
                _get_material(raw).strip().lower(),
                _get_page(raw),
                _get_chunk_index(raw),
                text.strip(),
            )
            if key in existing:
                continue

            # Keep supporting context close to the retrieved evidence when
            # page metadata exists. If page metadata is absent, topic matching
            # is still useful.
            expansion_score = _context_expansion_score(primary, raw)
            if expansion_score < 0.0:
                continue

            semantic = 0.0
            if i < len(embeddings):
                try:
                    semantic = float(np.dot(query_vector, embeddings[i]))
                except Exception:
                    semantic = 0.0

            related.append((expansion_score + semantic * 0.15, expansion_score, semantic, i, raw))

        related.sort(key=lambda x: x[0], reverse=True)

        for local_rank, (combined, expansion_score, semantic, i, raw) in enumerate(
            related[:max_context_per_hit], start=1
        ):
            result = dict(raw)
            result["text"] = _get_text(raw)
            result["topic"] = _get_topic(raw)
            result["course_id"] = _get_course(raw)
            result["material_id"] = _get_material(raw)
            result["page"] = _get_page(raw)
            result["image_path"] = _get_image_path(raw)
            result["semantic_score"] = round(semantic, 6)
            result["context_score"] = round(expansion_score, 6)
            result["score"] = round(combined, 6)
            result["evidence_type"] = "supporting_context"
            result["parent_result_rank"] = primary_rank
            result["context_rank"] = local_rank
            expanded.append(result)

    # Primary evidence always comes first. Supporting context follows it.
    for item in selected:
        item.setdefault("evidence_type", "primary")

    return selected + expanded


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def _topic_intent(query: str) -> List[str]:
    """
    Returns topic phrases that deserve an exact-topic boost.
    This is intentionally generic and does not depend on page numbers.
    """
    q = _clean_query(query)
    intents = []

    patterns = [
        (r"\blamport\b.*\bmutual exclusion\b|\bmutual exclusion\b.*\blamport\b",
         ["lamport mutual exclusion", "lamport"]),
        (r"\blamport\b.*\blogical clock\b|\blogical clock\b.*\blamport\b",
         ["lamport logical clock", "logical clock"]),
        (r"\bglobal state\b", ["global state"]),
        (r"\bcassandra\b", ["cassandra"]),
        (r"\bhadoop\b", ["hadoop"]),
        (r"\bmapreduce\b|\bmap reduce\b", ["mapreduce", "map reduce"]),
        (r"\bhdfs\b", ["hdfs"]),
        (r"\bspark\b", ["spark"]),
        (r"\barchitecture\b", ["architecture"]),
    ]

    for pattern, words in patterns:
        if re.search(pattern, q):
            intents.extend(words)

    return list(dict.fromkeys(intents))


def _keyword_bonus(query: str, text: str, topic: str) -> float:
    q = _clean_query(query)
    combined = f"{topic} {text}".lower()

    bonus = 0.0

    # Exact topic intent gets the strongest bonus.
    for intent in _topic_intent(query):
        if intent in topic.lower():
            bonus += 0.18
        elif intent in combined:
            bonus += 0.07

    # Important query terms appearing in the chunk.
    terms = [
        t for t in re.findall(r"[a-z0-9]+", q)
        if len(t) >= 4
    ]

    if terms:
        hits = sum(1 for t in set(terms) if t in combined)
        bonus += min(0.12, hits * 0.025)

    # Penalise likely table-of-contents chunks.
    toc_markers = [
        "table of contents",
        "contents",
        "unit i",
        "unit ii",
        "unit iii",
        "unit iv",
        "unit v",
    ]
    marker_hits = sum(1 for marker in toc_markers if marker in text.lower())
    if marker_hits >= 3:
        bonus -= 0.12

    return bonus


def _definition_subject(query: str) -> Optional[str]:
    """Extract the concept named by a definition-style question."""
    patterns = (
        r"^\s*what\s+(?:is|are)\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*define\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*definition\s+of\s+(.+?)\s*[?.!]*\s*$",
        r"^\s*what\s+does\s+(.+?)\s+mean\s*[?.!]*\s*$",
    )
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE)
            if re.search(r"[^\W_]", subject, flags=re.UNICODE):
                return subject
    return None


def _normalise_definition_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"(?<=[a-z0-9])[-\u00ad]\s*(?=[a-z0-9])", "", value)
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _definition_heading(item: Dict[str, Any]) -> str:
    """Collect available heading metadata, including builders' plural topics field."""
    headings = []
    for key in ("topic", "topic_name", "section", "title", "topics"):
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            headings.extend(str(part) for part in value if part)
        elif value:
            headings.append(str(value))
    return " ".join(headings)


def _definition_intent_bonus(
    subject: str,
    item: Dict[str, Any],
    text: str,
    page: Optional[int],
    first_page: Optional[int],
) -> float:
    """Rerank definition queries toward explicit, introductory concept evidence."""
    phrase = _normalise_definition_text(subject)
    if not phrase:
        return 0.0

    heading = _normalise_definition_text(_definition_heading(item))
    body = text.lower()
    subject_words = re.findall(r"\w+", phrase, flags=re.UNICODE)
    phrase_pattern = r"[\s\-\u00ad]+".join(re.escape(word) for word in subject_words)
    phrase_pattern = rf"\b{phrase_pattern}\b"
    in_body = re.search(phrase_pattern, body) is not None

    # Cues must describe the concept in the chunk body itself. Keep the phrase
    # adjacent to its predicate so punctuation cannot join unrelated clauses
    # (for example, "stage of NLP, is the process" defining tokenization).
    definition_patterns = (
        rf"{phrase_pattern}\s+(?:can be\s+)?defined as\b",
        rf"{phrase_pattern}\s+refers to\b",
        rf"{phrase_pattern}\s+is (?:a|an|the)\b",
        rf"{phrase_pattern}\s+means\b",
        rf"\bdefinition of\s+{phrase_pattern}\b",
    )
    has_definition_cue = any(re.search(pattern, body) for pattern in definition_patterns)

    explanatory_patterns = (
        rf"\b(?:an?|the)\s+(?:(?:early|first|basic|modern|common|typical|important)\s+)"
        rf"{phrase_pattern}\s+(?:system|field|discipline|area|process|task|technology|"
        rf"method|approach|technique|study)\b",
        rf"\bthis (?:chapter|section)\s+(?:introduces|describes|explains|defines)\s+"
        rf"{phrase_pattern}\b",
        rf"\bin this (?:chapter|section)\s+we\s+(?:introduce|describe|explain|define)\s+"
        rf"{phrase_pattern}\b",
    )
    has_explanatory_cue = any(re.search(pattern, body) for pattern in explanatory_patterns)
    section = _normalise_definition_text(str(item.get("section") or ""))
    has_intro_section_evidence = (
        in_body
        and bool(re.search(r"\bintroduction\b", section))
        and len(re.findall(r"[.!?]", text)) >= 2
    )
    has_real_definition = has_definition_cue or has_explanatory_cue or has_intro_section_evidence

    # Front matter often repeats the concept in a book title. Treat it as
    # metadata, not evidence, unless the chunk also contains a real definition.
    metadata_markers = (
        "copyright",
        "all rights reserved",
        "isbn",
        "published by",
        "printed in",
        "draft of",
        "edition",
        "university",
        "table of contents",
    )
    metadata_marker_count = sum(1 for marker in metadata_markers if marker in body[:1200])
    section_is_front_matter = bool(
        re.search(r"\b(?:cover|title page|copyright|contents|front matter)\b", section)
    )
    is_front_matter = section_is_front_matter or metadata_marker_count >= 2

    # Only definition prose receives a meaningful positive rerank. A heading
    # match may strengthen that prose, but a title or other metadata alone
    # cannot earn the bonus.
    bonus = 0.0
    if in_body and has_real_definition and not is_front_matter:
        bonus += 0.22 if has_definition_cue else 0.18
        if phrase in heading:
            bonus += 0.04
        if page is not None and first_page is not None:
            distance_from_start = page - first_page
            if 0 <= distance_from_start <= 6:
                bonus += 0.03
    elif in_body and not is_front_matter:
        # An isolated mention is weak evidence and should not beat an
        # explanatory definition simply by repeating the queried phrase.
        bonus -= 0.12

    structural_text = f"{heading} {_normalise_definition_text(text[:500])}"
    if re.search(r"\b(?:bibliography|references|works cited|literature cited)\b", structural_text):
        bonus -= 0.35
    if re.search(r"\btable of contents\b", structural_text) or section == "contents":
        bonus -= 0.35
    if re.search(r"\b(?:historical notes?|history of|history)\b", f"{heading} {structural_text}"):
        bonus -= 0.12
    if re.search(r"\bcontinued(?:\s*\.{1,3})?\b", f"{heading} {text[:180].lower()} {text[-120:].lower()}"):
        bonus -= 0.12
    if is_front_matter and not has_real_definition:
        bonus -= 0.50

    return max(-0.60, min(0.32, bonus))


def search(
    query: str,
    course_id: Optional[str] = None,
    material_id: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic + exact-topic retrieval.

    Args:
        query: Student's question.
        course_id: Optional exact course filter.
        material_id: Optional exact material filter.
        top_k: Number of results returned.

    Returns:
        List of dictionaries containing primary evidence plus optional
        supporting context from the same material/topic. Primary results are
        marked with evidence_type="primary"; expanded results use
        evidence_type="supporting_context".
    """
    if not query or not query.strip():
        return []

    data = _load_index()

    # The V2 builder stores chunks and embeddings. Support common key names.
    if isinstance(data, dict):
        chunks = (
            data.get("chunks")
            or data.get("documents")
            or data.get("items")
            or []
        )
        # Do not use Python's "or" with NumPy arrays.
        # NumPy arrays cannot be evaluated as a single True/False value.
        embeddings = data.get("embeddings")
        if embeddings is None:
            embeddings = data.get("vectors")
        if embeddings is None:
            embeddings = data.get("index")
    else:
        raise ValueError("Unexpected RAG V2 index format.")

    if embeddings is None:
        raise ValueError("RAG V2 index does not contain embeddings.")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    model = _load_model()
    qvec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    # Normalize index vectors defensively.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    candidates = []

    course_filter = _normalise(course_id).lower()
    material_filter = _normalise(material_id).lower()
    definition_subject = _definition_subject(query)

    first_page_by_group = {}
    if definition_subject:
        for item in chunks:
            if not isinstance(item, dict):
                continue
            item_course = _get_course(item)
            item_material = _get_material(item)
            if course_filter and item_course.lower() != course_filter:
                continue
            if material_filter and item_material.lower() != material_filter:
                continue
            page = _get_page(item)
            if page is None:
                continue
            group = (item_course, item_material)
            first_page_by_group[group] = min(page, first_page_by_group.get(group, page))

    for i, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue

        item_course = _get_course(item)
        item_material = _get_material(item)

        if course_filter and item_course.lower() != course_filter:
            continue

        if material_filter and item_material.lower() != material_filter:
            continue

        text = _get_text(item)
        if not text.strip():
            continue

        semantic_score = float(np.dot(qvec, embeddings[i]))
        topic = _get_topic(item)
        bonus = _keyword_bonus(query, text, topic)
        definition_bonus = 0.0
        if definition_subject:
            group = (item_course, item_material)
            definition_bonus = _definition_intent_bonus(
                definition_subject,
                item,
                text,
                _get_page(item),
                first_page_by_group.get(group),
            )
        final_score = semantic_score + bonus + definition_bonus

        candidates.append(
            (
                final_score,
                semantic_score,
                i,
                item,
            )
        )

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Avoid flooding the answer context with many chunks from one page.
    # Allow up to two chunks per page/material combination.
    selected = []
    page_counts = {}

    for final_score, semantic_score, i, item in candidates:
        page = _get_page(item)
        material = _get_material(item)
        page_key = (material, page)

        if page_key in page_counts and page_counts[page_key] >= 2:
            continue

        page_counts[page_key] = page_counts.get(page_key, 0) + 1

        result = dict(item)
        result["text"] = _get_text(item)
        result["topic"] = _get_topic(item)
        result["course_id"] = _get_course(item)
        result["material_id"] = _get_material(item)
        result["page"] = page
        result["image_path"] = _get_image_path(item)
        result["semantic_score"] = round(semantic_score, 6)
        result["score"] = round(final_score, 6)
        result["evidence_type"] = "primary"

        selected.append(result)

        if len(selected) >= max(1, int(top_k)):
            break

    # Step 2: expand each precise hit with nearby chunks from the same
    # material/topic. The original top-k evidence remains untouched and is
    # always returned first.
    selected = _expand_with_related_context(
        selected=selected,
        chunks=chunks,
        embeddings=embeddings,
        query_vector=qvec,
        max_context_per_hit=2,
    )

    return selected


def search_all(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience wrapper for searching across all V2 courses."""
    return search(query=query, course_id=None, material_id=None, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG V2 search.")
    parser.add_argument("query", help="Question to search")
    parser.add_argument("--course", default=None, help="Optional course_id")
    parser.add_argument("--material", default=None, help="Optional material_id")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    results = search(
        query=args.query,
        course_id=args.course,
        material_id=args.material,
        top_k=args.top_k,
    )

    print(f"\nQuery: {args.query}")
    print(f"Course: {args.course or 'ALL'}")
    print(f"Results: {len(results)}\n")

    for n, result in enumerate(results, 1):
        print("=" * 80)
        print(
            f"{n}. score={result['score']} "
            f"semantic={result['semantic_score']} "
            f"course={result.get('course_id')} "
            f"page={result.get('page')} "
            f"topic={result.get('topic')}"
        )
        print(result["text"][:700].replace("\n", " "))
