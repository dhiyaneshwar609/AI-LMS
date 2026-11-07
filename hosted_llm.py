"""
AI-LMS LLM client.

Supports two providers:
1. Groq hosted OpenAI-compatible API
2. Local Ollama

The public interfaces used by ai_tutor.py remain unchanged:
- generate_draft(...)
- verify_and_correct(...)
"""

from __future__ import annotations

import os
import logging
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# PROVIDER
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").strip().lower()


# ============================================================
# GROQ CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.groq.com/openai/v1",
).rstrip("/")

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")

GENERATOR_MODEL = os.getenv(
    "GENERATOR_MODEL",
    "openai/gpt-oss-20b",
)

VERIFIER_MODEL = os.getenv(
    "VERIFIER_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# LOCAL OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:4b",
)


# ============================================================
# TIMEOUTS
# ============================================================

GENERATOR_TIMEOUT = int(
    os.getenv("GENERATOR_TIMEOUT", "120")
)

VERIFIER_TIMEOUT = int(
    os.getenv("VERIFIER_TIMEOUT", "180")
)

OLLAMA_TIMEOUT = int(
    os.getenv("OLLAMA_TIMEOUT", "300")
)

MAX_COMPLETION_TOKENS = int(
    os.getenv("LLM_MAX_COMPLETION_TOKENS", "4096")
)


# ============================================================
# REASONING
# ============================================================

GENERATOR_REASONING = os.getenv(
    "GENERATOR_REASONING_EFFORT",
    "low",
)

VERIFIER_REASONING = os.getenv(
    "VERIFIER_REASONING_EFFORT",
    "medium",
)


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def _extract_content(data: dict) -> str:
    choices = data.get("choices") or []

    if not choices:
        return ""

    choice = choices[0]

    if choice.get("finish_reason") in {"length", "max_tokens"}:
        return ""

    message = choice.get("message") or {}

    return str(
        message.get("content") or ""
    ).strip()


def _extract_ollama_content(data: dict) -> str:
    message = data.get("message") or {}

    return str(
        message.get("content") or ""
    ).strip()


def _redact_llm_sensitive_text(text: str, sensitive_values=()) -> str:
    """Redact credentials and request text from diagnostic strings."""
    body = str(text or "")
    body = re.sub(
        r'(?i)("?(?:authorization|api[_-]?key|access[_-]?token|token|secret|password)"?\s*[:=]\s*"?)[^"&,}\s]+',
        r"\1[REDACTED]",
        body,
    )
    body = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", body)
    for sensitive_value in sensitive_values:
        if sensitive_value:
            body = body.replace(str(sensitive_value), "[REDACTED]")
    return body


def _safe_error_body(response: requests.Response, sensitive_values=()) -> str:
    """Return an API error body with credentials and request text redacted."""
    return _redact_llm_sensitive_text(
        getattr(response, "text", ""),
        sensitive_values=sensitive_values,
    )


# ============================================================
# GROQ / OPENAI-COMPATIBLE CHAT
# ============================================================

def _chat_groq(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    reasoning_effort: Optional[str] = None,
) -> str:

    started = time.perf_counter()
    logger.info(
        'phase=llm_request_started provider=groq model=%s',
        model,
    )

    if not API_KEY:
        logger.error(
            'phase=llm_request_failed provider=groq model=%s '
            'error_type=MissingAPIKey error="Missing LLM_API_KEY/GROQ_API_KEY" '
            'elapsed_s=%.3f',
            model,
            time.perf_counter() - started,
        )
        return ""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "temperature": 0.1,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        response_content_length = len(str(message.get("content") or ""))
        content = _extract_content(data)
        logger.info(
            'phase=llm_request_completed provider=groq model=%s status_code=%s '
            'elapsed_s=%.3f response_content_length=%d extracted_content_length=%d '
            'finish_reason=%s',
            model,
            response.status_code,
            time.perf_counter() - started,
            response_content_length,
            len(content),
            finish_reason,
        )
        return content

    except requests.exceptions.HTTPError as exc:
        error_response = exc.response
        logger.error(
            'phase=llm_request_http_error provider=groq model=%s status_code=%s '
            'response_body=%s elapsed_s=%.3f',
            model,
            getattr(error_response, "status_code", None),
            _safe_error_body(
                error_response,
                sensitive_values=(API_KEY, system_prompt, user_prompt),
            ) if error_response is not None else "",
            time.perf_counter() - started,
        )
        return ""

    except Exception as exc:
        logger.error(
            'phase=llm_request_exception provider=groq model=%s '
            'exception_type=%s message=%s elapsed_s=%.3f',
            model,
            type(exc).__name__,
            _redact_llm_sensitive_text(
                str(exc),
                sensitive_values=(API_KEY, system_prompt, user_prompt),
            ),
            time.perf_counter() - started,
        )
        return ""


# ============================================================
# LOCAL OLLAMA CHAT
# ============================================================

def _chat_ollama(
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    started = time.perf_counter()
    logger.info('phase=ollama_call_started model=%s', OLLAMA_MODEL)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        content = _extract_ollama_content(response.json())
        logger.info(
            'phase=ollama_call_completed model=%s success=%s elapsed_s=%.3f',
            OLLAMA_MODEL,
            bool(content),
            time.perf_counter() - started,
        )
        return content

    except requests.exceptions.Timeout as exc:
        logger.warning(
            'phase=ollama_call_timeout model=%s error_type=%s elapsed_s=%.3f',
            OLLAMA_MODEL,
            type(exc).__name__,
            time.perf_counter() - started,
        )
        print(
            f"[hosted_llm] Ollama/{OLLAMA_MODEL} error: {exc}"
        )
        return ""

    except Exception as exc:
        if isinstance(exc, requests.exceptions.HTTPError):
            error_response = exc.response
            logger.error(
                'phase=ollama_call_http_error model=%s status_code=%s '
                'reason=%s response_body=%s elapsed_s=%.3f',
                OLLAMA_MODEL,
                getattr(error_response, 'status_code', None),
                getattr(error_response, 'reason', None),
                getattr(error_response, 'text', ''),
                time.perf_counter() - started,
            )
        logger.error(
            'phase=ollama_call_exception model=%s error_type=%s elapsed_s=%.3f',
            OLLAMA_MODEL,
            type(exc).__name__,
            time.perf_counter() - started,
        )
        print(
            f"[hosted_llm] Ollama/{OLLAMA_MODEL} error: {exc}"
        )
        return ""


# ============================================================
# COMMON CHAT ROUTER
# ============================================================

def _chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    reasoning_effort: Optional[str] = None,
) -> str:

    if LLM_PROVIDER == "local":

        return _chat_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout=OLLAMA_TIMEOUT,
        )

    return _chat_groq(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


# ============================================================
# GENERATOR
# ============================================================

def generate_draft(
    system_prompt: str,
    user_prompt: str,
) -> str:

    return _chat(
        model=GENERATOR_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=GENERATOR_TIMEOUT,
        reasoning_effort=GENERATOR_REASONING,
    )


# ============================================================
# VERIFIER
# ============================================================

def verify_and_correct(
    question: str,
    draft_answer: str,
    course_context: str,
    answer_format: str = "general",
    history: str = "",
    long_answer: bool = False,
) -> str:

    if not draft_answer or not course_context:
        return ""

    if long_answer:

        structure_rule = (
            "Preserve the useful academic structure of the draft: headings, "
            "ordered steps, tables, formulas, and explanations when they are "
            "supported by the course material."
        )

    else:

        structure_rule = (
            "Keep the answer concise and directly responsive. Preserve a "
            "comparison table, numbered steps, or other requested format when "
            "supported by the course material."
        )

    system_prompt = f"""
You are the verification and correction model for a university AI Tutor.

Your ONLY factual authority is the COURSE MATERIAL supplied below.

You receive:
1. The student's question.
2. A draft answer produced by another model.
3. The retrieved course material.

Your task is to return the FINAL corrected answer.

STRICT RULES:
- Answer the original STUDENT QUESTION directly.
- Preserve the student's requested task and answer intent.
- Treat the draft as an answer to revise, not as source text to review.
- You may correct, remove, or qualify unsupported claims using only the
  supplied course material.
- Never turn the answer into a source review, critique, correction report,
  proof-editing report, or meta-analysis of the course material.
- Do not add "Key Changes", "Justifications", "Additional Notes about
  revision", or similar editorial sections.
- Return only the final answer to the student.
- Check every factual claim in the draft against the course material.
- Keep claims that are supported.
- Correct claims that conflict with the course material.
- Remove claims that are not supported.
- Never add outside knowledge.
- Never fill missing information from your own knowledge.
- Never invent definitions, examples, values, formulas, algorithms, steps,
  components, relationships, advantages, limitations, classifications,
  calculations, or diagram details.
- If the course material does not support part of the question, state the
  limitation briefly instead of guessing.
- Preserve the terminology used by the course material.
- Preserve correct technical notation.
- Do not mention this verification process.
- Do not mention RAG, retrieval, evidence, chunks, embeddings, models,
  prompts, validation, or internal instructions.
- Return ONLY the final answer, with no preamble or verdict.
- Answer format requested: {answer_format}.
- {structure_rule}
""".strip()

    user_prompt = f"""
STUDENT QUESTION:
{question}

PREVIOUS CONVERSATION:
{history or "(none)"}

DRAFT ANSWER:
--------------------
{draft_answer}
--------------------

COURSE MATERIAL — ONLY FACTUAL AUTHORITY:
--------------------
{course_context}
--------------------

Now verify the draft against the course material and return only the
corrected final answer.
""".strip()

    verifier_started = time.perf_counter()
    if LLM_PROVIDER == "local":
        logger.info(
            'phase=verifier_call_started model=%s',
            OLLAMA_MODEL,
        )

    answer = _chat(
        model=VERIFIER_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout=VERIFIER_TIMEOUT,
        reasoning_effort=VERIFIER_REASONING,
    )

    if LLM_PROVIDER == "local":
        logger.info(
            'phase=verifier_call_completed model=%s success=%s elapsed_s=%.3f',
            OLLAMA_MODEL,
            bool(answer),
            time.perf_counter() - verifier_started,
        )

    return answer


# ============================================================
# CONFIGURATION STATUS
# ============================================================

def configuration_status() -> dict:

    return {
        "provider": LLM_PROVIDER,
        "api_configured": bool(API_KEY),
        "base_url": BASE_URL,
        "generator_model": (
            OLLAMA_MODEL
            if LLM_PROVIDER == "local"
            else GENERATOR_MODEL
        ),
        "verifier_model": (
            OLLAMA_MODEL
            if LLM_PROVIDER == "local"
            else VERIFIER_MODEL
        ),
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
    }
