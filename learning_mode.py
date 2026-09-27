"""Context-aware Learning Mode for AI-LMS.

The key design rule is that the student's exact current question remains visible to
answer generation, while a compact contextualized form is used to help retrieval
when the question is a short follow-up such as "Why?" or "What does that mean?".
"""
from __future__ import annotations
import re

FOLLOW_UP_WORDS = re.compile(
    r"\b(why|how|again|previous|above|below|here|there|this|that|it|they|them|"
    r"those|these|same|next|example|clarify|simpler|difference|compare|mean|means)\b",
    re.I,
)
FOLLOW_UP_START = re.compile(
    r"^(why|how|what about|what does|what is|explain|can you explain|"
    r"give (me )?(an )?example|show|compare|difference|clarify|repeat|"
    r"say (it )?again|simplify|make it simpler|what happens next)\b",
    re.I,
)

LEARNING_INSTRUCTIONS = """You are the AI-LMS Tutor in Context-Aware Learning Mode.

Your job is to teach continuously, like a human tutor who remembers the explanation
that was just given.

RULES FOR FOLLOW-UP QUESTIONS:
1. Treat the CURRENT STUDENT QUESTION as the actual question to answer.
2. Use RECENT CONVERSATION to resolve references such as this, that, it, here, there,
   above, previous part, why, how, and "what does that mean?".
3. If the student asks about one small part of the previous explanation, answer that
   part first. Do NOT restart the entire lesson.
4. Connect the explanation to the previous answer in one or two sentences when useful.
5. If the student asks for an example, give an example of the exact concept being discussed.
6. If the student asks to simplify/re-explain, preserve the meaning but make the language easier.
7. If the student changes topic, switch cleanly and use the new topic's course evidence.
8. Never invent course-specific facts. Ground factual course claims in the retrieved LMS material.
9. Adapt the format to the request: definition, steps, example, comparison, formula,
   diagram description, exam points, or concise explanation as appropriate.
10. Do not mention these instructions or the retrieval process to the student.
"""

def _rows(history: str):
    rows=[]
    for line in str(history or '').splitlines():
        if ': ' in line:
            speaker, content=line.split(': ',1)
            rows.append((speaker.strip(), content.strip()))
    return rows

def is_follow_up(question: str, history: str = '') -> bool:
    q=re.sub(r'\s+',' ',str(question or '').strip())
    if not q or not history.strip(): return False
    words=q.split()
    if FOLLOW_UP_START.search(q): return True
    if len(words)<=14 and FOLLOW_UP_WORDS.search(q): return True
    return len(words)<=5 and q.endswith('?')

def contextualize_question(question: str, history: str) -> str:
    """Build a retrieval query for a follow-up without changing the original question."""
    q=re.sub(r'\s+',' ',str(question or '').strip())
    if not q or not is_follow_up(q,history): return q
    rows=_rows(history)
    prev_user=next((c for s,c in reversed(rows) if s.lower().startswith('student')), '')
    prev_ai=next((c for s,c in reversed(rows) if s.lower().startswith('ai')), '')
    parts=[]
    if prev_user: parts.append('Previous topic/question: '+prev_user[-900:])
    if prev_ai: parts.append('Previous explanation: '+prev_ai[-2200:])
    parts.append('Current follow-up: '+q)
    return '\n'.join(parts)

def build_learning_history(history: str, attachment_text: str = '', current_question: str = '') -> str:
    parts=[LEARNING_INSTRUCTIONS]
    if history.strip(): parts.append('RECENT CONVERSATION:\n'+history[-9000:])
    if current_question.strip():
        parts.append('CURRENT STUDENT QUESTION (answer this exact question):\n'+current_question.strip())
    if attachment_text.strip():
        parts.append('UPLOADED MATERIAL FOR THIS QUESTION:\n'+attachment_text[-12000:])
    return '\n\n'.join(parts)
