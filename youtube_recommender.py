"""YouTube recommendation helper for AI-LMS.

Uses yt-dlp's YouTube search without requiring a YouTube API key. The search is
ranked using topic relevance, educational-channel signals, views and duration.
No URL is fabricated: only URLs returned by YouTube search are returned.
"""
from __future__ import annotations

import re
from typing import Any
import socket

try:
    import yt_dlp
except Exception:  # optional dependency; the app can still run without it
    yt_dlp = None

STOP = {
    "what","why","how","is","are","the","a","an","to","of","in","on",
    "for","and","or","this","that","it","from","with","explain","explain",
    "please","give","me","example","examples","can","you","tell","about",
    "does","do","we","i","my","your","part","previous","above","below",
}
EDU_HINTS = {
    "nptel": 10, "iit": 8, "mit": 8, "stanford": 8, "khan academy": 9,
    "freecodecamp": 8, "simplilearn": 6, "great learning": 6, "gate": 5,
    "unacademy": 5, "coursera": 7, "edureka": 5, "academic": 5,
}

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+#.-]{1,}", text.lower())
    return {w for w in words if w not in STOP and len(w) > 1}

def _score(item: dict[str, Any], query: str) -> float:
    title = str(item.get("title", ""))
    channel = str(item.get("channel", ""))
    q = _tokens(query)
    t = _tokens(title)
    overlap = len(q & t)
    coverage = overlap / max(1, len(q))
    exact_phrase = 18 if query.strip().lower() in title.lower() else 0
    edu = 0
    ch = channel.lower()
    for hint, points in EDU_HINTS.items():
        if hint in ch:
            edu += points
    views = float(item.get("view_count") or 0)
    view_bonus = min(8.0, (views ** 0.5) / 300.0) if views > 0 else 0
    duration = float(item.get("duration") or 0)
    duration_bonus = 3 if 180 <= duration <= 3600 else 0
    live_penalty = 4 if "live" in title.lower() else 0
    shorts_penalty = 8 if "#shorts" in title.lower() or "/shorts/" in str(item.get("url", "")).lower() else 0
    return coverage * 35 + overlap * 2 + exact_phrase + edu + view_bonus + duration_bonus - live_penalty - shorts_penalty

def recommend_youtube(question: str, answer: str = "", history: str = "", max_results: int = 8) -> dict[str, Any] | None:
    if yt_dlp is None:
        return None
    question = str(question or "").strip()
    if not question:
        return None
    # Use the answer/history only to resolve contextual references such as "why?".
    context = "\n".join([str(history or "")[-1600:], str(answer or "")[-1800:]])
    query_tokens = _tokens(question)
    if len(query_tokens) <= 2 and context:
        context_tokens = _tokens(context)
        query = question + " " + " ".join(sorted(context_tokens)[:12])
    else:
        query = question
    query = re.sub(r"\s+", " ", query).strip()[:220]
    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }
    try:
        # Keep the recommendation optional and bounded. A YouTube network issue
        # must never block the AI answer for a long time.
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(8)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except Exception:
        return None
    finally:
        try:
            socket.setdefaulttimeout(old_timeout)
        except Exception:
            pass
    entries = (info or {}).get("entries") or []
    candidates = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        url = e.get("webpage_url") or e.get("url")
        if not url or "youtube.com" not in str(url):
            url = f"https://www.youtube.com/watch?v={e['id']}"
        item = {
            "title": str(e.get("title") or "YouTube video").strip(),
            "channel": str(e.get("channel") or e.get("uploader") or "").strip(),
            "url": url,
            "thumbnail": e.get("thumbnail"),
            "duration": e.get("duration") or 0,
            "view_count": e.get("view_count") or 0,
        }
        item["score"] = _score(item, query)
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    # Avoid returning a weak/random match for extremely short or unrelated searches.
    if best["score"] < 8 and len(_tokens(question)) > 1:
        return None
    best.pop("score", None)
    return best
