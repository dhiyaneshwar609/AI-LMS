import os
import re
import json
import secrets
import concurrent.futures
from pathlib import Path
from datetime import timedelta

from flask import Flask, render_template, request, jsonify, session, redirect, url_for

from database import (
    initialize_database,
    create_user,
    authenticate_user,
    get_courses,
    get_materials,
)
from rag_v2_search import search as rag_v2_search
from ai_tutor import get_ai_tutor
from youtube_recommender import recommend_youtube

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from docx import Document
except Exception:
    Document = None


BASE_DIR = Path(__file__).resolve().parent
CHAT_DATA_DIR = BASE_DIR / "student_chats"
FOLDER_DATA_DIR = BASE_DIR / "student_folders"
UPLOAD_DIR = BASE_DIR / "uploads"
AVATAR_DATA_PATH = BASE_DIR / "student_avatars.json"

CHAT_DATA_DIR.mkdir(exist_ok=True)
FOLDER_DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("AI_LMS_SECRET_KEY", "change-this-secret-key")
app.permanent_session_lifetime = timedelta(days=365)

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
JOBS = {}


# ============================================================
# USER / DATABASE
# ============================================================

initialize_database()


def normalize_user(user):
    if user is None:
        return None
    if isinstance(user, dict):
        return {
            "id": user.get("id"),
            "name": user.get("name", "Student"),
            "email": user.get("email", ""),
            "role": user.get("role", "Student"),
        }
    if isinstance(user, (tuple, list)) and len(user) >= 4:
        return {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "role": user[3],
        }
    return None


def get_current_user():
    return normalize_user(session.get("user"))


def require_user():
    user = get_current_user()
    if not user:
        return None, (jsonify({"ok": False, "error": "Not logged in"}), 401)
    return user, None


# ============================================================
# RAG COURSE DETECTION
# Same logic used by your current Streamlit application.
# ============================================================

def detect_course(question):
    try:
        results = rag_v2_search(
            question,
            course_id=None,
            material_id=None,
            top_k=12,
        )
    except Exception:
        return None

    if not results:
        return None

    course_scores = {}

    for result in results:
        course_id = str(result.get("course_id", "")).strip()
        if not course_id:
            continue

        score = float(
            result.get(
                "semantic_score",
                result.get("score", 0.0),
            ) or 0.0
        )

        course_scores[course_id] = max(
            course_scores.get(course_id, 0.0),
            score,
        )

    if not course_scores:
        return None

    best_course, best_score = max(
        course_scores.items(),
        key=lambda item: item[1],
    )

    return best_course if best_score >= 0.35 else None


def ask_tutor(question, history_text=""):
    course_id = detect_course(question)
    tutor = get_ai_tutor()
    return tutor.ask(
        question=question,
        course_id=course_id,
        history=history_text,
    )


def normalize_ai_result(raw):
    """
    Handles both:
      tutor.ask(...) -> {"answer": ..., "sources": ...}
    and older wrapped results.
    """
    if isinstance(raw, str):
        return {"answer": raw, "sources": [], "diagram": None}

    if not isinstance(raw, dict):
        return {
            "answer": str(raw),
            "sources": [],
            "diagram": None,
        }

    # Current/expected result.
    if "answer" in raw:
        return {
            "answer": str(raw.get("answer", "")),
            "sources": raw.get("sources", []) or [],
            "diagram": raw.get("diagram"),
        }

    # Older Streamlit wrapper:
    # {"type":"answer","result": {...}}
    nested = raw.get("result")
    if isinstance(nested, dict):
        return normalize_ai_result(nested)

    return {
        "answer": str(
            raw.get("content")
            or raw.get("response")
            or raw.get("text")
            or "I could not generate an answer."
        ),
        "sources": raw.get("sources", []) or [],
        "diagram": raw.get("diagram"),
    }


# ============================================================
# CHAT STORAGE
# ============================================================

def chat_file(user_id):
    return CHAT_DATA_DIR / f"student_{int(user_id)}.json"


def load_chats(user_id):
    path = chat_file(user_id)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    cleaned = []
    for chat in data:
        if not isinstance(chat, dict):
            continue
        chat.setdefault("pinned", False)
        chat.setdefault("folder_id", None)
        chat.setdefault("messages", [])
        if chat.get("messages"):
            cleaned.append(chat)

    return cleaned


def save_chats(user_id, chats):
    path = chat_file(user_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(chats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def chat_title(messages):
    for message in messages:
        if message.get("role") == "user":
            text = re.sub(
                r"\s+",
                " ",
                str(message.get("content", "")).strip(),
            )
            if not text:
                return "New Chat"
            return text[:42].rstrip() + ("..." if len(text) > 42 else "")
    return "New Chat"


def find_chat(chats, chat_id):
    return next(
        (c for c in chats if str(c.get("id")) == str(chat_id)),
        None,
    )


# ============================================================
# FOLDERS
# ============================================================

def folder_file(user_id):
    return FOLDER_DATA_DIR / f"student_{int(user_id)}.json"


def load_folders(user_id):
    path = folder_file(user_id)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            f for f in data
            if isinstance(f, dict) and f.get("id")
        ] if isinstance(data, list) else []
    except Exception:
        return []


def save_folders(user_id, folders):
    path = folder_file(user_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(folders, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ============================================================
# AVATAR
# ============================================================

def load_avatar_data():
    if not AVATAR_DATA_PATH.exists():
        return {}

    try:
        data = json.loads(AVATAR_DATA_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def avatar_for_user(user_id):
    data = load_avatar_data()
    value = data.get(str(user_id), "boy")

    if isinstance(value, dict):
        path = value.get("path")
        if path and Path(path).exists():
            return Path(path)
        gender = str(value.get("gender", "boy")).lower()
    else:
        gender = str(value).lower()

    candidates = []
    if gender == "girl":
        candidates = [
            BASE_DIR / "OIP.webp",
            BASE_DIR / "girl.webp",
            BASE_DIR / "girl.png",
        ]
    else:
        candidates = [
            BASE_DIR / "3d-student-boy-avatar-bring-tablete-icon-free-png.webp",
            BASE_DIR / "boy.webp",
            BASE_DIR / "boy.png",
        ]

    return next((p for p in candidates if p.exists()), None)


# ============================================================
# ATTACHMENT EXTRACTION
# ============================================================

def extract_pdf(path):
    if fitz is None:
        return ""

    parts = []
    try:
        pdf = fitz.open(str(path))
        for page in pdf:
            text = page.get_text("text").strip()

            if text:
                parts.append(text)
                continue

            if pytesseract is None or Image is None:
                continue

            try:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )
                image = Image.open(
                    __import__("io").BytesIO(pix.tobytes("png"))
                ).convert("RGB")
                ocr = pytesseract.image_to_string(
                    image,
                    config="--psm 6",
                ).strip()
                if ocr:
                    parts.append(ocr)
            except Exception:
                pass

        pdf.close()
    except Exception:
        return ""

    return "\n\n".join(parts).strip()


def extract_attachment(path, filename):
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(path)

    if ext in {".txt", ".md", ".py", ".json", ".csv"}:
        try:
            return Path(path).read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        except Exception:
            return ""

    if ext == ".docx" and Document is not None:
        try:
            doc = Document(str(path))
            parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    ]
                    if cells:
                        parts.append(" | ".join(cells))
            return "\n".join(parts).strip()
        except Exception:
            return ""

    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        if Image is None or pytesseract is None:
            return ""
        try:
            image = Image.open(path).convert("RGB")
            return pytesseract.image_to_string(
                image,
                config="--psm 6",
            ).strip()
        except Exception:
            return ""

    return ""


# ============================================================
# AI JOBS
# ============================================================

import sys
import subprocess
import time

JOB_DIR = BASE_DIR / "jobs"
JOB_DIR.mkdir(exist_ok=True)


def build_history(messages):
    lines = []
    for message in messages[-6:]:
        speaker = "Student" if message.get("role") == "user" else "AI Tutor"
        lines.append(f"{speaker}: {message.get('content', '')}")
    return "\n".join(lines)


def create_ai_job(user_id, chat_id, question, history_text, attachment_text, learning_mode=False):
    job_id = secrets.token_hex(12)
    request_path = JOB_DIR / f"{job_id}.request.json"
    result_path = JOB_DIR / f"{job_id}.result.json"
    payload = {
        "question": question,
        "history_text": history_text,
        "attachment_text": attachment_text,
        "learning_mode": bool(learning_mode),
    }
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    worker_path = BASE_DIR / "ai_worker.py"
    process = subprocess.Popen(
        [sys.executable, str(worker_path), str(request_path), str(result_path)],
        cwd=str(BASE_DIR),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    JOBS[job_id] = {
        "status": "running",
        "answer": "",
        "sources": [],
        "diagram": None,
        "youtube": None,
        "learning_mode": bool(learning_mode),
        "user_id": str(user_id),
        "chat_id": str(chat_id),
        "question": question,
        "process": process,
        "request_path": str(request_path),
        "result_path": str(result_path),
    }
    return job_id


def cleanup_job_files(job):
    for key in ("request_path", "result_path"):
        try:
            Path(job.get(key, "")).unlink(missing_ok=True)
        except Exception:
            pass


def read_finished_job(job):
    result_path = Path(job["result_path"])
    process = job.get("process")
    if process is not None and process.poll() is None:
        return False

    if not result_path.exists():
        if process is not None and process.returncode not in (0, None):
            job["status"] = "error"
            job["answer"] = "Generation failed. Please try again."
            job["sources"] = []
            job["detail"] = f"AI worker exited with code {process.returncode}. See logs/ai_worker.log."
            return True
        if process is not None and process.returncode == 0:
            job["status"] = "error"
            job["answer"] = "The AI worker finished without returning an answer."
            job["sources"] = []
            job["detail"] = "Missing worker result file."
            return True
        return False

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        result = {"ok": False, "answer": "Generation failed. Please try again.", "sources": []}

    if result.get("ok"):
        job["status"] = "done"
        job["answer"] = str(result.get("answer", ""))
        job["sources"] = result.get("sources", []) or []
        job["diagram"] = result.get("diagram")
        job["youtube"] = result.get("youtube")
        job["learning_mode"] = bool(result.get("learning_mode", job.get("learning_mode", False)))
    else:
        job["status"] = "error"
        job["answer"] = result.get("answer", "I could not generate the answer. Please try again.")
        job["sources"] = result.get("sources", []) or []
        job["diagram"] = result.get("diagram")
        job["detail"] = result.get("detail", "")
        job["youtube"] = None
    return True


# ============================================================
# PAGE ROUTES
# ============================================================

@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s from browsers that request a favicon automatically.
    return ("", 204)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "AI-LMS Flask Tutor"})


@app.get("/")
def index():
    if not get_current_user():
        return redirect(url_for("login_page"))
    return render_template("tutor.html", user=get_current_user())


@app.get("/login")
def login_page():
    if get_current_user():
        return redirect(url_for("index"))
    return render_template("login.html")


# ============================================================
# AUTH API
# ============================================================

@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not email or not password:
        return jsonify({"ok": False, "error": "Enter email and password."}), 400
    try:
        user = normalize_user(authenticate_user(email, password))
    except Exception:
        user = None
    if not user:
        return jsonify({"ok": False, "error": "Invalid email or password."}), 401
    session.permanent = True
    session["user"] = user
    return jsonify({"ok": True, "user": user})


@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if not name or not email or not password:
        return jsonify({"ok": False, "error": "All fields are required."}), 400
    try:
        success = create_user(name, email, password, "Student")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not success:
        return jsonify({"ok": False, "error": "An account with this email already exists."}), 409
    return jsonify({"ok": True, "message": "Account created successfully."})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


# ============================================================
# LMS API
# ============================================================

@app.get("/api/lms")
def api_lms():
    user, error = require_user()
    if error:
        return error
    try:
        courses = get_courses() or []
    except Exception:
        courses = []
    try:
        materials = get_materials() or []
    except Exception:
        materials = []
    return jsonify({"ok": True, "courses": courses, "materials": materials})


# ============================================================
# CHAT API
# ============================================================

@app.get("/api/chats")
def api_chats():
    user, error = require_user()
    if error:
        return error
    return jsonify({"ok": True, "chats": load_chats(user["id"])})


@app.post("/api/chats")
def api_create_chat():
    user, error = require_user()
    if error:
        return error
    chat = {"id": secrets.token_hex(8), "title": "New Chat", "pinned": False, "folder_id": None, "messages": []}
    chats = load_chats(user["id"])
    chats.insert(0, chat)
    save_chats(user["id"], chats)
    return jsonify({"ok": True, "chat": chat})


@app.get("/api/chats/<chat_id>")
def api_get_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    chat = find_chat(load_chats(user["id"]), chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    return jsonify({"ok": True, "chat": chat})


@app.patch("/api/chats/<chat_id>")
def api_update_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()[:120]
    if not title:
        return jsonify({"ok": False, "error": "Chat title cannot be empty."}), 400
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    chat["title"] = title
    save_chats(user["id"], chats)
    return jsonify({"ok": True, "chat": chat})


@app.delete("/api/chats/<chat_id>")
def api_delete_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    chats = load_chats(user["id"])
    chats = [c for c in chats if str(c.get("id")) != str(chat_id)]
    save_chats(user["id"], chats)
    return jsonify({"ok": True})


@app.post("/api/chats/<chat_id>/pin")
def api_pin_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    chat["pinned"] = not bool(chat.get("pinned"))
    save_chats(user["id"], chats)
    return jsonify({"ok": True, "pinned": chat["pinned"]})


@app.post("/api/chats/<chat_id>/move")
def api_move_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    folder_id = data.get("folder_id")
    if folder_id in (None, "", "null"):
        folder_id = None
    else:
        folder_id = str(folder_id)
        if not any(str(f.get("id")) == folder_id for f in load_folders(user["id"])):
            return jsonify({"ok": False, "error": "Folder not found."}), 404
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    chat["folder_id"] = folder_id
    save_chats(user["id"], chats)
    return jsonify({"ok": True, "chat": chat})


# ============================================================
# FOLDER API
# ============================================================

@app.get("/api/folders")
def api_folders():
    user, error = require_user()
    if error:
        return error
    return jsonify({"ok": True, "folders": load_folders(user["id"])})


@app.post("/api/folders")
def api_create_folder():
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:80]
    if not name:
        return jsonify({"ok": False, "error": "Folder name is required."}), 400
    folders = load_folders(user["id"])
    if any(str(f.get("name", "")).lower() == name.lower() for f in folders):
        return jsonify({"ok": False, "error": "Folder already exists."}), 409
    folder = {"id": secrets.token_hex(8), "name": name, "pinned": False}
    folders.insert(0, folder)
    save_folders(user["id"], folders)
    return jsonify({"ok": True, "folder": folder})


@app.post("/api/folders/<folder_id>/chats")
def api_create_folder_chat(folder_id):
    user, error = require_user()
    if error:
        return error

    folder_id = str(folder_id)
    folders = load_folders(user["id"])
    folder = next((f for f in folders if str(f.get("id")) == folder_id), None)
    if not folder:
        return jsonify({"ok": False, "error": "Folder not found."}), 404

    chat = {
        "id": secrets.token_hex(8),
        "title": "New Chat",
        "pinned": False,
        "folder_id": folder_id,
        "messages": [],
    }

    chats = load_chats(user["id"])
    chats.insert(0, chat)
    save_chats(user["id"], chats)
    return jsonify({"ok": True, "chat": chat, "folder": folder})


@app.patch("/api/folders/<folder_id>")
def api_update_folder(folder_id):
    user, error = require_user()
    if error:
        return error

    payload = request.get_json(silent=True) or {}
    folders = load_folders(user["id"])
    folder = next((f for f in folders if str(f.get("id")) == str(folder_id)), None)
    if not folder:
        return jsonify({"ok": False, "error": "Folder not found."}), 404

    name = str(payload.get("name", "")).strip()[:80]
    if not name:
        return jsonify({"ok": False, "error": "Folder name cannot be empty."}), 400

    if any(
        str(f.get("id")) != str(folder_id)
        and str(f.get("name", "")).strip().lower() == name.lower()
        for f in folders
    ):
        return jsonify({"ok": False, "error": "Folder already exists."}), 409

    folder["name"] = name
    save_folders(user["id"], folders)
    return jsonify({"ok": True, "folder": folder})


@app.post("/api/folders/<folder_id>/pin")
def api_pin_folder(folder_id):
    user, error = require_user()
    if error:
        return error

    folders = load_folders(user["id"])
    folder = next((f for f in folders if str(f.get("id")) == str(folder_id)), None)
    if not folder:
        return jsonify({"ok": False, "error": "Folder not found."}), 404

    folder["pinned"] = not bool(folder.get("pinned"))
    save_folders(user["id"], folders)
    return jsonify({"ok": True, "folder": folder})


@app.delete("/api/folders/<folder_id>")
def api_delete_folder(folder_id):
    user, error = require_user()
    if error:
        return error

    folders = load_folders(user["id"])
    folder = next((f for f in folders if str(f.get("id")) == str(folder_id)), None)
    if not folder:
        return jsonify({"ok": False, "error": "Folder not found."}), 404

    folder_id = str(folder_id)
    remaining = [f for f in folders if str(f.get("id")) != folder_id]
    save_folders(user["id"], remaining)

    # Deleting a folder does not delete its conversations. They become
    # normal recent chats instead of being silently destroyed.
    chats = load_chats(user["id"])
    changed = False
    for chat in chats:
        if str(chat.get("folder_id") or "") == folder_id:
            chat["folder_id"] = None
            changed = True
    if changed:
        save_chats(user["id"], chats)

    return jsonify({"ok": True, "folder_id": folder_id})


# ============================================================
# CHAT -> AI
# ============================================================

def start_chat_job(user_id, chat, question, attachment_text="", append_user=True):
    messages = chat.get("messages", [])
    history_text = build_history(messages)
    if append_user:
        messages.append({"id": secrets.token_hex(5), "role": "user", "content": question})
        chat["title"] = chat_title(messages)
    save_chats(user_id, load_chats(user_id))


def start_generation(user_id, chat, question, attachment_text="", append_user=True, learning_mode=False):
    chats = load_chats(user_id)
    stored = find_chat(chats, chat["id"])
    if not stored:
        chats.insert(0, chat)
        stored = chat
    history_text = build_history(stored.get("messages", []))
    if append_user:
        stored.setdefault("messages", []).append({"id": secrets.token_hex(5), "role": "user", "content": question})
        stored["title"] = chat_title(stored["messages"])
    save_chats(user_id, chats)
    return create_ai_job(user_id, stored["id"], question, history_text, attachment_text, learning_mode=learning_mode)


@app.post("/api/chat")
def api_chat():
    user, error = require_user()
    if error:
        return error
    question = str(request.form.get("question", "")).strip()
    chat_id = str(request.form.get("chat_id", "")).strip()
    learning_mode = str(request.form.get("learning_mode", "0")).lower() in {"1", "true", "yes", "on"}
    if not question:
        return jsonify({"ok": False, "error": "Question is empty."}), 400
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        chat = {"id": chat_id or secrets.token_hex(8), "title": "New Chat", "pinned": False, "folder_id": None, "messages": []}
        chats.insert(0, chat)
        save_chats(user["id"], chats)

    uploaded_texts = []
    for uploaded in request.files.getlist("files"):
        if not uploaded or not uploaded.filename:
            continue
        safe_name = Path(uploaded.filename).name
        saved_path = UPLOAD_DIR / f"{secrets.token_hex(8)}_{safe_name}"
        uploaded.save(saved_path)
        extracted = extract_attachment(saved_path, safe_name)
        if extracted:
            uploaded_texts.append(f"--- {safe_name} ---\n{extracted}")
    attachment_text = "\n\n".join(uploaded_texts)
    job_id = start_generation(user["id"], chat, question, attachment_text, append_user=True, learning_mode=learning_mode)
    return jsonify({"ok": True, "job_id": job_id, "chat_id": chat["id"]})


@app.post("/api/chats/<chat_id>/retry")
def api_retry_chat(chat_id):
    user, error = require_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    message_id = str(payload.get("message_id", "")).strip()
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    messages = chat.get("messages", [])
    answer_index = None
    if message_id:
        for i, msg in enumerate(messages):
            if str(msg.get("id", "")) == message_id and msg.get("role") == "assistant":
                answer_index = i
                break
    if answer_index is None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                answer_index = i
                break
    if answer_index is None:
        return jsonify({"ok": False, "error": "No answer to retry."}), 400
    question = ""
    for j in range(answer_index - 1, -1, -1):
        if messages[j].get("role") == "user":
            question = str(messages[j].get("content", "")).strip()
            break
    if not question:
        return jsonify({"ok": False, "error": "Question for this answer was not found."}), 400
    previous_answer = messages[answer_index]
    retry_learning_mode = bool(previous_answer.get("learning_mode", False))
    messages.pop(answer_index)
    save_chats(user["id"], chats)
    job_id = start_generation(user["id"], chat, question, "", append_user=False, learning_mode=retry_learning_mode)
    return jsonify({"ok": True, "job_id": job_id, "chat_id": chat_id, "question": question})


@app.post("/api/jobs/<job_id>/stop")
def api_stop_job(job_id):
    user, error = require_user()
    if error:
        return error
    job = JOBS.get(job_id)
    if not job or str(job.get("user_id")) != str(user["id"]):
        return jsonify({"ok": False, "error": "Job not found."}), 404
    if job.get("status") != "running":
        return jsonify({"ok": True, "status": job.get("status")})
    process = job.get("process")
    try:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
        job["status"] = "cancelled"
        cleanup_job_files(job)
        return jsonify({"ok": True, "status": "cancelled"})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/jobs/<job_id>")
def api_job(job_id):
    user, error = require_user()
    if error:
        return error
    job = JOBS.get(job_id)
    if not job or str(job.get("user_id")) != str(user["id"]):
        return jsonify({"ok": False, "error": "Job not found."}), 404
    if job.get("status") == "running":
        read_finished_job(job)
    if job.get("status") == "done":
        chats = load_chats(user["id"])
        target = find_chat(chats, job["chat_id"])
        message_id = secrets.token_hex(5)
        if target is not None:
            target.setdefault("messages", []).append({"id": message_id, "role": "assistant", "content": job.get("answer", ""), "sources": job.get("sources", []), "diagram": job.get("diagram"), "youtube": job.get("youtube"), "learning_mode": bool(job.get("learning_mode", False))})
            save_chats(user["id"], chats)
        result = {k: v for k, v in job.items() if k not in {"process", "request_path", "result_path"}}
        result["message_id"] = message_id
        result["ok"] = True
        cleanup_job_files(job)
        JOBS.pop(job_id, None)
        return jsonify(result)
    if job.get("status") == "error":
        result = {k: v for k, v in job.items() if k not in {"process", "request_path", "result_path"}}
        result["ok"] = True
        cleanup_job_files(job)
        JOBS.pop(job_id, None)
        return jsonify(result)
    return jsonify({"ok": True, "status": job.get("status", "running")})


# ============================================================
# ANSWER ACTIONS / PDF
# ============================================================

@app.post("/api/chats/<chat_id>/answer-folder")
def api_answer_folder(chat_id):
    return api_move_chat(chat_id)


@app.post("/api/chats/<chat_id>/answer-pdf")
def api_answer_pdf(chat_id):
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    message_id = str(data.get("message_id", "")).strip()
    chats = load_chats(user["id"])
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"ok": False, "error": "Chat not found."}), 404
    messages = chat.get("messages", [])
    answer = None
    question = None
    for i, msg in enumerate(messages):
        if message_id and str(msg.get("id", "")) == message_id and msg.get("role") == "assistant":
            answer = str(msg.get("content", ""))
            for j in range(i - 1, -1, -1):
                if messages[j].get("role") == "user":
                    question = str(messages[j].get("content", ""))
                    break
            break
    if answer is None and messages:
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "assistant":
                answer = str(messages[i].get("content", ""))
                question = next((str(messages[j].get("content", "")) for j in range(i-1, -1, -1) if messages[j].get("role") == "user"), "")
                break
    if not answer:
        return jsonify({"ok": False, "error": "Answer not found."}), 404
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import mm
    except Exception:
        return jsonify({"ok": False, "error": "PDF support is not installed. Run: python -m pip install reportlab"}), 500
    pdf_dir = BASE_DIR / "generated_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    filename = f"{secrets.token_hex(8)}.pdf"
    path = pdf_dir / filename
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("LMSNormal", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8)
    heading = ParagraphStyle("LMSHeading", parent=styles["Heading1"], fontSize=17, leading=22, alignment=TA_LEFT, spaceAfter=14)
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    import html as _html
    story = [Paragraph("Lee AI Tutor", heading), Paragraph("Question", styles["Heading2"]), Paragraph(_html.escape(question or ""), normal), Paragraph("Answer", styles["Heading2"])]
    for part in answer.split("\n"):
        story.append(Paragraph(_html.escape(part) if part.strip() else "&nbsp;", normal))
    doc.build(story)
    from flask import send_file
    return send_file(path, as_attachment=True, download_name="LeeLearn-Answer.pdf", mimetype="application/pdf")


def run_ai_job(job_id, question, history_text, attachment_text):
    try:
        final_question = question

        if attachment_text:
            final_question = f"""
The student uploaded the following material.

Use the attached material when it is relevant to the student's request.

--- ATTACHED MATERIAL ---
{attachment_text}
--- END ATTACHED MATERIAL ---

Student request:
{question}
""".strip()

        raw = ask_tutor(
            final_question,
            history_text,
        )

        JOBS[job_id] = {
            "status": "done",
            **normalize_ai_result(raw),
        }

    except Exception as exc:
        JOBS[job_id] = {
            "status": "error",
            "answer": "I could not generate the answer. Please try again.",
            "sources": [],
            "detail": str(exc),
        }



# ============================================================
# AVATAR
# ============================================================

@app.get("/api/avatar")
def api_avatar():
    user, error = require_user()
    if error:
        return error

    path = avatar_for_user(user["id"])
    if not path:
        return jsonify({"ok": True, "avatar": None})

    from flask import send_file
    return send_file(path)


# ============================================================
# YOUTUBE LEARNING RECOMMENDATION
# ============================================================

@app.post("/api/youtube-recommendation")
def api_youtube_recommendation():
    user, error = require_user()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    history = str(data.get("history", "")).strip()
    if not question:
        return jsonify({"ok": False, "error": "Question is required."}), 400
    try:
        video = recommend_youtube(question, answer, history)
        return jsonify({"ok": True, "youtube": video})
    except Exception as exc:
        return jsonify({"ok": False, "youtube": None, "error": str(exc)}), 500



if __name__ == "__main__":
    # Bind to all interfaces so the same Wi-Fi network can open AI-LMS from
    # another device. Disable Flask's reloader so it cannot start duplicate
    # workers or interrupt long-running local-model jobs.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
