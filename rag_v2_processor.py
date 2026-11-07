import os
import re
import json
import hashlib
import shutil
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# RAG V2 - UNIVERSAL MATERIAL PROCESSOR
# ============================================================
# Supports:
#   PDF  -> page-by-page direct text + OCR fallback
#   DOCX -> paragraphs + tables
#   PPTX -> slide text
#   TXT  -> direct text
#   PNG/JPG/JPEG -> OCR
#
# Every material is stored independently using:
#   course_id, material_id, filename, page/slide, section, topics
#
# process_material() is the function that app.py will call
# automatically after a teacher uploads a material.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data_v2"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".png", ".jpg", ".jpeg"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_name(name):
    name = os.path.splitext(os.path.basename(str(name)))[0]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120]


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def file_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# TOPIC DETECTION
# ============================================================

TOPIC_PATTERNS = [
    ("Cassandra", [
        r"\bcassandra\b",
        r"\bcassandra architecture\b",
        r"\bcassandra database\b",
    ]),

    ("Hadoop", [
        r"\bhadoop\b",
        r"\bhdfs\b",
        r"\byarn\b",
    ]),

    ("MapReduce", [
        r"\bmapreduce\b",
        r"\bmap function\b",
        r"\breduce function\b",
    ]),

    ("Big Data", [
        r"\bbig data\b",
        r"\bbig data analytics\b",
        r"\bvolume\b",
        r"\bvelocity\b",
        r"\bvariety\b",
    ]),

    ("Global State", [
        r"\bglobal state\b",
        r"\bconsistent global state\b",
    ]),

    ("Causal Ordering", [
        r"\bcausal ordering\b",
        r"\bcausal order\b",
    ]),

    ("Total Ordering", [
        r"\btotal ordering\b",
        r"\btotal order\b",
    ]),

    ("Lamport Mutual Exclusion", [
        r"\blamport.{0,80}mutual exclusion\b",
        r"\bdistributed mutual exclusion\b",
        r"\brequest\b.{0,150}\breply\b.{0,150}\brelease\b",
    ]),

    ("Lamport Logical Clock", [
        r"\blamport.{0,80}logical clock\b",
        r"\blogical clock\b",
        r"\blamport timestamp\b",
    ]),

    ("Rendezvous", [
        r"\brendezvous\b",
    ]),

    ("Distributed Computing", [
        r"\bdistributed computing\b",
        r"\bdistributed system\b",
    ]),
]


def detect_topics(text):
    if not text:
        return []

    lower = text.lower()
    topics = []

    for topic, patterns in TOPIC_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lower):
                topics.append(topic)
                break

    return topics


# ============================================================
# HEADING / SECTION DETECTION
# ============================================================

def looks_like_heading(line):
    line = line.strip()

    if not line or len(line) < 3 or len(line) > 80:
        return False

    numbered_heading = re.match(
        r"^(?:\d+(?:\.\d+)*|[IVX]+)[.)-]?\s+",
        line,
        re.IGNORECASE
    )
    if numbered_heading:
        line = line[numbered_heading.end():].strip()

    if not line or re.search(r"[\[\]{}()=<>|\\/:;,.!?`~@#$%^*+]", line):
        return False

    if re.search(r"(?<![A-Za-z0-9])-(?![A-Za-z0-9])", line):
        return False

    if re.match(
        r"^(?:import|from|print|return|def|class|for|while|if|elif|else|"
        r"try|except|finally|with|as|pass|raise|yield|async|await|lambda|"
        r"assert|del|global|nonlocal|break|continue|traceback|error|"
        r"exception|syntaxerror|warning|valueerror|typeerror|indexerror|"
        r"keyerror|attributeerror|nameerror|importerror|runtimeerror)\b",
        line,
        re.IGNORECASE
    ):
        return False

    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", line)
    if not words or len(words) > 8:
        return False

    sentence_words = {
        "am", "are", "be", "been", "being", "can", "could", "did",
        "do", "does", "had", "has", "have", "is", "may", "might",
        "must", "shall", "should", "was", "were", "will", "would",
        "helps", "uses", "treats", "contains", "provides", "consists",
        "represents", "refers", "means", "allows", "requires", "includes",
        "creates", "produces", "enables", "supports", "shows",
        "indicates", "makes", "takes", "gives", "splits", "maps",
    }
    if any(word.lower() in sentence_words for word in words):
        return False

    minor_words = {"a", "an", "and", "as", "at", "but", "by", "for",
                   "in", "nor", "of", "on", "or", "the", "to", "up", "via"}
    all_caps = all(
        not any(character.isalpha() for character in word)
        or word.isupper()
        for word in words
    )

    title_case = True
    for index, word in enumerate(words):
        if word.isupper():
            continue
        if index > 0 and word.lower() in minor_words and word.islower():
            continue
        if not word[0].isupper() or not word[1:].islower():
            title_case = False
            break

    return all_caps or title_case


def detect_section(text):
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not looks_like_heading(line):
            continue

        following = next(
            (candidate for candidate in lines[index + 1:] if candidate),
            ""
        )
        if not following:
            continue

        if re.search(r"[\[\]{}()=<>|\\/:;`~@#$%^*+]", following):
            continue
        if re.match(
            r"^(?:import|from|print|return|def|class|for|while|if|elif|"
            r"else|try|except|finally|with|as|pass|raise|yield|async|"
            r"await|lambda|assert|del|global|nonlocal|break|continue|"
            r"traceback|error|exception|syntaxerror|warning)\b",
            following,
            re.IGNORECASE
        ):
            continue

        following_words = re.findall(r"[A-Za-z]+", following)
        if (
            len(following) >= 30
            and len(following_words) >= 6
            and sum(word.islower() for word in following_words) >= 4
        ):
            return line[:150]

    topics = detect_topics(text)

    if topics:
        return topics[0]

    return "General"


# ============================================================
# IMAGE / OCR HELPERS
# ============================================================

def render_page_to_image(page, output_path, scale=2.0):
    matrix = pymupdf.Matrix(scale, scale)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    pix.save(str(output_path))


def ocr_image(image_path):
    try:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        return clean_text(text)

    except Exception as exc:
        print(f"OCR error: {exc}")
        return ""


def ocr_pdf_page(page, image_path):
    render_page_to_image(
        page,
        image_path,
        scale=2.0
    )

    return ocr_image(image_path)


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(
    pdf_path,
    material_dir,
    filename,
    course_id,
    material_id
):
    """
    Process every PDF page independently.

    Useful selectable text -> direct extraction.
    Little/no selectable text -> OCR only that page.

    This supports:
      - text PDFs
      - scanned PDFs
      - mixed PDFs
    """

    pdf = pymupdf.open(str(pdf_path))

    pages_dir = material_dir / "pages"
    ensure_directory(pages_dir)

    page_records = []

    total_pages = len(pdf)
    direct_pages = 0
    ocr_pages = 0

    print(f"PDF pages: {total_pages}")

    for page_index in range(total_pages):
        page_number = page_index + 1
        page = pdf[page_index]

        image_path = (
            pages_dir / f"page_{page_number:03d}.png"
        )

        text = clean_text(
            page.get_text("text")
        )

        if len(text) < 100:
            text = ocr_pdf_page(
                page,
                image_path
            )

            ocr_pages += 1
            extraction_method = "ocr"

        else:
            render_page_to_image(
                page,
                image_path,
                scale=1.5
            )

            direct_pages += 1
            extraction_method = "direct_text"

        text = clean_text(text)

        topics = detect_topics(text)
        section = detect_section(text)

        page_records.append({
            "page": page_number,
            "filename": filename,
            "course_id": str(course_id),
            "material_id": str(material_id),
            "text": text,
            "topics": topics,
            "section": section,
            "image_path": str(
                image_path.relative_to(BASE_DIR)
            ),
            "extraction_method": extraction_method,
        })

        print(
            f"[{page_number:03d}/{total_pages}] "
            f"{extraction_method:11s} "
            f"text={len(text):5d} "
            f"topics={topics[:3]}"
        )

    pdf.close()

    return page_records, {
        "direct_text_pages": direct_pages,
        "ocr_pages": ocr_pages,
    }


# ============================================================
# DOCX PROCESSING
# ============================================================

def process_docx(
    file_path,
    material_dir,
    filename,
    course_id,
    material_id
):
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for DOCX files."
        ) from exc

    document = Document(str(file_path))

    records = []
    block_number = 0

    for paragraph in document.paragraphs:
        text = clean_text(paragraph.text)

        if len(text) >= 20:
            records.append({
                "page": block_number + 1,
                "filename": filename,
                "course_id": str(course_id),
                "material_id": str(material_id),
                "text": text,
                "topics": detect_topics(text),
                "section": detect_section(text),
                "image_path": None,
                "extraction_method": "direct_text",
            })

            block_number += 1

    for table in document.tables:
        rows = []

        for row in table.rows:
            cells = [
                clean_text(cell.text)
                for cell in row.cells
            ]

            rows.append(" | ".join(cells))

        text = clean_text("\n".join(rows))

        if len(text) >= 20:
            records.append({
                "page": block_number + 1,
                "filename": filename,
                "course_id": str(course_id),
                "material_id": str(material_id),
                "text": text,
                "topics": detect_topics(text),
                "section": detect_section(text),
                "image_path": None,
                "extraction_method": "direct_text",
            })

            block_number += 1

    return records, {
        "direct_text_pages": len(records),
        "ocr_pages": 0,
    }


# ============================================================
# PPTX PROCESSING
# ============================================================

def process_pptx(
    file_path,
    material_dir,
    filename,
    course_id,
    material_id
):
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "python-pptx is required for PPTX files."
        ) from exc

    presentation = Presentation(str(file_path))

    records = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):
        texts = []

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = clean_text(shape.text)

                if text:
                    texts.append(text)

        combined_text = clean_text(
            "\n".join(texts)
        )

        records.append({
            "page": slide_number,
            "filename": filename,
            "course_id": str(course_id),
            "material_id": str(material_id),
            "text": combined_text,
            "topics": detect_topics(combined_text),
            "section": detect_section(combined_text),
            "image_path": None,
            "extraction_method": "direct_text",
        })

        print(
            f"[SLIDE {slide_number:03d}] "
            f"text={len(combined_text):5d}"
        )

    return records, {
        "direct_text_pages": len(records),
        "ocr_pages": 0,
    }


# ============================================================
# TXT PROCESSING
# ============================================================

def process_txt(
    file_path,
    material_dir,
    filename,
    course_id,
    material_id
):
    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore"
    )

    text = clean_text(text)

    return [{
        "page": 1,
        "filename": filename,
        "course_id": str(course_id),
        "material_id": str(material_id),
        "text": text,
        "topics": detect_topics(text),
        "section": detect_section(text),
        "image_path": None,
        "extraction_method": "direct_text",
    }], {
        "direct_text_pages": 1,
        "ocr_pages": 0,
    }


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    file_path,
    material_dir,
    filename,
    course_id,
    material_id
):
    pages_dir = material_dir / "pages"
    ensure_directory(pages_dir)

    output_path = pages_dir / "page_001.png"

    shutil.copy2(
        file_path,
        output_path
    )

    text = ocr_image(output_path)

    return [{
        "page": 1,
        "filename": filename,
        "course_id": str(course_id),
        "material_id": str(material_id),
        "text": text,
        "topics": detect_topics(text),
        "section": detect_section(text),
        "image_path": str(
            output_path.relative_to(BASE_DIR)
        ),
        "extraction_method": "ocr",
    }], {
        "direct_text_pages": 0,
        "ocr_pages": 1,
    }


# ============================================================
# SMART CHUNKING
# ============================================================

def create_chunks(page_records):
    """
    Keep page boundaries.

    Chunks are created within each page/block and never merge
    unrelated pages.
    """

    chunks = []
    chunk_id = 0

    max_chars = 1400
    overlap = 200

    for record in page_records:
        text = clean_text(
            record.get("text", "")
        )

        if len(text) < 40:
            continue

        page = record.get("page")
        filename = record.get("filename")
        course_id = record.get("course_id")
        material_id = record.get("material_id")
        topics = record.get("topics", [])
        section = record.get("section", "General")
        image_path = record.get("image_path")

        start = 0

        while start < len(text):
            end = min(
                start + max_chars,
                len(text)
            )

            if end < len(text):
                region = text[start:end]

                candidates = [
                    region.rfind(". "),
                    region.rfind("\n"),
                    region.rfind("; "),
                ]

                best_boundary = max(candidates)
                minimum_boundary = int(
                    len(region) * 0.65
                )

                if best_boundary >= minimum_boundary:
                    end = (
                        start
                        + best_boundary
                        + 1
                    )

            chunk_text = clean_text(
                text[start:end]
            )

            if len(chunk_text) >= 80:
                chunks.append({
                    "chunk_id": chunk_id,
                    "course_id": str(course_id),
                    "material_id": str(material_id),
                    "filename": filename,
                    "page": page,
                    "section": section,
                    "topics": topics,
                    "image_path": image_path,
                    "text": chunk_text,
                })

                chunk_id += 1

            if end >= len(text):
                break

            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks


# ============================================================
# SAVE MATERIAL
# ============================================================

def save_material(
    material_dir,
    filename,
    course_id,
    material_id,
    source_path,
    file_hash,
    page_records,
    chunks,
    stats,
    extension
):
    # Keep provenance useful without recording the machine-specific path of
    # the uploaded source. The course/material path is a stable project-local
    # identifier, and the original filename remains in its existing field.
    source_identifier = (
        f"rag_data_v2/{safe_name(course_id)}/{safe_name(material_id)}"
    )
    document = {
        "rag_version": "2",
        "filename": filename,
        "extension": extension,
        "course_id": str(course_id),
        "material_id": str(material_id),
        "source_file": source_identifier,
        "file_sha256": file_hash,
        "page_count": len(page_records),
        "chunk_count": len(chunks),
        "direct_text_pages": stats["direct_text_pages"],
        "ocr_pages": stats["ocr_pages"],
        "pages": page_records,
        "chunks": chunks,
    }

    with open(
        material_dir / "document.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            document,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        material_dir / "chunks.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    metadata = {
        "rag_version": "2",
        "filename": filename,
        "extension": extension,
        "course_id": str(course_id),
        "material_id": str(material_id),
        "source_file": source_identifier,
        "file_sha256": file_hash,
        "page_count": len(page_records),
        "chunk_count": len(chunks),
        "direct_text_pages": stats["direct_text_pages"],
        "ocr_pages": stats["ocr_pages"],
    }

    with open(
        material_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN MATERIAL PROCESSOR
# ============================================================

def process_material(
    file_path,
    course_id,
    material_id=None,
    replace_existing=True
):
    """
    Process ONE material.

    This is the function app.py will call automatically
    after a teacher upload.

    Only this material's RAG folder is replaced.
    Other course materials remain untouched.
    """

    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"Material not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    filename = file_path.name

    if material_id is None:
        material_id = safe_name(filename)

    material_id = str(material_id)
    course_id = str(course_id)

    material_dir = (
        RAG_DIR
        / safe_name(course_id)
        / safe_name(material_id)
    )

    current_hash = file_sha256(
        file_path
    )

    metadata_path = (
        material_dir / "metadata.json"
    )

    if (
        not replace_existing
        and metadata_path.exists()
    ):
        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as f:
                old_metadata = json.load(f)

            if old_metadata.get(
                "file_sha256"
            ) == current_hash:
                print(
                    "Material already processed."
                )
                return old_metadata

        except Exception:
            pass

    if (
        replace_existing
        and material_dir.exists()
    ):
        shutil.rmtree(material_dir)

    ensure_directory(material_dir)

    print()
    print("=" * 70)
    print("RAG V2 MATERIAL PROCESSOR")
    print("=" * 70)
    print(f"Course ID:   {course_id}")
    print(f"Material ID: {material_id}")
    print(f"File:        {filename}")
    print(f"Type:        {extension}")
    print("=" * 70)

    if extension == ".pdf":
        page_records, stats = process_pdf(
            file_path,
            material_dir,
            filename,
            course_id,
            material_id
        )

    elif extension == ".docx":
        page_records, stats = process_docx(
            file_path,
            material_dir,
            filename,
            course_id,
            material_id
        )

    elif extension == ".pptx":
        page_records, stats = process_pptx(
            file_path,
            material_dir,
            filename,
            course_id,
            material_id
        )

    elif extension == ".txt":
        page_records, stats = process_txt(
            file_path,
            material_dir,
            filename,
            course_id,
            material_id
        )

    elif extension in {
        ".png", ".jpg", ".jpeg"
    }:
        page_records, stats = process_image(
            file_path,
            material_dir,
            filename,
            course_id,
            material_id
        )

    else:
        raise ValueError(
            f"Unsupported extension: {extension}"
        )

    chunks = create_chunks(
        page_records
    )

    save_material(
        material_dir=material_dir,
        filename=filename,
        course_id=course_id,
        material_id=material_id,
        source_path=file_path,
        file_hash=current_hash,
        page_records=page_records,
        chunks=chunks,
        stats=stats,
        extension=extension,
    )

    result = {
        "course_id": course_id,
        "material_id": material_id,
        "filename": filename,
        "page_count": len(page_records),
        "chunk_count": len(chunks),
        "direct_text_pages": stats["direct_text_pages"],
        "ocr_pages": stats["ocr_pages"],
        "material_dir": str(material_dir),
        "file_sha256": current_hash,
    }

    print()
    print("=" * 70)
    print("MATERIAL PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Pages/blocks : {len(page_records)}")
    print(f"Chunks       : {len(chunks)}")
    print(
        f"Direct text  : {stats['direct_text_pages']}"
    )
    print(
        f"OCR          : {stats['ocr_pages']}"
    )
    print(
        f"RAG location : {material_dir}"
    )
    print("=" * 70)

    return result


# ============================================================
# INITIAL BUILD FROM EXISTING MATERIALS
# ============================================================

def detect_course_name(filename):
    """
    Used only for the initial rebuild.

    The final Teacher Upload workflow will use the course
    selected in the LMS database instead.
    """

    name = filename.lower()

    if (
        "bda" in name
        or "ccs334" in name
        or "big data" in name
    ):
        return "Big Data Analytics"

    if (
        "cs 3551" in name
        or "distributed" in name
        or "dc final" in name
    ):
        return "Distributed Computing"

    return "Unknown Course"


def process_existing_materials():
    """
    Initial one-time build from files already in materials/.

    Future uploads should call process_material() directly
    from app.py instead of running this function.
    """

    materials_dir = BASE_DIR / "materials"

    if not materials_dir.exists():
        print("Materials directory not found.")
        return

    files = sorted([
        path
        for path in materials_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    if not files:
        print("No supported materials found.")
        return

    print()
    print("=" * 70)
    print("INITIAL RAG V2 BUILD")
    print("=" * 70)
    print(f"Materials found: {len(files)}")
    print("=" * 70)

    for index, file_path in enumerate(
        files,
        start=1
    ):
        course_id = safe_name(
            detect_course_name(
                file_path.name
            )
        )

        material_id = safe_name(
            file_path.name
        )

        print()
        print(
            f"[{index}/{len(files)}] "
            f"{file_path.name}"
        )

        process_material(
            file_path=file_path,
            course_id=course_id,
            material_id=material_id,
            replace_existing=True,
        )


# ============================================================
# COMMAND LINE ENTRY
# ============================================================

if __name__ == "__main__":
    ensure_directory(RAG_DIR)

    process_existing_materials()

    print()
    print("=" * 70)
    print("INITIAL RAG V2 BUILD FINISHED")
    print("=" * 70)
