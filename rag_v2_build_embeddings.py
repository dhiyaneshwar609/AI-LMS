import json
import pickle
import argparse
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# RAG V2 EMBEDDING / INDEX BUILDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data_v2"

MODEL_NAME = "all-MiniLM-L6-v2"


def _validate_id(value, label):
    """Require an ID to already follow rag_v2_processor.safe_name()."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Invalid {label}: {value!r}")

    # Mirror safe_name() locally to avoid importing the PDF/OCR processor.
    safe_value = os.path.splitext(os.path.basename(value))[0]
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_value)[:120]
    if safe_value != value:
        raise ValueError(f"Invalid noncanonical {label}: {value!r}")


def resolve_approved_materials(approved_materials, rag_dir=RAG_DIR):
    """Resolve explicit course/material directories inside the RAG root."""
    if not approved_materials:
        raise ValueError("At least one --approved-material is required.")

    root = Path(rag_dir).resolve()
    resolved = []
    seen = set()

    for value in approved_materials:
        supplied = Path(value)
        requested_relative = supplied
        if supplied.is_absolute():
            try:
                requested_relative = supplied.relative_to(root)
            except ValueError:
                requested_relative = None
        if requested_relative is not None:
            if len(requested_relative.parts) != 2:
                raise ValueError(
                    "Approved material must use exactly two canonical path IDs: "
                    f"{value}"
                )
            _validate_id(requested_relative.parts[0], "course ID")
            _validate_id(requested_relative.parts[1], "material ID")

        directory = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
        try:
            relative = directory.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Approved material must be inside {root}: {value}"
            ) from exc

        if len(relative.parts) != 2 or not directory.is_dir():
            raise ValueError(
                "Approved material must identify an existing "
                "<course_id>/<material_id> directory: "
                f"{value}"
            )

        key = relative.as_posix()
        if key in seen:
            raise ValueError(f"Approved material was listed more than once: {key}")
        seen.add(key)

        course_id, material_id = relative.parts
        _validate_id(course_id, "course ID")
        _validate_id(material_id, "material ID")

        chunks_file = directory / "chunks.json"
        if chunks_file.is_symlink():
            raise ValueError(f"Approved material chunks.json must not be a symlink: {key}")
        try:
            resolved_chunks_file = chunks_file.resolve(strict=True)
            resolved_chunks_file.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Approved material chunks.json must resolve inside the RAG root: {key}"
            ) from exc
        if not resolved_chunks_file.is_file():
            raise ValueError(f"Approved material has no chunks.json: {key}")

        resolved.append({
            "directory": directory,
            "chunks_file": resolved_chunks_file,
            "course_id": course_id,
            "material_id": material_id,
            "relative_directory": key,
        })

    return resolved


def load_all_chunks(approved_materials, rag_dir=RAG_DIR, base_dir=BASE_DIR):
    """Load chunks only from explicitly approved course/material directories."""
    resolved = resolve_approved_materials(approved_materials, rag_dir=rag_dir)
    all_chunks = []
    approved_ids = []
    base = Path(base_dir).resolve()

    for material in resolved:
        chunks_file = material["chunks_file"]
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read valid chunks.json for "
                f"{material['relative_directory']}: {exc}"
            ) from exc

        if not isinstance(chunks, list) or not chunks:
            raise ValueError(
                f"chunks.json must contain a non-empty list: "
                f"{material['relative_directory']}"
            )

        relative_chunks_file = chunks_file.relative_to(base).as_posix()
        for chunk in chunks:
            if not isinstance(chunk, dict) or not str(chunk.get("text", "")).strip():
                raise ValueError(
                    f"Every chunk must be an object with non-empty text: "
                    f"{material['relative_directory']}"
                )
            if str(chunk.get("course_id", "")) != material["course_id"]:
                raise ValueError(
                    f"Chunk course_id does not match approved directory: "
                    f"{material['relative_directory']}"
                )
            if str(chunk.get("material_id", "")) != material["material_id"]:
                raise ValueError(
                    f"Chunk material_id does not match approved directory: "
                    f"{material['relative_directory']}"
                )

            chunk["_source_file"] = relative_chunks_file
            all_chunks.append(chunk)

        approved_ids.append({
            "course_id": material["course_id"],
            "material_id": material["material_id"],
        })

    return all_chunks, approved_ids


def build_index_payload(chunks, embeddings, approved_materials):
    """Build the existing search payload and portable index metadata."""
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    data = {
        "rag_version": "2",
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embedding_array.shape[1]),
        "normalized": True,
        "chunks": chunks,
        "embeddings": embedding_array,
    }
    metadata = {
        "rag_version": "2",
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embedding_array.shape[1]),
        "normalized": True,
        "chunk_count": len(chunks),
        "index_file": "rag_data_v2/index.pkl",
        "approved_materials": approved_materials,
    }
    return data, metadata


def build_embedding_text(chunk):
    """
    Add useful metadata to the text sent to the embedding model.

    This helps distinguish course/material/topic context without
    changing the original chunk text.
    """

    course_id = chunk.get(
        "course_id",
        ""
    )

    filename = chunk.get(
        "filename",
        ""
    )

    section = chunk.get(
        "section",
        ""
    )

    topics = chunk.get(
        "topics",
        []
    )

    topic_text = ", ".join(
        topics
    ) if topics else ""

    text = chunk.get(
        "text",
        ""
    )

    return (
        f"Course: {course_id}\n"
        f"Material: {filename}\n"
        f"Section: {section}\n"
        f"Topics: {topic_text}\n"
        f"Content: {text}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build an approved RAG V2 index.")
    parser.add_argument(
        "--approved-material",
        action="append",
        required=True,
        metavar="COURSE_ID/MATERIAL_ID",
        help="Material directory under rag_data_v2; repeat for each approved material.",
    )
    args = parser.parse_args(argv)

    print()
    print("=" * 70)
    print("RAG V2 EMBEDDING / INDEX BUILD")
    print("=" * 70)

    chunks, approved_materials = load_all_chunks(args.approved_material)

    if not chunks:
        raise RuntimeError(
            "No chunks.json files were found."
        )

    print(
        f"Total chunks found: {len(chunks)}"
    )

    print()
    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    embedding_texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    index_path = (
        RAG_DIR / "index.pkl"
    )

    data, metadata = build_index_payload(
        chunks,
        embeddings,
        approved_materials,
    )

    with open(
        index_path,
        "wb"
    ) as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    with open(
        RAG_DIR / "index_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("RAG V2 EMBEDDING BUILD COMPLETE")
    print("=" * 70)
    print(
        f"Chunks     : {len(chunks)}"
    )
    print(
        f"Dimensions : {embeddings.shape[1]}"
    )
    print(
        f"Index      : {index_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
