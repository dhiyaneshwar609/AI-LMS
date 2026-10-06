import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# RAG V2 EMBEDDING / INDEX BUILDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag_data_v2"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_all_chunks():
    """Load chunks from every processed course/material."""

    all_chunks = []

    if not RAG_DIR.exists():
        raise FileNotFoundError(
            f"RAG directory not found: {RAG_DIR}"
        )

    for chunks_file in sorted(
        RAG_DIR.rglob("chunks.json")
    ):
        try:
            with open(
                chunks_file,
                "r",
                encoding="utf-8"
            ) as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                continue

            for chunk in chunks:
                chunk["_source_file"] = str(
                    chunks_file.relative_to(BASE_DIR)
                )

            all_chunks.extend(chunks)

        except Exception as exc:
            print(
                f"Skipping {chunks_file}: {exc}"
            )

    return all_chunks


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


def main():
    print()
    print("=" * 70)
    print("RAG V2 EMBEDDING / INDEX BUILD")
    print("=" * 70)

    chunks = load_all_chunks()

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

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    index_path = (
        RAG_DIR / "index.pkl"
    )

    data = {
        "rag_version": "2",
        "model_name": MODEL_NAME,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "normalized": True,
        "chunks": chunks,
        "embeddings": embeddings,
    }

    with open(
        index_path,
        "wb"
    ) as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    metadata = {
        "rag_version": "2",
        "model_name": MODEL_NAME,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "normalized": True,
        "chunk_count": len(chunks),
        "index_file": str(
            index_path.relative_to(BASE_DIR)
        ),
    }

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
