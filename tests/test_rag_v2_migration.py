import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rag_v2_build_embeddings as builder
import rag_v2_processor


class RagV2MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1]
        )
        self.base_dir = Path(self.temp_dir.name)
        self.rag_dir = self.base_dir / "rag_data_v2"
        self.approved_dir = self.rag_dir / "4" / "public_demo_core"
        self.unapproved_dir = self.rag_dir / "4" / "unapproved_extra"
        self.approved_dir.mkdir(parents=True)
        self.unapproved_dir.mkdir(parents=True)
        self.approved_chunk = {
            "chunk_id": 0,
            "course_id": "4",
            "material_id": "public_demo_core",
            "filename": "LeeLearn-demo-core.pdf",
            "page": 1,
            "section": "Definitions",
            "topics": ["demo concept"],
            "image_path": None,
            "text": "A self-authored demo concept with enough text to retrieve.",
        }
        self.unapproved_chunk = {
            **self.approved_chunk,
            "course_id": "4",
            "material_id": "unapproved_extra",
            "text": "Unapproved corpus text must not be included.",
        }
        (self.approved_dir / "chunks.json").write_text(
            json.dumps([self.approved_chunk]), encoding="utf-8"
        )
        (self.unapproved_dir / "chunks.json").write_text(
            json.dumps([self.unapproved_chunk]), encoding="utf-8"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_saved_document_and_metadata_use_safe_source_identifier(self):
        output_dir = self.base_dir / "output"
        output_dir.mkdir()
        source_path = self.base_dir / "private" / "local-only-name.pdf"

        rag_v2_processor.save_material(
            material_dir=output_dir,
            filename="LeeLearn-demo-core.pdf",
            course_id="4",
            material_id="public_demo_core",
            source_path=source_path,
            file_hash="test-hash",
            page_records=[{"page": 1, "text": "Approved text."}],
            chunks=[self.approved_chunk],
            stats={"direct_text_pages": 1, "ocr_pages": 0},
            extension=".pdf",
        )

        document = json.loads((output_dir / "document.json").read_text(encoding="utf-8"))
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        expected_identifier = "rag_data_v2/4/public_demo_core"

        self.assertEqual(document["source_file"], expected_identifier)
        self.assertEqual(metadata["source_file"], expected_identifier)
        self.assertNotIn(str(self.base_dir), document["source_file"])
        self.assertNotIn(str(self.base_dir), metadata["source_file"])
        self.assertEqual(document["course_id"], "4")
        self.assertEqual(document["material_id"], "public_demo_core")
        self.assertEqual(metadata["course_id"], "4")
        self.assertEqual(metadata["material_id"], "public_demo_core")

    def test_missing_approved_material_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "(?i)at least one"):
            builder.load_all_chunks([], rag_dir=self.rag_dir, base_dir=self.base_dir)

        with self.assertRaises(SystemExit):
            builder.main([])

    def test_material_outside_rag_root_is_rejected(self):
        outside_dir = self.base_dir / "outside" / "4" / "other"
        outside_dir.mkdir(parents=True)
        (outside_dir / "chunks.json").write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "inside"):
            builder.load_all_chunks(
                [outside_dir], rag_dir=self.rag_dir, base_dir=self.base_dir
            )

    def test_symlinked_chunks_cannot_escape_rag_root(self):
        outside_chunks = self.base_dir / "outside_chunks.json"
        outside_chunks.write_text(
            json.dumps([self.approved_chunk]), encoding="utf-8"
        )
        chunks_link = self.approved_dir / "chunks.json"
        chunks_link.unlink()
        try:
            chunks_link.symlink_to(outside_chunks)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlink creation unsupported on this platform: {exc}")

        with self.assertRaisesRegex(ValueError, "symlink|resolve inside"):
            builder.load_all_chunks(
                ["4/public_demo_core"],
                rag_dir=self.rag_dir,
                base_dir=self.base_dir,
            )

    def test_duplicate_allowlist_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "listed more than once"):
            builder.load_all_chunks(
                ["4/public_demo_core", "4/public_demo_core/."],
                rag_dir=self.rag_dir,
                base_dir=self.base_dir,
            )

    def test_malformed_course_and_material_ids_are_rejected(self):
        invalid_material = self.rag_dir / "4" / "bad material"
        invalid_material.mkdir()
        (invalid_material / "chunks.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "material ID"):
            builder.resolve_approved_materials(
                ["4/bad material"], rag_dir=self.rag_dir
            )

        invalid_course = self.rag_dir / "bad course" / "public_demo_core"
        invalid_course.mkdir(parents=True)
        (invalid_course / "chunks.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "course ID"):
            builder.resolve_approved_materials(
                ["bad course/public_demo_core"], rag_dir=self.rag_dir
            )

        with self.assertRaisesRegex(ValueError, "exactly two canonical path IDs"):
            builder.resolve_approved_materials(
                ["4/public_demo_core/../public_demo_core"], rag_dir=self.rag_dir
            )

        for malformed in ("", ".", "..", "a/b", "a\\b", "name.pdf", "a" * 121):
            with self.subTest(identifier=malformed):
                with self.assertRaises(ValueError):
                    builder._validate_id(malformed, "material ID")

    def test_existing_course_and_material_ids_are_valid(self):
        for course_id in ("1", "3", "4"):
            builder._validate_id(course_id, "course ID")
        for material_id in ("public_demo_core", "public_demo_extra"):
            builder._validate_id(material_id, "material ID")

    def test_unapproved_chunks_are_excluded_and_ids_fields_preserved(self):
        chunks, approved_materials = builder.load_all_chunks(
            ["4/public_demo_core"],
            rag_dir=self.rag_dir,
            base_dir=self.base_dir,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], self.approved_chunk["text"])
        self.assertNotIn("Unapproved", chunks[0]["text"])
        self.assertEqual(chunks[0]["course_id"], "4")
        self.assertEqual(chunks[0]["material_id"], "public_demo_core")
        self.assertEqual(
            chunks[0]["_source_file"],
            "rag_data_v2/4/public_demo_core/chunks.json",
        )
        for field in (
            "chunk_id", "course_id", "material_id", "filename", "page",
            "section", "topics", "image_path", "text",
        ):
            self.assertIn(field, chunks[0])

        embeddings = np.asarray([[0.25, 0.75]], dtype=np.float32)
        index_data, index_metadata = builder.build_index_payload(
            chunks, embeddings, approved_materials
        )
        self.assertEqual(
            set(index_data),
            {
                "rag_version", "model_name", "embedding_dimension", "normalized",
                "chunks", "embeddings",
            },
        )
        self.assertEqual(index_data["chunks"][0]["course_id"], "4")
        self.assertEqual(index_data["chunks"][0]["material_id"], "public_demo_core")
        self.assertEqual(index_data["embeddings"].shape, (1, 2))
        self.assertEqual(
            index_metadata["approved_materials"],
            [{"course_id": "4", "material_id": "public_demo_core"}],
        )
        self.assertEqual(index_metadata["index_file"], "rag_data_v2/index.pkl")
        self.assertNotIn("\\", index_metadata["index_file"])

    def test_invalid_chunks_file_is_rejected(self):
        (self.approved_dir / "chunks.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            builder.load_all_chunks(
                ["4/public_demo_core"],
                rag_dir=self.rag_dir,
                base_dir=self.base_dir,
            )


if __name__ == "__main__":
    unittest.main()
