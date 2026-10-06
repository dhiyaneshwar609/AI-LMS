import unittest
from unittest.mock import Mock, patch

import ai_tutor
import generator_long
import generator_short


QUESTION = "Explain word tokenization with an example."
SOURCE_RESULTS = [
    {
        "page": 53,
        "text": (
            'text = "Tokenizing text is a core task of NLP."\n'
            "tokenized_text = list(text)\n"
            "print(tokenized_text)\n"
            "['T', 'o', 'k', 'e', 'n', 'i', 'z', 'i', 'n', 'g']"
        ),
    },
    {
        "page": 54,
        "text": (
            "input_ids = [token2idx[token] for token in tokenized_text]\n"
            "print(input_ids)\n"
            "[5, 14, 12, 8]"
        ),
    },
    {
        "page": 55,
        "text": (
            'print(f"Token: {tokenized_text[0]}")\n'
            "Token: T"
        ),
    },
    {
        "page": 56,
        "text": (
            "One simple class of word tokenizers uses whitespace to tokenize "
            "the text.\n"
            "tokenized_text = text.split()\n"
            "print(tokenized_text)\n"
            "['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP.']"
        ),
    },
]
SOURCE_OUTPUT = (
    "['Tokenizing', 'text', 'is', 'a', 'core', 'task', 'of', 'NLP.']"
)


class ExampleAnswerGroundingTests(unittest.TestCase):
    def test_example_answer_uses_retrieved_input_and_exact_output(self):
        generated = (
            "## Concept\nA generated explanation.\n\n"
            "## Course-Supported Example\n"
            "### Input\ninvented input\n"
            "### Output\ninvented output\n"
            "### Explanation of the Example\nSupported explanation."
        )
        with patch("ai_tutor.generate_short_answer", return_value=generated):
            answer = ai_tutor.AnswerAgent().run(
                QUESTION,
                SOURCE_RESULTS,
            )
        answer = ai_tutor._enforce_example_io(
            answer,
            SOURCE_RESULTS,
            QUESTION,
        )

        self.assertIn('text = "Tokenizing text is a core task of NLP."', answer)
        self.assertIn(SOURCE_OUTPUT, answer)
        self.assertNotIn("invented output", answer)
        self.assertLess(answer.index("### Input"), answer.index("### Output"))
        self.assertLess(answer.index("### Output"), answer.index("### Explanation"))

    def test_duplicate_example_sections_are_assembled_once(self):
        generated = (
            "## Concept\nA generated explanation.\n\n"
            "## Course-Supported Example\n"
            "### Input\n\n"
            "### Output\n\n"
            "### Explanation of the Example\nFirst explanation.\n\n"
            "## Course-Supported Example\n"
            "### Input\nsource's duplicate input\n"
            "### Output\nsource's duplicate output\n"
            "### Explanation of the Example\nSecond explanation."
        )
        with patch("ai_tutor.generate_short_answer", return_value=generated):
            answer = ai_tutor.AnswerAgent().run(
                QUESTION,
                SOURCE_RESULTS,
            )
        answer = ai_tutor._enforce_example_io(
            answer,
            SOURCE_RESULTS,
            QUESTION,
        )

        self.assertEqual(answer.count("## Course-Supported Example"), 1)
        self.assertEqual(answer.count("### Input"), 1)
        self.assertEqual(answer.count("### Output"), 1)
        self.assertEqual(answer.count("### Explanation of the Example"), 1)
        self.assertIn(SOURCE_OUTPUT, answer)
        self.assertNotIn("source's duplicate output", answer)
        self.assertNotIn("### Output\n\n###", answer)
        self.assertIn("First explanation.", answer)
        self.assertNotIn(
            "The retrieved course material does not provide additional "
            "explanation for this example.",
            answer,
        )
        self.assertIn("```text\n", answer)

    def test_final_assembly_runs_after_grounding_revision(self):
        revised = (
            "## Word Tokenization\nSupported concept.\n\n"
            "## Course-Supported Example\n"
            "### Input\n\n"
            "### Output\n\n"
            "### Explanation of the Example\n"
            "- Whitespace separates the words.\n"
            "- Punctuation remains attached to the token.\n\n"
            "## Course-Supported Example\n"
            "### Input\nother input\n"
            "### Output\nother output\n"
            "### Explanation of the Example\n"
            "The retrieved material does not provide additional explanation."
        )
        tutor = ai_tutor.AITutor.__new__(ai_tutor.AITutor)
        tutor.query_agent = Mock(run=Mock(return_value={
            "answer_type": "short",
            "retrieval_k": 6,
            "retry_k": 6,
        }))
        tutor.retrieval_agent = Mock(run=Mock(return_value=SOURCE_RESULTS))
        tutor.reranking_agent = Mock(
            run=Mock(side_effect=lambda question, results, top_k: results)
        )
        tutor.validation_agent = Mock(run=Mock(return_value={
            "sufficient": True,
            "reason": "",
            "best_score": 1.0,
            "evidence_count": len(SOURCE_RESULTS),
        }))
        tutor.answer_planner = Mock(run=Mock(return_value={"plan": []}))
        tutor.adaptive_learning_agent = Mock()
        tutor.adaptive_learning_agent.run.return_value = {}
        tutor.adaptive_learning_agent.instruction.return_value = ""
        tutor.answer_agent = Mock(run=Mock(return_value="Initial draft."))
        tutor.answer_grounding_agent = Mock()
        tutor.answer_grounding_agent.run.side_effect = [
            {"needs_revision": True, "grounded": False, "coverage": 0.1},
            {"needs_revision": False, "grounded": True, "coverage": 1.0},
        ]
        tutor.grounded_diagram_agent = Mock(run=Mock(return_value={
            "grounded": False,
            "mermaid": "",
        }))

        with (
            patch("ai_tutor.generate_short_answer", return_value=revised),
            patch(
                "ai_tutor._enforce_example_io",
                wraps=ai_tutor._enforce_example_io,
            ) as enforce,
        ):
            response = tutor.ask(QUESTION, course_id=4)

        answer = response["answer"]
        self.assertEqual(enforce.call_count, 1)
        self.assertEqual(answer.count("## Course-Supported Example"), 1)
        self.assertEqual(answer.count("### Input"), 1)
        self.assertEqual(answer.count("### Output"), 1)
        self.assertEqual(answer.count("### Explanation of the Example"), 1)
        self.assertIn(SOURCE_OUTPUT, answer)
        self.assertIn("Whitespace separates the words.", answer)
        self.assertNotIn(
            "The retrieved material does not provide additional explanation.",
            answer,
        )

    def test_missing_source_output_is_marked_unavailable(self):
        answer = ai_tutor._enforce_example_io(
            "## Concept\nExample.\n\n## Course-Supported Example\n"
            "### Input\nsource input\n### Output\ninvented output",
            [{"text": "Input: source input\nprint(tokenize(text))"}],
            QUESTION,
        )

        self.assertIn(
            "exact output is not available in the extracted text",
            answer,
        )
        self.assertNotIn("invented output", answer)

    def test_unrelated_printed_data_is_not_used_as_example_output(self):
        input_text, output_text = ai_tutor._example_io_from_results(
            SOURCE_RESULTS[1:3],
            QUESTION,
        )

        self.assertEqual(input_text, "")
        self.assertEqual(output_text, "")

    def test_example_format_does_not_change_normal_explanation_routing(self):
        self.assertEqual(
            generator_short.detect_answer_format("Explain word tokenization."),
            "explanation",
        )
        self.assertEqual(
            generator_long.detect_answer_format("Explain word tokenization."),
            "explanation",
        )
        self.assertEqual(
            generator_short.detect_answer_format(QUESTION),
            "example",
        )
        self.assertEqual(
            generator_long.detect_answer_format(QUESTION),
            "example",
        )

        generated = "## Explanation\nExisting explanation."
        with patch("ai_tutor.generate_short_answer", return_value=generated):
            answer = ai_tutor.AnswerAgent().run(
                "Define word tokenization.",
                [{"text": "Word tokenization separates text into words."}],
            )
        self.assertEqual(answer, generated)

    def test_source_records_remain_available(self):
        engine = ai_tutor.AITutor.__new__(ai_tutor.AITutor)
        source = {
            "filename": "NLP-with-Transformer-Models.pdf",
            "page": 56,
            "material_id": 3,
            "course_id": 4,
            "text": "Example source.",
        }

        self.assertEqual(
            engine._source_records([source]),
            [
                {
                    "page": 56,
                    "filename": "NLP-with-Transformer-Models.pdf",
                    "material_id": 3,
                    "course_id": 4,
                    "section": None,
                    "topic": None,
                    "image_path": None,
                    "evidence_type": "primary",
                    "context_score": None,
                    "parent_result_rank": None,
                    "context_rank": None,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
