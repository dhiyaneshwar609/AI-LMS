import unittest

import rag_v2_processor


class SectionDetectionTests(unittest.TestCase):
    def test_rejects_code_output_and_prose(self):
        lines = [
            '{"Name": ["Bumblebee", "Optimus Prime", "Megatron"], '
            '"Label ID":[0,1,2]})',
            "Token: T",
            "structure in the text and treats the whole string as a stream "
            "of characters. Although",
            "This helps deal with misspellings and rare words",
            "The rows of this DataFrame are the one-hot vectors",
            "One simple class of word tokenizers uses whitespace to tokenize "
            "the text",
        ]

        for line in lines:
            with self.subTest(line=line):
                self.assertFalse(rag_v2_processor.looks_like_heading(line))

    def test_accepts_standalone_heading_examples(self):
        lines = [
            "Word Tokenization",
            "Introduction",
            "Natural Language Processing",
            "Text Normalization",
            "Advantages and Disadvantages",
            "HMM-DNN Systems",
            "Components",
            "Architecture",
            "Applications",
        ]

        for line in lines:
            with self.subTest(line=line):
                self.assertTrue(rag_v2_processor.looks_like_heading(line))

    def test_scans_past_code_and_first_thirty_lines(self):
        text = "\n".join(
            ["print(input_ids)"] * 31
            + [
                "Token: T",
                "Word Tokenization",
                "Instead of splitting the text into characters, we can split "
                "it into words and map each word to an integer.",
            ]
        )

        self.assertEqual(rag_v2_processor.detect_section(text), "Word Tokenization")

    def test_preserves_topic_and_general_fallbacks(self):
        self.assertEqual(
            rag_v2_processor.detect_section(
                "This is explanatory material about Cassandra systems and storage."
            ),
            "Cassandra",
        )
        self.assertEqual(
            rag_v2_processor.detect_section(
                "A lower-case body sentence with no known topic."
            ),
            "General",
        )


if __name__ == "__main__":
    unittest.main()
