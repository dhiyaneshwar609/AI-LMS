import unittest
from unittest.mock import Mock

from ai_tutor import AITutor
from multi_question import decompose_questions


class MultiQuestionTests(unittest.TestCase):
    def texts(self, message, history=""):
        return [item["text"] for item in decompose_questions(message, history)["questions"]]

    def test_requested_cases(self):
        cases = [
            ("What is Hadoop?", 1),
            ("What is Hadoop? Explain HDFS.", 2),
            ("Explain Hadoop architecture. Compare HDFS and HBase.", 2),
            ("Explain Hadoop architecture and describe NameNode and DataNode.", 1),
            ("Give an example of MapReduce processing.", 1),
            ("Explain the steps:\n1. Split\n2. Map\n3. Shuffle\n4. Reduce", 1),
            ("What is HDFS? Give 2 marks answer.", 1),
            ("Explain Hadoop architecture for 13 marks. Give 2 marks answer for HDFS.", 2),
            ("Explain it simply with an example.", 1),
            ("   \n", 0),
        ]
        for message, expected_count in cases:
            with self.subTest(message=message):
                self.assertEqual(len(self.texts(message, "What is Lamport's algorithm?")), expected_count)

    def test_marks_requirements_are_scoped_to_individual_questions(self):
        result = decompose_questions(
            "Explain Hadoop architecture for 13 marks. Give 2 marks answer for HDFS."
        )
        self.assertEqual(result["questions"][0]["text"], "Explain Hadoop architecture for 13 marks.")
        self.assertEqual(result["questions"][1]["text"], "Give 2 marks answer for HDFS.")

    def test_contextual_follow_up_is_separate_and_inherits_previous_topic(self):
        result = decompose_questions(
            "Give 2 marks answer for HDFS. Explain it simply with an example."
        )
        self.assertEqual(len(result["questions"]), 2)
        self.assertEqual(result["questions"][1]["text"], "Explain it simply with an example.")
        self.assertEqual(
            result["questions"][1]["context_question"],
            "Give 2 marks answer for HDFS.",
        )

    def test_single_contextual_follow_up_remains_single(self):
        result = decompose_questions("Explain it simply with an example.")
        self.assertFalse(result["is_multi_question"])
        self.assertEqual(len(result["questions"]), 1)

    def test_unsupported_question_does_not_erase_supported_answer(self):
        tutor = AITutor.__new__(AITutor)
        supported = {
            "answer": "Hadoop is a distributed framework.",
            "sources": [{"filename": "course.pdf", "page": 2, "material_id": "m1"}],
            "validation": {"sufficient": True},
            "answer_grounding": {"grounded": True},
            "attempts": 1,
        }
        unsupported = {
            "answer": "I could not find enough relevant information.",
            "sources": [],
            "validation": {"sufficient": False},
            "answer_grounding": {"grounded": False},
            "attempts": 2,
        }
        tutor.ask = Mock(side_effect=[supported, unsupported])

        response = AITutor.ask(
            tutor,
            "What is Hadoop? Explain Lamport's algorithm.",
        )

        self.assertIn(supported["answer"], response["answer"])
        self.assertIn(unsupported["answer"], response["answer"])
        self.assertTrue(response["validation"]["sufficient"])
        self.assertTrue(response["validation"]["partial"])
        self.assertEqual([item["order"] for item in response["questions"]], [1, 2])
        self.assertEqual(response["questions"][0]["result"]["sources"], supported["sources"])

    def test_multi_question_follow_up_receives_local_topic_context(self):
        tutor = AITutor.__new__(AITutor)
        tutor.ask = Mock(side_effect=[
            {"answer": "HDFS stores files.", "sources": [], "validation": {}, "answer_grounding": {}, "attempts": 1},
            {"answer": "Simple explanation.", "sources": [], "validation": {}, "answer_grounding": {}, "attempts": 1},
        ])

        AITutor.ask(
            tutor,
            "What is HDFS? Explain it simply with an example.",
            history="Student: Earlier unrelated question?",
        )

        follow_up_call = tutor.ask.call_args_list[1]
        self.assertEqual(
            follow_up_call.kwargs["history"],
            "Student: What is HDFS?",
        )


if __name__ == "__main__":
    unittest.main()
