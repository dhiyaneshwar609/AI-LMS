import unittest
from unittest.mock import Mock

import ai_tutor
from ai_tutor import AnswerRequirementEngine
from multi_question import decompose_questions


class AnswerRequirementEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnswerRequirementEngine()

    def test_definition_question_is_concise_and_direct(self):
        requirements = self.engine.resolve("What is Hadoop?")

        self.assertEqual(requirements["intent"], "definition")
        self.assertEqual(requirements["detail_level"], "concise")
        self.assertEqual(requirements["topic"], "Hadoop")
        self.assertIn("direct_definition", requirements["required_sections"])

    def test_architecture_question_requests_components_and_relationships(self):
        requirements = self.engine.resolve("Explain Hadoop architecture.")

        self.assertEqual(requirements["intent"], "architecture")
        self.assertEqual(requirements["detail_level"], "structured")
        self.assertEqual(requirements["topic"], "Hadoop architecture")
        self.assertIn("components_and_roles", requirements["required_sections"])
        self.assertIn("core_relationships", requirements["required_sections"])
        self.assertNotIn("roles", requirements["required_sections"])

    def test_algorithm_definition_has_ordered_steps(self):
        requirements = self.engine.resolve("What is Lamport's algorithm?")

        self.assertEqual(requirements["intent"], "algorithm")
        self.assertEqual(requirements["answer_format"], "algorithm")
        self.assertEqual(requirements["topic"], "Lamport's algorithm")
        self.assertIn("definition_or_purpose", requirements["required_sections"])
        self.assertIn("ordered_steps", requirements["required_sections"])

    def test_marked_algorithm_answer_is_detailed_and_has_conclusion(self):
        requirements = self.engine.resolve("Explain Lamport's algorithm for 13 marks.")

        self.assertEqual(requirements["intent"], "algorithm")
        self.assertEqual(requirements["answer_type"], "detailed")
        self.assertEqual(requirements["exam_marks"], 13)
        self.assertEqual(requirements["topic"], "Lamport's algorithm")
        self.assertIn("supported_working_conditions", requirements["required_sections"])
        self.assertIn("conclusion", requirements["required_sections"])

    def test_marked_architecture_answer_requests_supported_diagram_and_conclusion(self):
        requirements = self.engine.resolve("Explain Hadoop architecture for 13 marks.")

        self.assertEqual(requirements["exam_marks"], 13)
        self.assertEqual(requirements["topic"], "Hadoop architecture")
        self.assertTrue(requirements["include_diagram"])
        self.assertIn("conclusion", requirements["required_sections"])

    def test_comparison_requests_a_table(self):
        requirements = self.engine.resolve("Differentiate HDFS and HBase.")

        self.assertEqual(requirements["intent"], "comparison")
        self.assertEqual(requirements["answer_format"], "comparison")
        self.assertEqual(requirements["topic"], "HDFS and HBase")
        self.assertIn("comparison_table", requirements["required_sections"])

    def test_two_mark_request_is_short_exam_style(self):
        requirements = self.engine.resolve("Give 2 marks answer for HDFS.")

        self.assertEqual(requirements["exam_marks"], 2)
        self.assertEqual(requirements["answer_type"], "short")
        self.assertEqual(requirements["detail_level"], "concise_exam")
        self.assertEqual(requirements["answer_format"], "concise_exam")
        self.assertEqual(requirements["topic"], "HDFS")
        self.assertEqual(requirements["required_sections"], ["direct_answer", "key_points"])

    def test_contextual_follow_up_inherits_topic_and_adds_current_preferences(self):
        requirements = self.engine.resolve(
            "Explain it simply with an example.",
            history="Student: What is Lamport's algorithm?\nAI Tutor: ...",
        )

        self.assertEqual(requirements["intent"], "algorithm")
        self.assertEqual(requirements["answer_format"], "algorithm")
        self.assertEqual(requirements["language_style"], "simple")
        self.assertTrue(requirements["include_example"])
        self.assertEqual(requirements["topic"], "Lamport's algorithm")
        self.assertEqual(requirements["provenance"]["intent"], "recent_conversation")
        self.assertEqual(requirements["provenance"]["language_style"], "current_question")

    def test_independent_definition_ignores_previous_comparison(self):
        requirements = self.engine.resolve(
            "What is Hadoop?",
            history=(
                "Student: Differentiate between HDFS and HBase.\n"
                "AI Tutor: HDFS and HBase differ in their roles."
            ),
        )
        self.assertEqual(requirements["intent"], "definition")
        self.assertEqual(requirements["topic"], "Hadoop")
        self.assertIsNone(requirements["exam_marks"])
        self.assertEqual(requirements["answer_format"], "definition")

    def test_independent_two_mark_question_ignores_previous_thirteen_mark_request(self):
        requirements = self.engine.resolve(
            "Give 2 marks answer for HDFS.",
            history=(
                "Student: Explain Hadoop architecture for 13 marks.\n"
                "AI Tutor: A detailed answer."
            ),
        )
        self.assertEqual(requirements["intent"], "general")
        self.assertEqual(requirements["topic"], "HDFS")
        self.assertEqual(requirements["exam_marks"], 2)
        self.assertEqual(requirements["answer_format"], "concise_exam")

    def test_follow_up_uses_latest_newline_delimited_student_turn(self):
        requirements = self.engine.resolve(
            "Explain it simply with an example.",
            history=(
                "Student: Differentiate between HDFS and HBase.\n"
                "AI Tutor: They serve different purposes.\n"
                "Student: What is HDFS?\n"
                "AI Tutor: HDFS is a distributed file system."
            ),
        )
        self.assertEqual(requirements["topic"], "HDFS")
        self.assertEqual(requirements["intent"], "definition")
        self.assertEqual(requirements["answer_format"], "definition")
        self.assertTrue(requirements["include_example"])
        self.assertEqual(requirements["language_style"], "simple")

    def test_same_message_follow_up_uses_immediately_preceding_question(self):
        decomposition = decompose_questions(
            "What is HDFS? Explain it simply with an example."
        )
        follow_up = decomposition["questions"][1]
        requirements = self.engine.resolve(
            follow_up["text"],
            history=f"Student: {follow_up['context_question']}",
        )
        self.assertEqual(requirements["topic"], "HDFS")
        self.assertEqual(requirements["intent"], "definition")
        self.assertTrue(requirements["include_example"])

    def test_current_marks_and_format_override_follow_up_requirements(self):
        requirements = self.engine.resolve(
            "Explain it, but compare it with HBase for 2 marks.",
            history=(
                "Student: Explain Hadoop architecture for 13 marks.\n"
                "AI Tutor: A detailed architecture answer."
            ),
        )
        self.assertEqual(requirements["intent"], "comparison")
        self.assertEqual(requirements["exam_marks"], 2)
        self.assertEqual(requirements["answer_type"], "short")
        self.assertEqual(requirements["answer_format"], "comparison")

    def test_history_normalization_preserves_turn_boundaries(self):
        history = ai_tutor._extract_history_text(
            " Student: Compare HDFS and HBase.  \r\n"
            "AI Tutor: They are different.\r\n"
            "Student: What is HDFS? "
        )
        self.assertEqual(
            history,
            "Student: Compare HDFS and HBase.\n"
            "AI Tutor: They are different.\n"
            "Student: What is HDFS?",
        )
        self.assertEqual(
            AnswerRequirementEngine._student_turns(history),
            ["Compare HDFS and HBase.", "What is HDFS?"],
        )

    def test_ask_observes_requirements_without_changing_planner_plan(self):
        tutor = ai_tutor.AITutor.__new__(ai_tutor.AITutor)
        tutor.answer_requirement_engine = self.engine
        tutor.query_agent = Mock(run=Mock(return_value={
            "intent": "algorithm",
            "answer_type": "medium",
            "retrieval_k": 6,
            "retry_k": 6,
        }))
        tutor.retrieval_agent = Mock(run=Mock(return_value=[{
            "text": "Lamport's algorithm uses ordered steps.",
            "evidence_type": "primary",
        }]))
        tutor.reranking_agent = Mock(run=Mock(side_effect=lambda question, results, top_k: results))
        tutor.validation_agent = Mock(run=Mock(return_value={
            "sufficient": True,
            "reason": "",
            "best_score": 1.0,
            "evidence_count": 1,
        }))
        planner = ai_tutor.AnswerPlanner()
        planner_run = Mock(wraps=planner.run)
        tutor.answer_planner = Mock(run=planner_run)
        tutor.adaptive_learning_agent = Mock()
        tutor.adaptive_learning_agent.run.return_value = {}
        tutor.adaptive_learning_agent.instruction.return_value = ""
        tutor.answer_agent = Mock(run=Mock(return_value="A concise answer."))
        tutor.answer_grounding_agent = Mock(run=Mock(return_value={
            "needs_revision": False,
            "grounded": True,
            "coverage": 1.0,
            "supported_claims": 1,
            "unsupported_claims": 0,
        }))
        tutor.grounded_diagram_agent = Mock(run=Mock(return_value={
            "grounded": False,
            "mermaid": "",
        }))

        with self.assertLogs("ai_tutor", level="INFO") as captured:
            response = tutor.ask("What is Lamport's algorithm?", course_id=4)

        requirements = response["answer_requirements"]
        self.assertEqual(
            planner_run.call_args.kwargs["requirements_context"],
            requirements,
        )
        self.assertEqual(response["answer_plan"]["requirements_context"], requirements)
        self.assertEqual(
            response["answer_plan"]["plan"],
            [
                "purpose_or_concept",
                "ordered_steps",
                "pseudocode_or_flow_when_supported",
                "supported_working_conditions_when_supported",
            ],
        )
        self.assertIn('phase=answer_requirements_resolved', "\n".join(captured.output))
        self.assertIn('phase=answer_planner_requirements', "\n".join(captured.output))
        self.assertIn('influenced_fields', "\n".join(captured.output))
        for field in (
            "question", "intent", "answer_type", "detail_level", "exam_marks",
            "answer_format", "required_sections", "include_example", "include_diagram",
            "include_pseudocode", "include_complexity", "language_style", "topic", "provenance",
        ):
            self.assertIn(f'"{field}"', "\n".join(captured.output))


class AnswerPlannerRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnswerRequirementEngine()
        self.planner = ai_tutor.AnswerPlanner()

    def _plan(self, question, history=""):
        requirements = self.engine.resolve(question, history=history)
        return self.planner.run(
            question=question,
            answer_type=requirements["answer_type"],
            results=[],
            requirements_context=requirements,
        )

    def test_definition_requirement_refines_without_changing_definition_plan(self):
        plan = self._plan("What is Hadoop?")

        self.assertEqual(plan["intent"], "definition")
        self.assertIn("concise_definition_first", plan["plan"])
        self.assertIn("relevant_key_points", plan["plan"])

    def test_architecture_requirement_adds_introduction_and_components(self):
        plan = self._plan("Explain Hadoop architecture.")

        self.assertEqual(plan["intent"], "architecture")
        self.assertIn("introduction", plan["plan"])
        self.assertIn("components_and_roles", plan["plan"])
        self.assertIn("core_relationships", plan["plan"])
        self.assertNotIn("roles", plan["plan"])

    def test_how_requirement_preserves_ordered_working_plan(self):
        plan = self._plan("How does HDFS work?")

        self.assertEqual(plan["intent"], "how")
        self.assertIn("steps_in_correct_sequence", plan["plan"])
        self.assertIn("working_or_flow_when_supported", plan["plan"])

    def test_comparison_requirement_keeps_table_and_key_differences(self):
        plan = self._plan("Differentiate between HDFS and HBase.")

        self.assertEqual(plan["intent"], "comparison")
        self.assertIn("comparison_table_when_appropriate", plan["plan"])
        self.assertIn("key_differences", plan["plan"])

    def test_algorithm_requirements_add_supported_working_and_steps(self):
        plan = self._plan("Explain Lamport's algorithm.")

        self.assertEqual(plan["intent"], "algorithm")
        self.assertIn("purpose_or_concept", plan["plan"])
        self.assertIn("ordered_steps", plan["plan"])
        self.assertIn("supported_working_conditions_when_supported", plan["plan"])
        self.assertIn(
            "optional_if_supported_by_course_evidence",
            [item["status"] for item in plan["requirement_section_status"]],
        )

    def test_thirteen_mark_architecture_adds_detailed_sections_conditionally(self):
        plan = self._plan("Explain Hadoop architecture for 13 marks.")

        self.assertEqual(plan["exam_marks"], 13)
        self.assertEqual(plan["response_depth"], "complete_structured_exam_answer")
        self.assertIn("complete_structured_exam_answer", plan["plan"])
        self.assertIn("conclusion_from_supported_points", plan["plan"])
        self.assertIn("grounded_diagram_when_supported", plan["plan"])

    def test_two_mark_request_is_concise(self):
        plan = self._plan("Give 2 marks answer for HDFS.")

        self.assertEqual(plan["answer_type"], "short")
        self.assertEqual(plan["plan"], [
            "direct_answer",
            "relevant_key_points",
            "concise_exam_oriented_response",
        ])
        self.assertEqual(plan["response_depth"], "concise_exam_oriented_response")

    def test_follow_up_uses_recent_intent_and_current_style_and_example(self):
        plan = self._plan(
            "Explain it simply with an example.",
            history="Student: What is Lamport's algorithm?\nAI Tutor: ...",
        )

        self.assertEqual(plan["intent"], "algorithm")
        self.assertEqual(plan["topic"], "Lamport's algorithm")
        self.assertIn("purpose_or_concept", plan["plan"])
        self.assertIn("simple_language_style", plan["plan"])
        self.assertIn("course_supported_example_when_supported", plan["plan"])

    def test_planner_output_is_unchanged_without_requirements_context(self):
        plan = self.planner.run("Explain Hadoop architecture.", answer_type="medium")

        self.assertEqual(plan, {
            "intent": "architecture",
            "topic": "",
            "answer_type": "medium",
            "response_depth": "moderately_detailed_response",
            "plan": ["components_and_roles", "core_relationships"],
            "evidence_priority": ["primary", "supporting_context"],
        })

    def test_detailed_architecture_plan_keeps_expanded_sections(self):
        plan = self._plan("Explain Hadoop architecture for 13 marks.")

        self.assertIn("components", plan["plan"])
        self.assertIn("roles", plan["plan"])
        self.assertIn("relationships", plan["plan"])
        self.assertEqual(plan["response_depth"], "complete_structured_exam_answer")


if __name__ == "__main__":
    unittest.main()
