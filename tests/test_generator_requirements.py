import unittest
from unittest.mock import patch

import ai_tutor
import generator_long
import generator_short
from ai_tutor import AnswerRequirementEngine, AnswerPlanner


class GeneratorRequirementIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnswerRequirementEngine()
        self.planner = AnswerPlanner()

    def _planner_context(self, question, history=""):
        requirements = self.engine.resolve(question, history=history)
        return self.planner.run(
            question,
            answer_type=requirements["answer_type"],
            results=[],
            requirements_context=requirements,
        )

    def test_requirement_instruction_for_eight_request_types(self):
        cases = [
            ("What is Hadoop?", "definition"),
            ("Explain Hadoop architecture.", "architecture"),
            ("How does HDFS work?", "how"),
            ("Differentiate between HDFS and HBase.", "comparison"),
            ("Explain Lamport's algorithm.", "algorithm"),
            ("Explain Hadoop architecture for 13 marks.", "13"),
            ("Give 2 marks answer for HDFS.", "2"),
            (
                "Explain it simply with an example.",
                "simple",
            ),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                history = (
                    "Student: What is Lamport's algorithm?\nAI Tutor: ..."
                    if "it simply" in question
                    else ""
                )
                context = self._planner_context(question, history)
                instruction = generator_long._requirements_instruction(context)
                self.assertIn("ANSWER PRESENTATION REQUIREMENTS", instruction)
                self.assertIn(expected, instruction)
                self.assertIn("not factual evidence", instruction)
                self.assertTrue(
                    "unsupported" in instruction or "when supported" in instruction
                )
                self.assertNotIn("Lamport's algorithm is", instruction)

    def test_requirements_are_passed_to_long_generator_through_answer_agent(self):
        context = self._planner_context("Explain Lamport's algorithm.")
        results = [{"text": "Lamport's algorithm orders events using logical clocks."}]
        with patch.object(ai_tutor, "generate_long_answer", return_value="A structured response.") as gen:
            answer = ai_tutor.AnswerAgent().run(
                "Explain Lamport's algorithm.",
                results,
                answer_type="medium",
                answer_plan=context,
            )

        self.assertEqual(answer, "A structured response.")
        self.assertEqual(gen.call_args.kwargs["requirements_context"], context)

    def test_requirements_are_passed_to_short_generator_through_answer_agent(self):
        context = self._planner_context("Give 2 marks answer for HDFS.")
        results = [{"text": "HDFS stores files across a distributed cluster."}]
        with patch.object(ai_tutor, "generate_short_answer", return_value="HDFS is distributed storage.") as gen:
            answer = ai_tutor.AnswerAgent().run(
                "Give 2 marks answer for HDFS.",
                results,
                answer_type="short",
                answer_plan=context,
            )

        self.assertEqual(answer, "HDFS is distributed storage.")
        self.assertEqual(gen.call_args.kwargs["requirements_context"], context)

    def test_generators_leave_prompt_unchanged_when_context_is_absent(self):
        long_prompt = generator_long._build_user_prompt(
            "What is Hadoop?", "Hadoop evidence.", "definition"
        )
        self.assertNotIn("ANSWER PRESENTATION REQUIREMENTS", long_prompt)
        self.assertEqual(generator_long._requirements_instruction(None), "")
        self.assertEqual(generator_short._requirements_instruction(None), "")

        with patch.object(generator_short, "generate_draft", return_value="draft") as draft:
            generator_short._call_generator("What is Hadoop?", "Hadoop evidence.", "definition")

        self.assertNotIn("ANSWER PRESENTATION REQUIREMENTS", draft.call_args.kwargs["user_prompt"])

    def test_optional_content_is_conditioned_on_course_support(self):
        requirements = self.engine.resolve("Explain Lamport's algorithm.")
        requirements.update({
            "include_example": True,
            "include_pseudocode": True,
            "include_complexity": True,
        })
        context = self.planner.run(
            "Explain Lamport's algorithm.",
            answer_type=requirements["answer_type"],
            results=[],
            requirements_context=requirements,
        )
        instruction = generator_short._requirements_instruction(context)
        self.assertIn("example", instruction)
        self.assertIn("pseudocode or flow", instruction)
        self.assertIn("complexity details", instruction)
        self.assertIn("only when supported by course evidence", instruction)

    def test_hadoop_definition_fallback_prefers_one_definition_and_relevant_support(self):
        evidence = [
            {"title": "Hadoop overview", "source_url": "https://course.test/hadoop", "text": (
                "Hadoop is a framework for distributed storage and processing. "
                "It stores large datasets across clusters of commodity computers. "
                "Hadoop processes data in parallel across those machines."
            )},
            {"title": "Unrelated", "source_url": "https://course.test/other", "text": (
                "A database table stores rows and columns. Hadoop is sometimes compared with databases."
            )},
        ]
        with patch.object(generator_short, "_call_generator", return_value=""), \
             patch.object(generator_short, "_verify_answer", return_value=""):
            answer = generator_short.generate_short_answer("What is Hadoop?", evidence)
        self.assertIn("Hadoop is a framework", answer)
        self.assertIn("It stores large datasets", answer)
        self.assertNotIn("database table", answer)
        self.assertLessEqual(len([s for s in answer.split(".") if s.strip()]), 3)

    def test_hdfs_fallback_excludes_unrelated_chunks(self):
        evidence = [
            {"title": "HDFS", "text": "HDFS stores files across a distributed cluster. Files are split into blocks and distributed across machines."},
            {"title": "Lamport", "text": "Lamport's algorithm orders distributed events using logical clocks."},
        ]
        answer = generator_short._fallback_answer("How does HDFS work?", evidence)
        self.assertIn("HDFS stores files", answer)
        self.assertNotIn("Lamport", answer)

    def test_two_mark_fallback_is_limited_to_relevant_supported_points(self):
        evidence = [{"text": (
            "HDFS is a distributed file system. It stores files across multiple machines. "
            "Lamport's algorithm orders events using logical clocks."
        )}]
        requirements = {"exam_marks": 2}
        answer = generator_short._fallback_answer(
            "Give 2 marks answer for HDFS.", evidence, requirements
        )
        self.assertLessEqual(len([s for s in answer.split(".") if s.strip()]), 2)
        self.assertIn("HDFS", answer)
        self.assertNotIn("Lamport", answer)

    def test_two_mark_instruction_requires_definition_and_one_supported_point(self):
        requirements = self.engine.resolve("Give 2 marks answer for HDFS.")
        instruction = generator_short._requirements_instruction(requirements)
        self.assertIn("at most two concise sentences", instruction)
        self.assertIn("one distinct, relevant supporting point", instruction)
        self.assertIn("Do not restate the first claim", instruction)
        self.assertIn("if the evidence supports only one point", instruction)

    def test_architecture_instructions_align_with_normal_and_exam_detail(self):
        normal = self._planner_context("Explain Hadoop architecture.")
        exam = self._planner_context("Explain Hadoop architecture for 13 marks.")
        normal_instruction = generator_long._requirements_instruction(normal)
        exam_instruction = generator_long._requirements_instruction(exam)

        self.assertIn("combine each core component with its role", normal_instruction)
        self.assertIn("Keep the answer moderately detailed", normal_instruction)
        self.assertIn("omit peripheral ecosystem tools", normal_instruction)
        self.assertIn("components_and_roles", normal["plan"])
        self.assertIn("complete planned structure", exam_instruction)
        self.assertIn("components", exam["plan"])
        self.assertIn("roles", exam["plan"])

    def test_two_mark_long_generator_instruction_uses_distinct_supported_point(self):
        requirements = self.engine.resolve("Give 2 marks answer for HDFS.")
        instruction = generator_long._requirements_instruction(requirements)
        self.assertIn("at most two concise sentences", instruction)
        self.assertIn("one distinct, relevant supporting point", instruction)
        self.assertIn("Do not restate the first claim", instruction)

    def test_two_mark_repeated_hdfs_replication_claim_is_rejected(self):
        evidence = [{"text": (
            "HDFS is a distributed file system designed to run on commodity hardware. "
            "When a file is saved, it is split into blocks. Data stored in HDFS is "
            "automatically replicated across different nodes to provide fault tolerance."
        )}]
        repeated = (
            "HDFS is a distributed file system that stores files in blocks and "
            "replicates them across nodes to ensure fault tolerance. "
            "HDFS replicates data blocks across different nodes, which provides fault tolerance."
        )
        distinct = (
            "HDFS is a distributed file system that stores files in blocks. "
            "Its data is replicated across different nodes to provide fault tolerance."
        )
        one_point = "HDFS is a distributed file system that stores files in blocks."
        requirements = self.engine.resolve("Give 2 marks answer for HDFS.")

        self.assertFalse(generator_short._is_valid_answer(
            "Give 2 marks answer for HDFS.", repeated, evidence, requirements
        ))
        self.assertTrue(generator_short._is_valid_answer(
            "Give 2 marks answer for HDFS.", distinct, evidence, requirements
        ))
        self.assertTrue(generator_short._is_valid_answer(
            "Give 2 marks answer for HDFS.", one_point, evidence, requirements
        ))

    def test_hdfs_data_locality_instruction_preserves_actor_attribution(self):
        requirements = self.engine.resolve("How does HDFS work?")
        context = self.planner.run(
            "How does HDFS work?",
            answer_type=requirements["answer_type"],
            results=[],
            requirements_context=requirements,
        )
        instruction = generator_long._requirements_instruction(context)
        self.assertIn("Preserve the actor and action stated in the evidence", instruction)
        self.assertIn("MapReduce schedules mappers near HDFS blocks", instruction)
        self.assertIn("do not attribute that scheduling action to HDFS alone", instruction)

    def test_incomplete_architecture_candidate_is_rejected_even_with_grounding(self):
        answer = (
            "Hadoop architecture combines storage and processing. HDFS stores files "
            "across DataNodes while the NameNode manages metadata and block locations. "
            "MapReduce processes input in parallel, and YARN allocates cluster resources. "
            "The components communicate so data can be stored and processed across a "
            "cluster. Their roles support distributed execution and resilience. The "
            "components work together to execute"
        )
        results = [{"text": (
            "Hadoop architecture uses HDFS, DataNodes, a NameNode, MapReduce, and YARN "
            "to store and process data across a cluster."
        )}]
        with patch.object(generator_long, "_source_grounding_score", return_value=1.0):
            self.assertFalse(generator_long._is_usable_answer(
                "Explain Hadoop architecture.", answer, "architecture", results
            ))

    def test_incomplete_draft_with_verifier_failure_uses_bounded_retry(self):
        incomplete = (
            "Hadoop architecture combines storage and processing. HDFS stores files "
            "across DataNodes while the NameNode manages metadata and block locations. "
            "MapReduce processes input in parallel, and YARN allocates cluster resources. "
            "The components communicate so data can be stored and processed across a "
            "cluster. Their roles support distributed execution and resilience. The "
            "components work together to execute"
        )
        results = [{"text": incomplete}]
        with (
            patch.object(generator_long, "_call_generator", return_value=incomplete) as draft,
            patch.object(generator_long, "_verify_answer", return_value=""),
            patch.object(generator_long, "_source_grounding_score", return_value=1.0),
        ):
            generated = generator_long.generate_long_answer(
                "Explain Hadoop architecture.", results
            )

        self.assertEqual(generated, "")
        self.assertEqual(draft.call_count, 2)

    def test_complete_answer_is_accepted_after_verifier_failure(self):
        answer = (
            "Hadoop architecture organizes storage and processing across a cluster. "
            "HDFS stores files as blocks on DataNodes, while the NameNode maintains "
            "the file system namespace and block metadata. MapReduce divides work "
            "into map and reduce tasks that process data in parallel. YARN manages "
            "cluster resources and schedules application tasks. These components "
            "have distinct responsibilities and work together: HDFS provides the "
            "stored data, MapReduce performs computation, and YARN coordinates "
            "resources for that work. This keeps the architecture distributed while "
            "allowing storage and processing to cooperate across the cluster."
        )
        results = [{"text": answer}]
        with (
            patch.object(generator_long, "_call_generator", return_value=answer) as draft,
            patch.object(generator_long, "_verify_answer", return_value=""),
            patch.object(generator_long, "_source_grounding_score", return_value=1.0),
        ):
            generated = generator_long.generate_long_answer(
                "Explain Hadoop architecture.", results
            )

        self.assertEqual(generated, answer)
        self.assertEqual(draft.call_count, 1)

    def test_blank_lines_between_ordered_steps_remain_one_frontend_list(self):
        from pathlib import Path

        source = Path("static/js/tutor.js").read_text(encoding="utf-8")
        list_renderer = source.split("function renderMarkdownList", 1)[1].split("function isAsciiBoxLine", 1)[0]
        self.assertIn("if(!lines[index].trim())", list_renderer)
        self.assertIn("nextIndent===baseIndent", list_renderer)
        self.assertIn("===ordered", list_renderer)
        self.assertIn("function renderCodeBlock", source)


if __name__ == "__main__":
    unittest.main()
