"""Tests for update loops with mocked LLM responses."""

from unittest.mock import patch

from bin_eval.schemas import BinaryQuestion, Evaluation
from bin_eval.updates import (
    PromptVersion,
    UpdateHistory,
    cross_model_evaluator_update,
    deduplicate_lessons,
    extract_lessons,
    self_prompt_update,
    update_prompt,
)

MOCK_LESSONS_RESPONSE = """{
  "lessons": [
    "Always verify dates against the original document format",
    "Check that currency values include the correct decimal places"
  ]
}"""

MOCK_UPDATED_PROMPT = "You are an improved evaluator. Verify dates against original format. Check currency decimal places."


class TestDeduplicateLessons:
    def test_no_duplicates(self):
        existing = ["Always verify dates"]
        new = ["Check currency formats"]
        result = deduplicate_lessons(existing, new, similarity_threshold=0.8)
        assert len(result) == 1
        assert "currency" in result[0]

    def test_removes_duplicates(self):
        existing = ["Always verify dates against the original document"]
        new = ["Always verify dates against the original document format"]
        result = deduplicate_lessons(existing, new, similarity_threshold=0.7)
        assert len(result) == 0

    def test_empty_lists(self):
        assert deduplicate_lessons([], ["new lesson"]) == ["new lesson"]
        assert deduplicate_lessons(["existing"], []) == []


class TestUpdateHistory:
    def test_add_and_rollback(self):
        history = UpdateHistory()
        v0 = PromptVersion(version=0, prompt_text="original")
        v1 = PromptVersion(version=1, prompt_text="updated")
        history.add_version(v0)
        history.add_version(v1)

        assert history.current.prompt_text == "updated"
        rolled_back = history.rollback()
        assert rolled_back.prompt_text == "original"
        assert history.current.prompt_text == "original"

    def test_rollback_single_version(self):
        history = UpdateHistory()
        history.add_version(PromptVersion(version=0, prompt_text="only"))
        result = history.rollback()
        assert result is None


class TestExtractLessons:
    @patch("bin_eval.updates.call_llm_sync", return_value=MOCK_LESSONS_RESPONSE)
    def test_extract_lessons(self, mock_llm):
        failures = [
            Evaluation(question_id="q_1", verdict=0, explanation="Date wrong", evidence=[]),
        ]
        questions = [
            BinaryQuestion(
                id="q_1", requirement_id="req_1", dimension="accuracy", text="Is date correct?"
            ),
        ]
        lessons = extract_lessons(failures, questions, "Extract invoice")
        assert len(lessons) == 2
        assert "dates" in lessons[0].lower()


class TestUpdatePrompt:
    @patch("bin_eval.updates.call_llm_sync", return_value=MOCK_UPDATED_PROMPT)
    def test_update_prompt(self, mock_llm):
        result = update_prompt(
            current_prompt="You are an evaluator.",
            lessons=["Verify dates", "Check currency"],
            max_length=4000,
        )
        assert "improved" in result.lower() or "evaluator" in result.lower()


class TestCrossModelUpdate:
    @patch("bin_eval.updates.call_llm_sync")
    def test_cross_model_converges(self, mock_llm):
        mock_llm.side_effect = [MOCK_LESSONS_RESPONSE, MOCK_UPDATED_PROMPT]

        source = [
            Evaluation(question_id="q_0", verdict=1, explanation="", evidence=[]),
            Evaluation(question_id="q_1", verdict=0, explanation="", evidence=[]),
        ]
        target = [
            Evaluation(question_id="q_0", verdict=0, explanation="disagree", evidence=[]),
            Evaluation(question_id="q_1", verdict=0, explanation="agree", evidence=[]),
        ]
        questions = [
            BinaryQuestion(id="q_0", requirement_id="r0", dimension="d", text="Q0?"),
            BinaryQuestion(id="q_1", requirement_id="r1", dimension="d", text="Q1?"),
        ]

        prompt, history = cross_model_evaluator_update(
            source_evaluations=source,
            target_evaluations=target,
            questions=questions,
            task_prompt="Test task",
            target_prompt="Original prompt",
            max_iterations=1,
        )
        assert len(history.versions) >= 2
        assert prompt != "Original prompt"


class TestSelfPromptUpdate:
    @patch("bin_eval.updates.call_llm_sync")
    def test_self_update_with_failures(self, mock_llm):
        mock_llm.side_effect = [MOCK_LESSONS_RESPONSE, MOCK_UPDATED_PROMPT]

        evaluations = [
            Evaluation(question_id="q_0", verdict=0, explanation="failed", evidence=[]),
            Evaluation(question_id="q_1", verdict=1, explanation="passed", evidence=[]),
        ]
        questions = [
            BinaryQuestion(id="q_0", requirement_id="r0", dimension="d", text="Q0?"),
            BinaryQuestion(id="q_1", requirement_id="r1", dimension="d", text="Q1?"),
        ]

        prompt, history = self_prompt_update(
            evaluations=evaluations,
            questions=questions,
            task_prompt="Extract",
            generator_prompt="Original",
            max_iterations=3,
        )
        assert len(history.versions) >= 2

    @patch("bin_eval.updates.call_llm_sync")
    def test_self_update_no_failures(self, mock_llm):
        evaluations = [
            Evaluation(question_id="q_0", verdict=1, explanation="ok", evidence=[]),
        ]
        questions = [
            BinaryQuestion(id="q_0", requirement_id="r0", dimension="d", text="Q?"),
        ]

        prompt, history = self_prompt_update(
            evaluations=evaluations,
            questions=questions,
            task_prompt="Extract",
            generator_prompt="Original",
        )
        # No failures -> no update needed
        assert prompt == "Original"
        assert len(history.versions) == 1
