"""Tests for binary evaluator with mocked LLM responses.

Verifies evaluator returns one verdict+explanation per question.
"""

from unittest.mock import patch

from bin_eval.evaluators import evaluate_all_questions, evaluate_question
from bin_eval.schemas import BinaryQuestion

MOCK_EVAL_PASS = """{
  "question_id": "q_1",
  "verdict": 1,
  "explanation": "The invoice number INV-2024-0042 in the extraction matches the source document.",
  "evidence": ["INVOICE #INV-2024-0042"]
}"""

MOCK_EVAL_FAIL = """{
  "question_id": "q_2",
  "verdict": 0,
  "explanation": "The date format does not match the expected ISO format.",
  "evidence": ["Date: March 15, 2024"]
}"""


class TestEvaluateQuestion:
    @patch("bin_eval.evaluators.call_llm_sync", return_value=MOCK_EVAL_PASS)
    def test_evaluate_pass(self, mock_llm):
        q = BinaryQuestion(
            id="q_1",
            requirement_id="req_1",
            dimension="factual_support",
            text="Does the invoice number match?",
        )
        result = evaluate_question("source doc", "extraction", q)
        assert result.verdict == 1
        assert result.question_id == "q_1"
        assert "INV-2024-0042" in result.explanation
        assert len(result.evidence) > 0

    @patch("bin_eval.evaluators.call_llm_sync", return_value=MOCK_EVAL_FAIL)
    def test_evaluate_fail(self, mock_llm):
        q = BinaryQuestion(
            id="q_2",
            requirement_id="req_2",
            dimension="formatting_compliance",
            text="Is the date in ISO format?",
        )
        result = evaluate_question("source doc", "extraction", q)
        assert result.verdict == 0
        assert result.question_id == "q_2"
        assert result.explanation != ""

    @patch("bin_eval.evaluators.call_llm_sync", return_value="invalid json response")
    def test_evaluate_handles_invalid_json(self, mock_llm):
        q = BinaryQuestion(
            id="q_3",
            requirement_id="req_3",
            dimension="completeness",
            text="Is the field present?",
        )
        result = evaluate_question("source", "extraction", q)
        # Should still return an Evaluation with fallback logic
        assert result.question_id == "q_3"
        assert result.verdict in (0, 1)


class TestEvaluateAllQuestions:
    @patch("bin_eval.evaluators.call_llm_sync")
    def test_one_evaluation_per_question(self, mock_llm):
        # Each mock returns the correct question_id for its respective question
        mock_responses = [
            '{"question_id": "q_0", "verdict": 1, "explanation": "pass", "evidence": []}',
            '{"question_id": "q_1", "verdict": 0, "explanation": "fail", "evidence": []}',
            '{"question_id": "q_2", "verdict": 1, "explanation": "pass", "evidence": []}',
        ]
        mock_llm.side_effect = mock_responses

        questions = [
            BinaryQuestion(
                id=f"q_{i}",
                requirement_id=f"req_{i}",
                dimension="test",
                text=f"Question {i}?",
            )
            for i in range(3)
        ]

        results = evaluate_all_questions("source", "extraction", questions)
        assert len(results) == 3
        # Each evaluation maps to its question
        for i, result in enumerate(results):
            assert result.question_id == f"q_{i}"
            assert result.explanation != ""
