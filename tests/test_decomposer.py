"""Tests for decomposer with mocked LLM responses.

Verifies decomposition produces atomic non-redundant questions.
"""

from unittest.mock import patch

from bin_eval.decomposer import decompose

MOCK_DECOMPOSE_RESPONSE = """{
  "requirements": [
    {"id": "req_1", "text": "Invoice number must be extracted", "dimension": "completeness"},
    {"id": "req_2", "text": "Invoice number must match source", "dimension": "factual_support"},
    {"id": "req_3", "text": "Date format must be consistent", "dimension": "formatting_compliance"}
  ],
  "questions": [
    {"id": "q_1", "requirement_id": "req_1", "dimension": "completeness", "text": "Is the invoice_number field present in the extraction output?", "violation_example": "Field is missing from output"},
    {"id": "q_2", "requirement_id": "req_2", "dimension": "factual_support", "text": "Does the extracted invoice_number match the source document?", "violation_example": "Source has INV-001 but extracted INV-002"},
    {"id": "q_3", "requirement_id": "req_3", "dimension": "formatting_compliance", "text": "Is the date extracted in a standard format?", "violation_example": "Date is 'March 15' instead of '2024-03-15'"}
  ]
}"""


class TestDecomposer:
    @patch("bin_eval.decomposer.call_llm_sync", return_value=MOCK_DECOMPOSE_RESPONSE)
    def test_decompose_returns_requirements_and_questions(self, mock_llm):
        requirements, questions = decompose(
            task_prompt="Extract invoice fields",
            extraction_schema='{"sections": [{"name": "metadata", "fields": []}]}',
        )
        assert len(requirements) == 3
        assert len(questions) == 3

    @patch("bin_eval.decomposer.call_llm_sync", return_value=MOCK_DECOMPOSE_RESPONSE)
    def test_questions_are_atomic(self, mock_llm):
        """Each question tests one property (no merged criteria)."""
        _, questions = decompose(
            task_prompt="Extract invoice",
            extraction_schema="{}",
        )
        for q in questions:
            # Atomic: single question mark, no 'and' joining separate criteria
            assert q.text.count("?") == 1
            # Each belongs to exactly one dimension
            assert q.dimension in [
                "completeness",
                "factual_support",
                "formatting_compliance",
            ]

    @patch("bin_eval.decomposer.call_llm_sync", return_value=MOCK_DECOMPOSE_RESPONSE)
    def test_questions_have_violation_examples(self, mock_llm):
        _, questions = decompose(
            task_prompt="Extract",
            extraction_schema="{}",
        )
        for q in questions:
            assert q.violation_example != ""

    @patch("bin_eval.decomposer.call_llm_sync", return_value=MOCK_DECOMPOSE_RESPONSE)
    def test_no_duplicate_questions(self, mock_llm):
        _, questions = decompose(
            task_prompt="Extract",
            extraction_schema="{}",
        )
        question_texts = [q.text for q in questions]
        assert len(question_texts) == len(set(question_texts))

    @patch("bin_eval.decomposer.call_llm_sync", return_value="not json")
    def test_decompose_handles_invalid_json(self, mock_llm):
        requirements, questions = decompose(
            task_prompt="Extract",
            extraction_schema="{}",
        )
        assert requirements == []
        assert questions == []

    @patch(
        "bin_eval.decomposer.call_llm_sync",
        return_value='''```json
{
  "requirements": [
    {"id": "req_1", "text": "Invoice number must be extracted", "dimension": "completeness"}
  ],
  "questions": [
    {"id": "q_1", "requirement_id": "req_1", "dimension": "completeness", "text": "Is the invoice_number field present in the extraction output?", "violation_example": "Field is missing from output"}
  ]
```''',
    )
    def test_decompose_handles_fenced_json(self, mock_llm):
        requirements, questions = decompose(
            task_prompt="Extract invoice fields",
            extraction_schema="{}",
        )
        assert len(requirements) == 1
        assert len(questions) == 1
