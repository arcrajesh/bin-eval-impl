"""Tests for BinEval schemas validation."""

import pytest

from bin_eval.schemas import (
    BinaryQuestion,
    DimensionScore,
    DocumentEvaluation,
    Evaluation,
    EvaluationResult,
    ExtractedDocument,
    ExtractedField,
    ExtractedSection,
    Requirement,
)


class TestRequirement:
    def test_create_requirement(self):
        req = Requirement(id="req_1", text="Field must be present", dimension="completeness")
        assert req.id == "req_1"
        assert req.dimension == "completeness"

    def test_requirement_serialization(self):
        req = Requirement(id="req_1", text="Test", dimension="accuracy")
        data = req.model_dump()
        assert data["id"] == "req_1"
        assert data["dimension"] == "accuracy"


class TestBinaryQuestion:
    def test_create_question(self):
        q = BinaryQuestion(
            id="q_1",
            requirement_id="req_1",
            dimension="factual_support",
            text="Is the invoice number correctly extracted?",
            violation_example="Invoice number is INV-001 but extracted as INV-002",
        )
        assert q.id == "q_1"
        assert q.dimension == "factual_support"

    def test_question_default_violation(self):
        q = BinaryQuestion(
            id="q_1",
            requirement_id="req_1",
            dimension="completeness",
            text="Is the field present?",
        )
        assert q.violation_example == ""


class TestEvaluation:
    def test_verdict_pass(self):
        ev = Evaluation(
            question_id="q_1",
            verdict=1,
            explanation="Field matches source",
            evidence=["span1"],
        )
        assert ev.verdict == 1

    def test_verdict_fail(self):
        ev = Evaluation(
            question_id="q_1",
            verdict=0,
            explanation="Field does not match",
            evidence=[],
        )
        assert ev.verdict == 0

    def test_verdict_bounds(self):
        with pytest.raises(Exception):
            Evaluation(question_id="q_1", verdict=2, explanation="bad", evidence=[])


class TestDimensionScore:
    def test_score_bounds(self):
        ds = DimensionScore(dimension="accuracy", score=0.75, num_questions=4, num_passed=3)
        assert 0.0 <= ds.score <= 1.0

    def test_score_validation(self):
        with pytest.raises(Exception):
            DimensionScore(dimension="x", score=1.5, num_questions=1, num_passed=1)


class TestExtractedDocument:
    def test_document_structure(self):
        doc = ExtractedDocument(
            document_id="doc_1",
            sections=[
                ExtractedSection(
                    name="metadata",
                    fields=[
                        ExtractedField(
                            name="invoice_number",
                            value="INV-001",
                            source_span="INVOICE #INV-001",
                        )
                    ],
                )
            ],
            raw_text="INVOICE #INV-001",
        )
        assert len(doc.sections) == 1
        assert doc.sections[0].fields[0].value == "INV-001"


class TestDocumentEvaluation:
    def test_full_evaluation(self):
        doc = ExtractedDocument(document_id="doc_1", sections=[])
        eval_result = EvaluationResult(
            task_prompt="Extract invoice",
            overall_score=0.8,
        )
        doc_eval = DocumentEvaluation(
            extracted_document=doc,
            evaluation_result=eval_result,
            overall_confidence=0.8,
        )
        assert doc_eval.overall_confidence == 0.8
