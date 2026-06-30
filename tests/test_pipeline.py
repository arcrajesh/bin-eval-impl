"""Tests for pipeline aggregation logic.

Verifies:
- Dimension scores equal arithmetic means of verdicts
- Overall score equals mean of all verdicts
- Affine rescaling is correct
- Audit trail links scores to questions/explanations
- Field and section scores computed correctly
"""


from bin_eval.pipeline import (
    compute_dimension_balanced_score,
    compute_dimension_scores,
    compute_field_scores,
    compute_overall_score,
    compute_rescaled_scores,
    compute_section_scores,
    rescale_score,
)
from bin_eval.schemas import (
    BinaryQuestion,
    DimensionScore,
    Evaluation,
    ExtractedDocument,
    ExtractedField,
    ExtractedSection,
    FieldScore,
)


def make_questions(dimensions_verdicts: dict[str, list[int]]) -> tuple[
    list[BinaryQuestion], list[Evaluation]
]:
    """Helper to create questions and evaluations from dimension->verdicts mapping."""
    questions = []
    evaluations = []
    q_idx = 0

    for dim, verdicts in dimensions_verdicts.items():
        for verdict in verdicts:
            qid = f"q_{q_idx}"
            questions.append(
                BinaryQuestion(
                    id=qid,
                    requirement_id=f"req_{q_idx}",
                    dimension=dim,
                    text=f"Question {q_idx} for {dim}",
                )
            )
            evaluations.append(
                Evaluation(
                    question_id=qid,
                    verdict=verdict,
                    explanation=f"Verdict {verdict} for {dim}",
                    evidence=[],
                )
            )
            q_idx += 1

    return questions, evaluations


class TestDimensionScores:
    def test_single_dimension_all_pass(self):
        questions, evaluations = make_questions({"accuracy": [1, 1, 1]})
        scores = compute_dimension_scores(questions, evaluations)
        assert scores["accuracy"].score == 1.0
        assert scores["accuracy"].num_passed == 3

    def test_single_dimension_mixed(self):
        questions, evaluations = make_questions({"accuracy": [1, 0, 1, 0]})
        scores = compute_dimension_scores(questions, evaluations)
        assert scores["accuracy"].score == 0.5
        assert scores["accuracy"].num_passed == 2
        assert scores["accuracy"].num_questions == 4

    def test_multiple_dimensions(self):
        questions, evaluations = make_questions({
            "factual_support": [1, 1, 0],  # 2/3
            "completeness": [1, 1, 1, 1],  # 4/4
            "formatting": [0, 0],  # 0/2
        })
        scores = compute_dimension_scores(questions, evaluations)
        assert abs(scores["factual_support"].score - 2 / 3) < 1e-10
        assert scores["completeness"].score == 1.0
        assert scores["formatting"].score == 0.0

    def test_dimension_score_is_arithmetic_mean(self):
        """Paper formula: S_d = (1/|Q_d|) * sum(verdicts in d)."""
        verdicts = [1, 0, 1, 1, 0]
        questions, evaluations = make_questions({"test_dim": verdicts})
        scores = compute_dimension_scores(questions, evaluations)
        expected = sum(verdicts) / len(verdicts)
        assert abs(scores["test_dim"].score - expected) < 1e-10


class TestOverallScore:
    def test_all_pass(self):
        evaluations = [
            Evaluation(question_id=f"q_{i}", verdict=1, explanation="", evidence=[])
            for i in range(5)
        ]
        assert compute_overall_score(evaluations) == 1.0

    def test_all_fail(self):
        evaluations = [
            Evaluation(question_id=f"q_{i}", verdict=0, explanation="", evidence=[])
            for i in range(5)
        ]
        assert compute_overall_score(evaluations) == 0.0

    def test_mixed_is_arithmetic_mean(self):
        """Paper formula: S = (1/N) * sum(all verdicts)."""
        verdicts = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        evaluations = [
            Evaluation(question_id=f"q_{i}", verdict=v, explanation="", evidence=[])
            for i, v in enumerate(verdicts)
        ]
        expected = sum(verdicts) / len(verdicts)
        assert abs(compute_overall_score(evaluations) - expected) < 1e-10

    def test_empty_evaluations(self):
        assert compute_overall_score([]) == 0.0


class TestDimensionBalancedScore:
    def test_balanced_mean(self):
        scores = {
            "dim_a": DimensionScore(dimension="dim_a", score=0.8, num_questions=10, num_passed=8),
            "dim_b": DimensionScore(dimension="dim_b", score=0.4, num_questions=2, num_passed=1),
        }
        balanced = compute_dimension_balanced_score(scores)
        # Mean of dimension scores regardless of question count
        assert abs(balanced - 0.6) < 1e-10

    def test_empty(self):
        assert compute_dimension_balanced_score({}) == 0.0


class TestAffineRescaling:
    def test_rescale_to_1_5(self):
        """S' = S*(b-a) + a with a=1, b=5."""
        assert abs(rescale_score(0.0, 1.0, 5.0) - 1.0) < 1e-10
        assert abs(rescale_score(1.0, 1.0, 5.0) - 5.0) < 1e-10
        assert abs(rescale_score(0.5, 1.0, 5.0) - 3.0) < 1e-10

    def test_rescale_to_0_100(self):
        """S' = S*(100-0) + 0."""
        assert abs(rescale_score(0.0, 0.0, 100.0) - 0.0) < 1e-10
        assert abs(rescale_score(1.0, 0.0, 100.0) - 100.0) < 1e-10
        assert abs(rescale_score(0.75, 0.0, 100.0) - 75.0) < 1e-10

    def test_rescale_identity(self):
        """Rescaling to [0,1] should be identity."""
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert abs(rescale_score(score, 0.0, 1.0) - score) < 1e-10

    def test_rescaled_scores_dict(self):
        dim_scores = {
            "accuracy": DimensionScore(
                dimension="accuracy", score=0.8, num_questions=5, num_passed=4
            ),
        }
        rescaled = compute_rescaled_scores(dim_scores, scale_min=1.0, scale_max=5.0)
        assert "accuracy" in rescaled
        assert abs(rescaled["accuracy"].rescaled_score - 4.2) < 1e-10
        assert rescaled["accuracy"].original_score == 0.8


class TestFieldScores:
    def test_field_score_computation(self):
        doc = ExtractedDocument(
            document_id="test",
            sections=[
                ExtractedSection(
                    name="metadata",
                    fields=[
                        ExtractedField(name="invoice_number", value="INV-001"),
                    ],
                )
            ],
        )
        questions = [
            BinaryQuestion(
                id="q_0",
                requirement_id="req_0",
                dimension="metadata",
                text="Is the invoice_number correctly extracted?",
            ),
            BinaryQuestion(
                id="q_1",
                requirement_id="req_1",
                dimension="metadata",
                text="Is the invoice_number format valid?",
            ),
        ]
        evaluations = [
            Evaluation(question_id="q_0", verdict=1, explanation="correct", evidence=[]),
            Evaluation(question_id="q_1", verdict=0, explanation="wrong format", evidence=[]),
        ]

        field_scores = compute_field_scores(questions, evaluations, doc)
        assert len(field_scores) == 1
        # invoice_number matched both questions
        assert field_scores[0].field_name == "invoice_number"
        assert abs(field_scores[0].score - 0.5) < 1e-10


class TestSectionScores:
    def test_section_score_is_mean_of_fields(self):
        field_scores = [
            FieldScore(
                field_name="f1", section_name="sec_a", score=0.8, question_ids=[], verdicts=[]
            ),
            FieldScore(
                field_name="f2", section_name="sec_a", score=0.6, question_ids=[], verdicts=[]
            ),
            FieldScore(
                field_name="f3", section_name="sec_b", score=1.0, question_ids=[], verdicts=[]
            ),
        ]
        section_scores = compute_section_scores(field_scores)
        sec_a = next(s for s in section_scores if s.section_name == "sec_a")
        sec_b = next(s for s in section_scores if s.section_name == "sec_b")
        assert abs(sec_a.score - 0.7) < 1e-10
        assert sec_b.score == 1.0


class TestAuditTrail:
    def test_scores_link_to_questions(self):
        """Verify audit trail: scores can be reconstructed from question-level outcomes."""
        questions, evaluations = make_questions({
            "dim_a": [1, 0, 1],
            "dim_b": [1, 1],
        })
        dim_scores = compute_dimension_scores(questions, evaluations)
        overall = compute_overall_score(evaluations)

        # Reconstruct overall from individual verdicts
        all_verdicts = [e.verdict for e in evaluations]
        assert abs(overall - sum(all_verdicts) / len(all_verdicts)) < 1e-10

        # Reconstruct dimension scores
        for dim, ds in dim_scores.items():
            dim_evals = [
                e
                for e, q in zip(evaluations, questions)
                if q.dimension == dim
            ]
            expected = sum(e.verdict for e in dim_evals) / len(dim_evals)
            assert abs(ds.score - expected) < 1e-10

        # Each evaluation has explanation linking to the question
        for e in evaluations:
            assert e.explanation != ""
            matching_q = next(q for q in questions if q.id == e.question_id)
            assert matching_q is not None
