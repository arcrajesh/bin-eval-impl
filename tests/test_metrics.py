"""Tests for correlation and product metrics."""

import pytest

from bin_eval.metrics import compute_correlation_metrics, compute_product_metrics
from bin_eval.schemas import (
    DocumentEvaluation,
    Evaluation,
    EvaluationResult,
    ExtractedDocument,
)


class TestCorrelationMetrics:
    def test_perfect_correlation(self):
        system = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        human = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        metrics = compute_correlation_metrics(system, human)
        assert abs(metrics.spearman_rho - 1.0) < 1e-5
        assert abs(metrics.pearson_r - 1.0) < 1e-5
        assert abs(metrics.kendall_tau - 1.0) < 1e-5

    def test_inverse_correlation(self):
        system = [0.1, 0.2, 0.3, 0.4, 0.5]
        human = [0.5, 0.4, 0.3, 0.2, 0.1]
        metrics = compute_correlation_metrics(system, human)
        assert metrics.spearman_rho < -0.9
        assert metrics.pearson_r < -0.9

    def test_empty_scores(self):
        metrics = compute_correlation_metrics([], [])
        assert metrics.spearman_rho == 0.0

    def test_single_score(self):
        metrics = compute_correlation_metrics([0.5], [0.5])
        assert metrics.spearman_rho == 0.0

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_correlation_metrics([0.1, 0.2], [0.1])


class TestProductMetrics:
    def test_average_overall_score(self):
        evals = [
            DocumentEvaluation(
                extracted_document=ExtractedDocument(),
                evaluation_result=EvaluationResult(task_prompt="test", overall_score=s),
                overall_confidence=s,
            )
            for s in [0.8, 0.6, 1.0]
        ]
        metrics = compute_product_metrics(evals)
        assert abs(metrics.average_overall_score - 0.8) < 1e-10

    def test_disagreement_rate(self):
        source = [
            Evaluation(question_id="q_0", verdict=1, explanation="", evidence=[]),
            Evaluation(question_id="q_1", verdict=0, explanation="", evidence=[]),
            Evaluation(question_id="q_2", verdict=1, explanation="", evidence=[]),
        ]
        target = [
            Evaluation(question_id="q_0", verdict=1, explanation="", evidence=[]),
            Evaluation(question_id="q_1", verdict=1, explanation="", evidence=[]),  # disagree
            Evaluation(question_id="q_2", verdict=0, explanation="", evidence=[]),  # disagree
        ]
        metrics = compute_product_metrics(
            [], source_evaluations=source, target_evaluations=target
        )
        # 2 out of 3 disagree
        assert abs(metrics.disagreement_rate - 2 / 3) < 1e-10

    def test_empty_evaluations(self):
        metrics = compute_product_metrics([])
        assert metrics.average_overall_score == 0.0
