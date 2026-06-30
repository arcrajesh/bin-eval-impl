"""Metrics for BinEval evaluation.

Research/benchmark metrics vs human labels:
- Spearman's rho (primary)
- Kendall's tau
- Pearson's r

Product metrics:
- Average overall score
- Average per-dimension score
- Question pass-rate histograms
- Per-dimension failure frequency
- Source/target disagreement rate
- Prompt-update gain per iteration
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from bin_eval.schemas import DocumentEvaluation, Evaluation
from bin_eval.updates import UpdateHistory


@dataclass
class CorrelationMetrics:
    """Research/benchmark correlation metrics against human labels."""

    spearman_rho: float = 0.0
    spearman_pvalue: float = 1.0
    kendall_tau: float = 0.0
    kendall_pvalue: float = 1.0
    pearson_r: float = 0.0
    pearson_pvalue: float = 1.0


@dataclass
class ProductMetrics:
    """Product-level evaluation metrics."""

    average_overall_score: float = 0.0
    average_per_dimension: dict[str, float] = field(default_factory=dict)
    question_pass_rates: dict[str, float] = field(default_factory=dict)
    dimension_failure_frequency: dict[str, float] = field(default_factory=dict)
    disagreement_rate: float = 0.0
    prompt_update_gains: list[float] = field(default_factory=list)


def compute_correlation_metrics(
    system_scores: list[float],
    human_scores: list[float],
) -> CorrelationMetrics:
    """Compute correlation metrics between system scores and human labels.

    Args:
        system_scores: Scores produced by the BinEval system.
        human_scores: Corresponding human judgment scores.

    Returns:
        CorrelationMetrics with Spearman, Kendall, and Pearson correlations.
    """
    if len(system_scores) != len(human_scores):
        raise ValueError(
            f"Score lists must have equal length: {len(system_scores)} vs {len(human_scores)}"
        )

    if len(system_scores) < 2:
        return CorrelationMetrics()

    sys_arr = np.array(system_scores)
    human_arr = np.array(human_scores)

    # Check for zero variance
    if np.std(sys_arr) == 0 or np.std(human_arr) == 0:
        return CorrelationMetrics()

    spearman_result = stats.spearmanr(sys_arr, human_arr)
    kendall_result = stats.kendalltau(sys_arr, human_arr)
    pearson_result = stats.pearsonr(sys_arr, human_arr)

    return CorrelationMetrics(
        spearman_rho=float(spearman_result.statistic),
        spearman_pvalue=float(spearman_result.pvalue),
        kendall_tau=float(kendall_result.statistic),
        kendall_pvalue=float(kendall_result.pvalue),
        pearson_r=float(pearson_result.statistic),
        pearson_pvalue=float(pearson_result.pvalue),
    )


def compute_product_metrics(
    evaluations_list: list[DocumentEvaluation],
    source_evaluations: list[Evaluation] | None = None,
    target_evaluations: list[Evaluation] | None = None,
    update_history: UpdateHistory | None = None,
) -> ProductMetrics:
    """Compute product-level metrics across multiple document evaluations.

    Args:
        evaluations_list: List of DocumentEvaluation results.
        source_evaluations: Optional source evaluator results for disagreement rate.
        target_evaluations: Optional target evaluator results for disagreement rate.
        update_history: Optional prompt update history for gain tracking.

    Returns:
        ProductMetrics with averages, pass rates, and failure frequencies.
    """
    # Average overall score
    if evaluations_list:
        overall_scores = [de.overall_confidence for de in evaluations_list]
        avg_overall = sum(overall_scores) / len(overall_scores)
    else:
        avg_overall = 0.0

    # Average per-dimension scores
    dim_scores_agg: dict[str, list[float]] = {}
    for de in evaluations_list:
        for dim, ds in de.evaluation_result.dimension_scores.items():
            dim_scores_agg.setdefault(dim, []).append(ds.score)

    avg_per_dim = {
        dim: sum(scores) / len(scores) for dim, scores in dim_scores_agg.items()
    }

    # Question pass-rate: per question_id, what fraction of documents passed it
    question_verdicts: dict[str, list[int]] = {}
    for de in evaluations_list:
        for ev in de.evaluation_result.evaluations:
            question_verdicts.setdefault(ev.question_id, []).append(ev.verdict)

    pass_rates = {
        qid: sum(verdicts) / len(verdicts)
        for qid, verdicts in question_verdicts.items()
    }

    # Per-dimension failure frequency: fraction of questions failing per dimension
    dim_failures: dict[str, list[int]] = {}
    for de in evaluations_list:
        for ev in de.evaluation_result.evaluations:
            # Find the question's dimension
            for q in de.evaluation_result.questions:
                if q.id == ev.question_id:
                    dim_failures.setdefault(q.dimension, []).append(1 - ev.verdict)
                    break

    failure_freq = {
        dim: sum(fails) / len(fails) if fails else 0.0
        for dim, fails in dim_failures.items()
    }

    # Disagreement rate
    disagreement_rate = 0.0
    if source_evaluations and target_evaluations:
        source_map = {e.question_id: e.verdict for e in source_evaluations}
        target_map = {e.question_id: e.verdict for e in target_evaluations}
        common_ids = set(source_map.keys()) & set(target_map.keys())
        if common_ids:
            disagreements = sum(
                1 for qid in common_ids if source_map[qid] != target_map[qid]
            )
            disagreement_rate = disagreements / len(common_ids)

    # Prompt-update gains
    gains: list[float] = []
    if update_history and len(update_history.versions) > 1:
        for i in range(1, len(update_history.versions)):
            prev = update_history.versions[i - 1]
            curr = update_history.versions[i]
            gain = curr.score_before - prev.score_before
            gains.append(gain)

    return ProductMetrics(
        average_overall_score=avg_overall,
        average_per_dimension=avg_per_dim,
        question_pass_rates=pass_rates,
        dimension_failure_frequency=failure_freq,
        disagreement_rate=disagreement_rate,
        prompt_update_gains=gains,
    )
