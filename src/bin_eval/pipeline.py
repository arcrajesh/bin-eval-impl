"""Pipeline orchestration and score aggregation.

Uses ADK SequentialAgent for ordering: extraction -> decomposition -> evaluation -> aggregation.
Uses ADK ParallelAgent concept for concurrent binary evaluation.

Aggregation formulas (paper-faithful):
- Dimension score: S_d = (1/|Q_d|) * sum(verdicts in dimension d)
- Overall score: S = (1/N) * sum(all verdicts)
- Field score: mean of verdicts for that field's questions
- Section score: mean across field scores
- Document overall_confidence: overall mean of all verdicts
- Optional affine rescaling: S' = S*(b-a) + a
"""

from __future__ import annotations

import json
from collections import defaultdict

from google.adk.agents import SequentialAgent

from bin_eval.decomposer import build_decomposer_agent, decompose
from bin_eval.evaluators import (
    evaluate_all_questions,
)
from bin_eval.extractor import build_extractor_agent, extract_document
from bin_eval.schemas import (
    BinaryQuestion,
    DimensionScore,
    DocumentEvaluation,
    Evaluation,
    EvaluationResult,
    ExtractedDocument,
    FieldScore,
    RescaledScore,
    SectionScore,
)


def compute_dimension_scores(
    questions: list[BinaryQuestion],
    evaluations: list[Evaluation],
) -> dict[str, DimensionScore]:
    """Compute per-dimension scores as mean of verdicts.

    S_d = (1/|Q_d|) * sum(verdicts in dimension d)
    """
    # Map question_id -> evaluation
    eval_map = {e.question_id: e for e in evaluations}

    # Group questions by dimension
    dim_questions: dict[str, list[BinaryQuestion]] = defaultdict(list)
    for q in questions:
        dim_questions[q.dimension].append(q)

    scores = {}
    for dim, qs in dim_questions.items():
        verdicts = [eval_map[q.id].verdict for q in qs if q.id in eval_map]
        num_passed = sum(verdicts)
        num_total = len(verdicts)
        score = num_passed / num_total if num_total > 0 else 0.0
        scores[dim] = DimensionScore(
            dimension=dim,
            score=score,
            num_questions=num_total,
            num_passed=num_passed,
        )

    return scores


def compute_overall_score(evaluations: list[Evaluation]) -> float:
    """Compute overall score as mean of all verdicts.

    S = (1/N) * sum(all verdicts)
    """
    if not evaluations:
        return 0.0
    return sum(e.verdict for e in evaluations) / len(evaluations)


def compute_dimension_balanced_score(
    dimension_scores: dict[str, DimensionScore],
) -> float:
    """Compute dimension-balanced overall as mean of dimension scores.

    Engineering extension: treats all dimensions equally regardless of question count.
    """
    if not dimension_scores:
        return 0.0
    return sum(ds.score for ds in dimension_scores.values()) / len(dimension_scores)


def rescale_score(
    score: float,
    scale_min: float = 0.0,
    scale_max: float = 1.0,
) -> float:
    """Affine rescaling: S' = S*(b-a) + a.

    Applied at OUTPUT layer only; internal logic stays in [0,1].
    """
    return score * (scale_max - scale_min) + scale_min


def compute_rescaled_scores(
    dimension_scores: dict[str, DimensionScore],
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> dict[str, RescaledScore]:
    """Compute affine-rescaled scores for all dimensions."""
    rescaled = {}
    for dim, ds in dimension_scores.items():
        rescaled[dim] = RescaledScore(
            dimension=dim,
            original_score=ds.score,
            rescaled_score=rescale_score(ds.score, scale_min, scale_max),
            scale_min=scale_min,
            scale_max=scale_max,
        )
    return rescaled


def compute_field_scores(
    questions: list[BinaryQuestion],
    evaluations: list[Evaluation],
    extracted_document: ExtractedDocument,
) -> list[FieldScore]:
    """Compute per-field scores by matching questions to fields.

    Field score = mean of verdicts for that field's questions.
    Questions are matched to fields by checking if the field name appears in the question text.
    """
    eval_map = {e.question_id: e for e in evaluations}
    field_scores = []

    for section in extracted_document.sections:
        for field in section.fields:
            # Find questions related to this field
            field_questions = [
                q
                for q in questions
                if field.name.lower() in q.text.lower()
                or field.name.lower() in q.id.lower()
            ]
            if not field_questions:
                # If no specific questions, use section-level questions
                field_questions = [q for q in questions if q.dimension == section.name]

            verdicts = []
            question_ids = []
            for q in field_questions:
                if q.id in eval_map:
                    verdicts.append(eval_map[q.id].verdict)
                    question_ids.append(q.id)

            score = sum(verdicts) / len(verdicts) if verdicts else 1.0
            field_scores.append(
                FieldScore(
                    field_name=field.name,
                    section_name=section.name,
                    score=score,
                    question_ids=question_ids,
                    verdicts=verdicts,
                )
            )

    return field_scores


def compute_section_scores(field_scores: list[FieldScore]) -> list[SectionScore]:
    """Compute per-section scores as mean of field scores within each section."""
    section_fields: dict[str, list[FieldScore]] = defaultdict(list)
    for fs in field_scores:
        section_fields[fs.section_name].append(fs)

    section_scores = []
    for section_name, fields in section_fields.items():
        score = sum(f.score for f in fields) / len(fields) if fields else 0.0
        section_scores.append(
            SectionScore(
                section_name=section_name,
                score=score,
                field_scores=fields,
            )
        )

    return section_scores


def build_pipeline_agents() -> SequentialAgent:
    """Build the full ADK pipeline as a SequentialAgent.

    Order: extraction -> decomposition -> parallel binary evaluation -> aggregation.
    """
    extractor = build_extractor_agent()
    decomposer_agent = build_decomposer_agent()

    # The parallel evaluation agents are built dynamically based on questions
    # This returns the orchestration structure
    return SequentialAgent(
        name="bin_eval_pipeline",
        sub_agents=[extractor, decomposer_agent],
        description="BinEval pipeline: extract -> decompose -> evaluate -> aggregate",
    )


def run_pipeline(
    source_document: str,
    task_prompt: str,
    extraction_schema: str,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> DocumentEvaluation:
    """Run the complete BinEval pipeline synchronously.

    Steps:
    1. Extract structured data from source document.
    2. Decompose task into requirements and binary questions.
    3. Evaluate each question against (source, extraction).
    4. Aggregate scores at field, section, dimension, and overall levels.

    Args:
        source_document: Raw source document text.
        task_prompt: Description of what to extract.
        extraction_schema: JSON schema of fields/sections.
        scale_min: Optional minimum for affine rescaling.
        scale_max: Optional maximum for affine rescaling.

    Returns:
        DocumentEvaluation with complete audit trail.
    """
    # Step 1: Extract
    extracted = extract_document(source_document, extraction_schema)

    # Step 2: Decompose
    requirements, questions = decompose(task_prompt, extraction_schema)

    # Step 3: Evaluate (sequentially; ADK ParallelAgent used in async mode)
    extraction_output = json.dumps(extracted.model_dump(), indent=2)
    evaluations = evaluate_all_questions(source_document, extraction_output, questions)

    # Step 4: Aggregate
    dimension_scores = compute_dimension_scores(questions, evaluations)
    overall_score = compute_overall_score(evaluations)
    balanced_score = compute_dimension_balanced_score(dimension_scores)

    # Optional rescaling
    rescaled_scores: dict[str, RescaledScore] = {}
    if scale_min is not None and scale_max is not None:
        rescaled_scores = compute_rescaled_scores(dimension_scores, scale_min, scale_max)

    # Build evaluation result
    eval_result = EvaluationResult(
        task_prompt=task_prompt,
        requirements=requirements,
        questions=questions,
        evaluations=evaluations,
        dimension_scores=dimension_scores,
        overall_score=overall_score,
        dimension_balanced_score=balanced_score,
        rescaled_scores=rescaled_scores,
    )

    # Compute field and section scores
    field_scores = compute_field_scores(questions, evaluations, extracted)
    section_scores = compute_section_scores(field_scores)

    return DocumentEvaluation(
        extracted_document=extracted,
        evaluation_result=eval_result,
        field_scores=field_scores,
        section_scores=section_scores,
        overall_confidence=overall_score,
    )
