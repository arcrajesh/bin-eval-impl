"""Pydantic schemas matching the BinEval paper's output structure.

Covers: requirements, binary questions, evaluations, dimension/overall scores,
and document extraction structures with per-field/section scoring.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Core BinEval schemas (paper-faithful) ---


class Requirement(BaseModel):
    """A single requirement derived from the task prompt."""

    id: str = Field(description="Unique requirement identifier")
    text: str = Field(description="Requirement description")
    dimension: str = Field(description="Quality dimension this requirement belongs to")


class BinaryQuestion(BaseModel):
    """An atomic yes/no question testing one property."""

    id: str = Field(description="Unique question identifier")
    requirement_id: str = Field(description="ID of the parent requirement")
    dimension: str = Field(description="Quality dimension (e.g., factual_support, completeness)")
    text: str = Field(description="The binary question text; 'yes' = compliance")
    violation_example: str = Field(
        default="",
        description="Example of what would constitute a violation",
    )


class Evaluation(BaseModel):
    """Result of evaluating a single binary question against (x, y)."""

    question_id: str = Field(description="ID of the question being evaluated")
    verdict: int = Field(description="0 (fail) or 1 (pass)", ge=0, le=1)
    explanation: str = Field(description="Reasoning for the verdict")
    evidence: list[str] = Field(
        default_factory=list,
        description="Relevant spans from source or output supporting the verdict",
    )


class DimensionScore(BaseModel):
    """Aggregated score for a quality dimension."""

    dimension: str
    score: float = Field(description="Mean of verdicts in [0,1]", ge=0.0, le=1.0)
    num_questions: int = Field(description="Number of questions in this dimension")
    num_passed: int = Field(description="Number of questions with verdict=1")


class RescaledScore(BaseModel):
    """Affine-rescaled score for display purposes."""

    dimension: str
    original_score: float = Field(ge=0.0, le=1.0)
    rescaled_score: float
    scale_min: float
    scale_max: float


class EvaluationResult(BaseModel):
    """Top-level BinEval result with full audit trail."""

    task_prompt: str = Field(description="The original task prompt T")
    requirements: list[Requirement] = Field(default_factory=list)
    questions: list[BinaryQuestion] = Field(default_factory=list)
    evaluations: list[Evaluation] = Field(default_factory=list)
    dimension_scores: dict[str, DimensionScore] = Field(default_factory=dict)
    overall_score: float = Field(
        default=0.0, description="Mean of all verdicts in [0,1]", ge=0.0, le=1.0
    )
    dimension_balanced_score: float = Field(
        default=0.0,
        description="Mean of dimension scores (engineering extension)",
        ge=0.0,
        le=1.0,
    )
    rescaled_scores: dict[str, RescaledScore] = Field(default_factory=dict)


# --- Document extraction schemas ---


class ExtractedField(BaseModel):
    """A single extracted field from a document."""

    name: str = Field(description="Field name/label")
    value: str = Field(description="Extracted value")
    confidence: float = Field(
        default=1.0, description="Per-field confidence score [0,1]", ge=0.0, le=1.0
    )
    source_span: str = Field(
        default="", description="Original text span this was extracted from"
    )


class ExtractedSection(BaseModel):
    """A group of related extracted fields (maps to a dimension)."""

    name: str = Field(description="Section/dimension name")
    fields: list[ExtractedField] = Field(default_factory=list)
    section_score: float = Field(
        default=1.0, description="Mean confidence across fields", ge=0.0, le=1.0
    )


class ExtractedDocument(BaseModel):
    """Complete extraction result with structured fields grouped by section."""

    document_id: str = Field(default="", description="Identifier for the source document")
    sections: list[ExtractedSection] = Field(default_factory=list)
    raw_text: str = Field(default="", description="Original source document text")


class FieldScore(BaseModel):
    """Per-field evaluation score with audit trail."""

    field_name: str
    section_name: str
    score: float = Field(ge=0.0, le=1.0)
    question_ids: list[str] = Field(default_factory=list)
    verdicts: list[int] = Field(default_factory=list)


class SectionScore(BaseModel):
    """Per-section (dimension) evaluation score."""

    section_name: str
    score: float = Field(ge=0.0, le=1.0)
    field_scores: list[FieldScore] = Field(default_factory=list)


class DocumentEvaluation(BaseModel):
    """Final evaluation combining extracted data with BinEval scores.

    Links the complete extracted document with per-field scores,
    per-section (dimension) scores, and an overall confidence.
    """

    extracted_document: ExtractedDocument
    evaluation_result: EvaluationResult
    field_scores: list[FieldScore] = Field(default_factory=list)
    section_scores: list[SectionScore] = Field(default_factory=list)
    overall_confidence: float = Field(
        default=0.0,
        description="Overall document confidence = mean of all verdicts",
        ge=0.0,
        le=1.0,
    )
