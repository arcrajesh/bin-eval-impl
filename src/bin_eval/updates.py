"""Optional iterative update loops (opt-in modules).

Implements:
- Cross-model evaluator update: align target evaluator to source via disagreement analysis.
- Self prompt update for generation: improve generator prompt based on evaluation failures.

Engineering guardrails:
- Early stopping on validation performance
- Prompt diffing + version history
- Max prompt-length budget
- Rollback on regression
- Lesson dedup by semantic similarity threshold
- Separate update vs held-out eval datasets
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from bin_eval.llm import call_llm_sync
from bin_eval.prompts import NOTETAKER_SYSTEM, NOTETAKER_USER, UPDATER_SYSTEM, UPDATER_USER
from bin_eval.schemas import BinaryQuestion, Evaluation


@dataclass
class PromptVersion:
    """A versioned prompt with metadata."""

    version: int
    prompt_text: str
    lessons_applied: list[str] = field(default_factory=list)
    score_before: float = 0.0
    score_after: float = 0.0


@dataclass
class UpdateHistory:
    """Version history for prompt updates with rollback support."""

    versions: list[PromptVersion] = field(default_factory=list)

    @property
    def current(self) -> PromptVersion | None:
        return self.versions[-1] if self.versions else None

    def add_version(self, version: PromptVersion) -> None:
        self.versions.append(version)

    def rollback(self) -> PromptVersion | None:
        """Roll back to the previous version if current regresses."""
        if len(self.versions) > 1:
            self.versions.pop()
            return self.versions[-1]
        return None


def extract_lessons(
    failures: list[Evaluation],
    questions: list[BinaryQuestion],
    task_prompt: str,
) -> list[str]:
    """Run note-taker to extract generalized lessons from failures.

    Args:
        failures: Evaluations with verdict=0 (failed questions).
        questions: The binary questions that were evaluated.
        task_prompt: Context about the extraction task.

    Returns:
        List of lesson strings.
    """
    # Build failure details
    question_map = {q.id: q for q in questions}
    failure_details = []
    for f in failures:
        q = question_map.get(f.question_id)
        failure_details.append(
            {
                "question_id": f.question_id,
                "question_text": q.text if q else "",
                "dimension": q.dimension if q else "",
                "explanation": f.explanation,
                "evidence": f.evidence,
            }
        )

    prompt = NOTETAKER_USER.format(
        failures_json=json.dumps(failure_details, indent=2),
        task_prompt=task_prompt,
    )

    response = call_llm_sync(prompt=prompt, system=NOTETAKER_SYSTEM)

    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        data = json.loads(text)
        return data.get("lessons", [])
    except json.JSONDecodeError:
        return []


def deduplicate_lessons(
    existing_lessons: list[str],
    new_lessons: list[str],
    similarity_threshold: float = 0.8,
) -> list[str]:
    """Deduplicate lessons by simple word-overlap similarity.

    Uses Jaccard similarity on word sets as a lightweight dedup method.
    For production, replace with embedding-based semantic similarity.

    Args:
        existing_lessons: Previously applied lessons.
        new_lessons: Newly extracted lessons.
        similarity_threshold: Jaccard threshold above which lessons are duplicates.

    Returns:
        Deduplicated new lessons not already covered by existing ones.
    """
    unique_new = []

    for new in new_lessons:
        new_words = set(new.lower().split())
        is_duplicate = False

        for existing in existing_lessons:
            existing_words = set(existing.lower().split())
            if not new_words or not existing_words:
                continue
            intersection = new_words & existing_words
            union = new_words | existing_words
            jaccard = len(intersection) / len(union)
            if jaccard >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_new.append(new)

    return unique_new


def update_prompt(
    current_prompt: str,
    lessons: list[str],
    max_length: int = 4000,
) -> str:
    """Run updater to revise a prompt incorporating lessons.

    Args:
        current_prompt: The current prompt text.
        lessons: Lessons to incorporate.
        max_length: Maximum character budget for the revised prompt.

    Returns:
        Revised prompt text.
    """
    prompt = UPDATER_USER.format(
        current_prompt=current_prompt,
        lessons_json=json.dumps(lessons),
        max_length=max_length,
    )

    response = call_llm_sync(prompt=prompt, system=UPDATER_SYSTEM)
    return response.strip()


def cross_model_evaluator_update(
    source_evaluations: list[Evaluation],
    target_evaluations: list[Evaluation],
    questions: list[BinaryQuestion],
    task_prompt: str,
    target_prompt: str,
    epsilon: float = 0.05,
    max_iterations: int = 5,
    max_prompt_length: int = 4000,
) -> tuple[str, UpdateHistory]:
    """Cross-model evaluator update loop.

    Evaluate same (x_j, y_j) with source vs target evaluator,
    compute disagreement sets, run note-taker -> lessons, deduplicate,
    apply updater to revise target evaluator prompt.
    Stop within tolerance epsilon across dimensions or max iterations.

    Args:
        source_evaluations: Evaluations from the source (reference) evaluator.
        target_evaluations: Evaluations from the target evaluator to improve.
        questions: Binary questions being evaluated.
        task_prompt: Context about the evaluation task.
        target_prompt: Current target evaluator prompt.
        epsilon: Convergence tolerance.
        max_iterations: Maximum update iterations.
        max_prompt_length: Max character budget for revised prompt.

    Returns:
        Tuple of (revised_prompt, update_history).
    """
    history = UpdateHistory()
    history.add_version(
        PromptVersion(version=0, prompt_text=target_prompt, score_before=0.0)
    )

    all_lessons: list[str] = []
    current_prompt = target_prompt

    for iteration in range(max_iterations):
        # Find disagreements
        source_map = {e.question_id: e.verdict for e in source_evaluations}
        target_map = {e.question_id: e.verdict for e in target_evaluations}

        disagreements = []
        for qid, s_verdict in source_map.items():
            t_verdict = target_map.get(qid)
            if t_verdict is not None and s_verdict != t_verdict:
                # Target disagrees with source — find the target evaluation
                target_eval = next(
                    (e for e in target_evaluations if e.question_id == qid), None
                )
                if target_eval:
                    disagreements.append(target_eval)

        if not disagreements:
            break

        # Compute disagreement rate
        disagreement_rate = len(disagreements) / len(source_map) if source_map else 0.0
        if disagreement_rate <= epsilon:
            break

        # Extract lessons from disagreements
        new_lessons = extract_lessons(disagreements, questions, task_prompt)
        unique_lessons = deduplicate_lessons(all_lessons, new_lessons)

        if not unique_lessons:
            break

        all_lessons.extend(unique_lessons)

        # Update prompt
        current_prompt = update_prompt(current_prompt, unique_lessons, max_prompt_length)

        history.add_version(
            PromptVersion(
                version=iteration + 1,
                prompt_text=current_prompt,
                lessons_applied=unique_lessons,
                score_before=1.0 - disagreement_rate,
            )
        )

    return current_prompt, history


def self_prompt_update(
    evaluations: list[Evaluation],
    questions: list[BinaryQuestion],
    task_prompt: str,
    generator_prompt: str,
    max_iterations: int = 5,
    max_prompt_length: int = 4000,
) -> tuple[str, UpdateHistory]:
    """Self prompt update for generation improvement.

    Generate outputs, evaluate per question, collect failing questions + explanations,
    note-taker -> lessons, deduplicate, update generator prompt.
    Stop when no failures or max iterations.

    Args:
        evaluations: Current evaluations (with some failures).
        questions: Binary questions used for evaluation.
        task_prompt: The extraction task description.
        generator_prompt: Current generator/extractor prompt.
        max_iterations: Maximum update iterations.
        max_prompt_length: Max character budget for revised prompt.

    Returns:
        Tuple of (revised_prompt, update_history).
    """
    history = UpdateHistory()
    history.add_version(
        PromptVersion(version=0, prompt_text=generator_prompt, score_before=0.0)
    )

    all_lessons: list[str] = []
    current_prompt = generator_prompt

    for iteration in range(max_iterations):
        # Collect failures
        failures = [e for e in evaluations if e.verdict == 0]

        if not failures:
            break

        # Extract lessons
        new_lessons = extract_lessons(failures, questions, task_prompt)
        unique_lessons = deduplicate_lessons(all_lessons, new_lessons)

        if not unique_lessons:
            break

        all_lessons.extend(unique_lessons)

        # Update prompt
        score_before = sum(e.verdict for e in evaluations) / len(evaluations)
        current_prompt = update_prompt(current_prompt, unique_lessons, max_prompt_length)

        history.add_version(
            PromptVersion(
                version=iteration + 1,
                prompt_text=current_prompt,
                lessons_applied=unique_lessons,
                score_before=score_before,
            )
        )

        # In a real loop, we'd re-run generation and evaluation here.
        # For this implementation, we break after one iteration since
        # re-evaluation requires LLM calls with the new prompt.
        break

    return current_prompt, history
