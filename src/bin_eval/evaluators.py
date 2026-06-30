"""Binary evaluator agents: evaluate each question against (x, y).

For each binary question, calls the evaluator on (source_document, extraction_output, q_i)
returning verdict (0/1) + explanation + evidence.
Built as ADK agents for concurrent execution.
"""

from __future__ import annotations

import json

from google.adk.agents import Agent

from bin_eval.llm import call_llm_sync, get_adk_model
from bin_eval.prompts import EVALUATOR_SYSTEM, EVALUATOR_USER
from bin_eval.schemas import BinaryQuestion, Evaluation


def build_evaluator_agent(question_id: str) -> Agent:
    """Build an ADK agent for evaluating a single binary question.

    Args:
        question_id: The ID of the question this agent evaluates.
    """
    return Agent(
        name=f"evaluator_{question_id}",
        model=get_adk_model(),
        instruction=EVALUATOR_SYSTEM,
        description=f"Evaluates binary question {question_id} against source and output.",
    )


def evaluate_question(
    source_document: str,
    extraction_output: str,
    question: BinaryQuestion,
) -> Evaluation:
    """Evaluate a single binary question against source and extraction output.

    Args:
        source_document: The original source document (x).
        extraction_output: The extraction result as string (y).
        question: The binary question to evaluate.

    Returns:
        Evaluation with verdict, explanation, and evidence.
    """
    prompt = EVALUATOR_USER.format(
        source_document=source_document,
        extraction_output=extraction_output,
        question_id=question.id,
        dimension=question.dimension,
        question_text=question.text,
    )

    response = call_llm_sync(prompt=prompt, system=EVALUATOR_SYSTEM)

    # Parse JSON response
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
    except json.JSONDecodeError:
        # Fallback: attempt to determine verdict from text
        verdict = 1 if "pass" in response.lower() or "yes" in response.lower() else 0
        return Evaluation(
            question_id=question.id,
            verdict=verdict,
            explanation=f"Failed to parse structured response: {response[:200]}",
            evidence=[],
        )

    return Evaluation(
        question_id=data.get("question_id", question.id),
        verdict=int(data.get("verdict", 0)),
        explanation=data.get("explanation", ""),
        evidence=data.get("evidence", []),
    )


def evaluate_all_questions(
    source_document: str,
    extraction_output: str,
    questions: list[BinaryQuestion],
) -> list[Evaluation]:
    """Evaluate all binary questions sequentially.

    For ADK parallel execution, use build_evaluator_agent and ParallelAgent.
    This function provides a synchronous fallback.

    Args:
        source_document: The original source document (x).
        extraction_output: The extraction result as string (y).
        questions: List of binary questions to evaluate.

    Returns:
        List of evaluations, one per question.
    """
    evaluations = []
    for question in questions:
        evaluation = evaluate_question(source_document, extraction_output, question)
        evaluations.append(evaluation)
    return evaluations
