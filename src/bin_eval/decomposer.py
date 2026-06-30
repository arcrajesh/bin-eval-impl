"""Decomposer: generates Requirements and BinaryQuestions from a task prompt.

From the task prompt T (per field/section), produces atomic binary questions
grouped into quality dimensions (factual_support, formatting_compliance,
completeness, consistency, relevance).
"""

from __future__ import annotations

import json
import re

from google.adk.agents import Agent

from bin_eval.llm import call_llm_sync, get_adk_model
from bin_eval.prompts import DECOMPOSER_SYSTEM, DECOMPOSER_USER
from bin_eval.schemas import BinaryQuestion, Requirement


def build_decomposer_agent() -> Agent:
    """Build an ADK agent for requirement decomposition."""
    return Agent(
        name="decomposer",
        model=get_adk_model(),
        instruction=DECOMPOSER_SYSTEM,
        description="Decomposes task prompts into atomic binary evaluation questions.",
    )


def decompose(
    task_prompt: str,
    extraction_schema: str,
) -> tuple[list[Requirement], list[BinaryQuestion]]:
    """Decompose a task prompt into requirements and binary questions.

    Args:
        task_prompt: The extraction task description.
        extraction_schema: JSON schema of fields/sections to extract.

    Returns:
        Tuple of (requirements, binary_questions).
    """
    prompt = DECOMPOSER_USER.format(
        task_prompt=task_prompt,
        extraction_schema=extraction_schema,
    )

    response = call_llm_sync(prompt=prompt, system=DECOMPOSER_SYSTEM)

    # Parse JSON response
    def _extract_json_block(text: str) -> str:
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
            return "\n".join(json_lines)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        return match.group(1) if match else text

    def _repair_json(text: str) -> str:
        text = text.strip()

        # Trim trailing incomplete content until the JSON ends with a closing
        # brace or bracket.
        while text and text[-1] not in "}]":
            text = text[:-1].rstrip()

        # Remove trailing commas before closing brackets/braces.
        text = re.sub(r",\s*([\]}])\s*$", r"\1", text)

        # Balance braces and brackets if the response was truncated.
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_brackets > 0:
            text += "]" * open_brackets
        if open_braces > 0:
            text += "}" * open_braces

        return text

    try:
        text = response.strip()
        text = _extract_json_block(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            repaired = _repair_json(text)
            data = json.loads(repaired)
    except json.JSONDecodeError:
        return [], []

    requirements = [
        Requirement(
            id=r.get("id", f"req_{i}"),
            text=r.get("text", ""),
            dimension=r.get("dimension", ""),
        )
        for i, r in enumerate(data.get("requirements", []))
    ]

    questions = [
        BinaryQuestion(
            id=q.get("id", f"q_{i}"),
            requirement_id=q.get("requirement_id", ""),
            dimension=q.get("dimension", ""),
            text=q.get("text", ""),
            violation_example=q.get("violation_example", ""),
        )
        for i, q in enumerate(data.get("questions", []))
    ]

    return requirements, questions
