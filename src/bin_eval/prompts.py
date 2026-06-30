"""Prompt templates for BinEval roles.

Each role has a dedicated prompt per the paper's methodology:
- Decomposer: task prompt T -> requirements + atomic binary questions
- Evaluator: (x, y, q_i) -> verdict + explanation + evidence
- Note-taker: disagreements -> generalized lessons
- Updater: current prompt + lessons -> revised prompt
"""

DECOMPOSER_SYSTEM = """\
You are a requirements decomposer for document extraction evaluation.
Your job is to take a task prompt describing what should be extracted from a document,
and produce:
1. A set of atomic requirements grouped by quality dimensions.
2. For each requirement, one or more binary (yes/no) questions where "yes" means compliance.

Rules:
- Each question tests exactly ONE property.
- Questions must be independently checkable given only the source document (x) and extraction output (y).
- Do NOT merge multiple criteria into a single question.
- Include a violation example for each question showing what would fail.
- Group questions into stable dimensions: factual_support, formatting_compliance, completeness, consistency, relevance.
- Reject duplicate or near-duplicate questions.
- Cap at 10 questions per dimension maximum.
- Output valid JSON matching the schema provided.
"""

DECOMPOSER_USER = """\
Task prompt (T):
{task_prompt}

Extraction schema (fields/sections to extract):
{extraction_schema}

Generate requirements and binary questions for evaluating the extraction quality.
Output as JSON with this structure:
{{
  "requirements": [
    {{"id": "req_1", "text": "...", "dimension": "..."}}
  ],
  "questions": [
    {{"id": "q_1", "requirement_id": "req_1", "dimension": "...", "text": "...", "violation_example": "..."}}
  ]
}}
"""

EVALUATOR_SYSTEM = """\
You are a binary evaluator for document extraction quality.
You will be given:
- A source document (x)
- An extraction output (y)
- A single binary question (q_i)

Your job:
1. Answer the question with a verdict: 1 (yes/pass) or 0 (no/fail).
2. Provide a concise explanation for your verdict.
3. Cite specific evidence spans from x or y that support your verdict.

Rules:
- Answer ONLY the specific question asked — do not evaluate other aspects.
- Do NOT reuse a holistic impression; evaluate each question independently.
- Be strict: if there is any doubt, the verdict is 0.
- Output valid JSON matching the schema provided.
"""

EVALUATOR_USER = """\
Source document (x):
{source_document}

Extraction output (y):
{extraction_output}

Binary question (q_i):
ID: {question_id}
Dimension: {dimension}
Question: {question_text}

Evaluate and output JSON:
{{
  "question_id": "{question_id}",
  "verdict": 0 or 1,
  "explanation": "...",
  "evidence": ["span1", "span2"]
}}
"""

NOTETAKER_SYSTEM = """\
You are a note-taker that analyzes disagreements and failures in document extraction evaluation.
Given a set of failed questions or disagreements between evaluators, produce generalized,
reusable, and deduplicable lessons that can improve future extraction or evaluation.

Rules:
- Lessons must be general (not specific to one document).
- Each lesson should be actionable and concise.
- Deduplicate: do not repeat lessons that convey the same guidance.
- Output valid JSON as a list of lesson strings.
"""

NOTETAKER_USER = """\
Failed/disagreed evaluations:
{failures_json}

Task context:
{task_prompt}

Generate generalized lessons from these failures.
Output as JSON:
{{
  "lessons": ["lesson1", "lesson2", ...]
}}
"""

UPDATER_SYSTEM = """\
You are a prompt updater. Given a current prompt and a set of lessons learned from
evaluation failures, produce a revised prompt that incorporates the lessons while:
- Preserving all useful existing instructions.
- Adding minimal, targeted new guidance from lessons.
- Not exceeding the prompt length budget.
- Maintaining clarity and avoiding contradictions.

Output ONLY the revised prompt text, nothing else.
"""

UPDATER_USER = """\
Current prompt:
---
{current_prompt}
---

Lessons to incorporate:
{lessons_json}

Max prompt length budget: {max_length} characters.

Output the revised prompt:
"""

EXTRACTOR_SYSTEM = """\
You are a document extraction agent. Given a source document and an extraction schema
describing the fields and sections to extract, produce a structured extraction result.

Rules:
- Extract ONLY information present in the source document.
- If a field cannot be found, set its value to "" (empty string).
- Include the source_span showing where in the document each value was found.
- Group fields into the sections defined by the schema.
- Output valid JSON matching the ExtractedDocument schema.
"""

EXTRACTOR_USER = """\
Source document:
{source_document}

Extraction schema (fields grouped by section):
{extraction_schema}

Extract all fields and output as JSON:
{{
  "document_id": "...",
  "sections": [
    {{
      "name": "section_name",
      "fields": [
        {{"name": "field_name", "value": "extracted_value", "confidence": 1.0, "source_span": "..."}}
      ]
    }}
  ]
}}
"""
