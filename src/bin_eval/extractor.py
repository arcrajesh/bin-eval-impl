"""Extractor agent: takes source document + schema, returns ExtractedDocument.

Uses ADK agent with the OpenRouter model for structured extraction.
"""

from __future__ import annotations

import json

from google.adk.agents import Agent

from bin_eval.llm import get_adk_model
from bin_eval.prompts import EXTRACTOR_SYSTEM, EXTRACTOR_USER
from bin_eval.schemas import ExtractedDocument, ExtractedField, ExtractedSection


def build_extractor_agent() -> Agent:
    """Build an ADK agent for document extraction."""
    return Agent(
        name="document_extractor",
        model=get_adk_model(),
        instruction=EXTRACTOR_SYSTEM,
        description="Extracts structured fields from documents according to a schema.",
    )


def extract_document(
    source_document: str,
    extraction_schema: str,
) -> ExtractedDocument:
    """Run extraction synchronously using the OpenAI-compatible client.

    Args:
        source_document: The raw source document text.
        extraction_schema: JSON string describing fields/sections to extract.

    Returns:
        ExtractedDocument with structured extraction results.
    """
    from bin_eval.llm import call_llm_sync

    prompt = EXTRACTOR_USER.format(
        source_document=source_document,
        extraction_schema=extraction_schema,
    )

    response = call_llm_sync(prompt=prompt, system=EXTRACTOR_SYSTEM)

    # Parse JSON from response
    try:
        # Try to extract JSON from possible markdown code block
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
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
        # Fallback: return empty document
        return ExtractedDocument(document_id="parse_error", raw_text=source_document)

    # Build ExtractedDocument from parsed data
    sections = []
    for section_data in data.get("sections", []):
        fields = []
        for field_data in section_data.get("fields", []):
            fields.append(
                ExtractedField(
                    name=field_data.get("name", ""),
                    value=field_data.get("value", ""),
                    confidence=field_data.get("confidence", 1.0),
                    source_span=field_data.get("source_span", ""),
                )
            )
        sections.append(
            ExtractedSection(
                name=section_data.get("name", ""),
                fields=fields,
            )
        )

    return ExtractedDocument(
        document_id=data.get("document_id", ""),
        sections=sections,
        raw_text=source_document,
    )
