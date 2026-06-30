"""CLI entry point for BinEval document extraction evaluation.

Accepts a document path/text and schema, runs the pipeline, and writes
the aggregated DocumentEvaluation JSON with complete audit trail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Run the BinEval pipeline from the command line."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="BinEval: Binary evaluation for document extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  bin-eval --document invoice.txt --schema schema.json --output result.json
  bin-eval --document-text "..." --schema schema.json --task "Extract invoice fields"
        """,
    )
    parser.add_argument(
        "--document",
        type=str,
        help="Path to the source document file",
    )
    parser.add_argument(
        "--document-text",
        type=str,
        help="Source document text (alternative to --document)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path to JSON extraction schema file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Extract all fields from the document according to the schema.",
        help="Task prompt describing what to extract",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_result.json",
        help="Output path for the evaluation JSON (default: evaluation_result.json)",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=None,
        help="Minimum value for affine rescaling (optional)",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=None,
        help="Maximum value for affine rescaling (optional)",
    )

    args = parser.parse_args()

    # Load source document
    if args.document:
        doc_path = Path(args.document)
        if not doc_path.exists():
            print(f"Error: Document file not found: {args.document}", file=sys.stderr)
            sys.exit(1)
        source_document = doc_path.read_text(encoding="utf-8")
    elif args.document_text:
        source_document = args.document_text
    else:
        print("Error: Provide either --document or --document-text", file=sys.stderr)
        sys.exit(1)

    # Load extraction schema
    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"Error: Schema file not found: {args.schema}", file=sys.stderr)
        sys.exit(1)
    extraction_schema = schema_path.read_text(encoding="utf-8")

    # Run pipeline
    from bin_eval.pipeline import run_pipeline

    print("Running BinEval pipeline...")
    print(f"  Document: {args.document or '(inline text)'}")
    print(f"  Schema: {args.schema}")
    print(f"  Task: {args.task}")

    result = run_pipeline(
        source_document=source_document,
        task_prompt=args.task,
        extraction_schema=extraction_schema,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )

    # Write output
    output_path = Path(args.output)
    output_data = result.model_dump()
    output_path.write_text(json.dumps(output_data, indent=2, default=str), encoding="utf-8")

    print(f"\nResults written to: {output_path}")
    print(f"  Overall confidence: {result.overall_confidence:.4f}")
    print(f"  Sections evaluated: {len(result.section_scores)}")
    print(f"  Questions evaluated: {len(result.evaluation_result.evaluations)}")

    if result.evaluation_result.dimension_scores:
        print("\n  Dimension scores:")
        for dim, ds in result.evaluation_result.dimension_scores.items():
            print(f"    {dim}: {ds.score:.4f} ({ds.num_passed}/{ds.num_questions} passed)")


if __name__ == "__main__":
    main()
