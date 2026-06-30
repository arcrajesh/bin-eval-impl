# BinEval — Binary Evaluation Framework for Document Extraction

A Python implementation of the BinEval methodology from ["Ask, Don't Judge: Binary Questions for Interpretable LLM Evaluation and Self-Improvement"](https://arxiv.org/html/2606.27226v1), applied to document extraction tasks.

**Core principle:** Replace one holistic judgment with atomic yes/no checks, aggregated into dimension and overall scores by simple averaging, with a full audit trail linking every scalar score back to question-level verdicts and explanations.

## Setup

```bash
# Clone and install
git clone https://github.com/arcrajesh/bin-eval-impl.git
cd bin-eval-impl
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your OpenRouter credentials:
#   OPENROUTER_API_KEY=your-key-here
#   OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | (required) |
| `OPENROUTER_MODEL` | Model to use via OpenRouter | `google/gemini-2.0-flash-001` |
| `OPENROUTER_TEMPERATURE` | LLM temperature (0 = deterministic) | `0` |
| `OPENROUTER_MAX_TOKENS` | Max response tokens | `4096` |

## Architecture

```
src/bin_eval/
├── __init__.py          # Package init
├── llm.py              # OpenRouter LLM backend + ADK model wrapper
├── schemas.py          # Pydantic models (paper-faithful + document extraction)
├── prompts.py          # Separate prompts for each role (decomposer, evaluator, note-taker, updater)
├── extractor.py        # Document extraction agent
├── decomposer.py       # Task → requirements + binary questions
├── evaluators.py       # Binary question evaluator agents
├── pipeline.py         # Orchestration (Sequential/Parallel) + aggregation
├── updates.py          # Optional iterative update loops
├── metrics.py          # Correlation + product metrics
└── main.py             # CLI entry point
```

## How to Run

```bash
# Run evaluation on a sample document from the repo root
cd /Users/test/git-repos/bin-eval-impl
uv run bin-eval \
  --document examples/sample_document.txt \
  --schema examples/sample_schema.json \
  --task "Extract all invoice fields including metadata, billing info, line items, and totals" \
  --output result.json

# With affine rescaling to 1-5 scale
uv run bin-eval \
  --document examples/sample_document.txt \
  --schema examples/sample_schema.json \
  --scale-min 1.0 --scale-max 5.0 \
  --output result.json
```

Example output:

```text
Results written to: result.json
Overall confidence: 1.0000
Sections evaluated: 6
Questions evaluated: 17

Dimension scores:
  factual_support: 1.0000 (17/17 passed)
```

## BinEval Concepts → Document Extraction Mapping

| BinEval Concept | Document Extraction Application |
|-----------------|-------------------------------|
| Task prompt `T` | Extraction schema (fields/sections to extract) |
| Source `x` | Raw source document |
| Output `y` | Extracted structured data |
| Dimensions | factual_support, formatting_compliance, completeness, consistency, relevance |
| Requirements | Per-field/section extraction criteria |
| Binary questions | "Is field X correctly extracted?", "Does value match source?" |
| Verdict (0/1) | Pass/fail per atomic check |
| Dimension score `S_d` | Mean of verdicts in dimension: `S_d = (1/|Q_d|) * Σ verdicts` |
| Overall score `S` | Mean of all verdicts: `S = (1/N) * Σ verdicts` |
| Affine rescaling | `S' = S*(b-a) + a` (e.g., to 1-5 or 0-100) |
| Audit trail | Every score links back to question → verdict → explanation → evidence |

## Pipeline Flow

1. **Extraction** — ADK agent extracts structured fields from the source document.
2. **Decomposition** — Generates atomic binary questions per requirement, grouped by dimension.
3. **Binary Evaluation** — Each question evaluated independently against (source, extraction).
4. **Aggregation** — Scores computed at field, section, dimension, and overall levels.
5. **Optional: Update Loops** — Cross-model alignment or self-improvement via lesson extraction.

## Aggregation Formulas (Paper-Faithful)

- **Dimension score:** `S_d = (1/|Q_d|) * Σ(verdicts in dimension d)`
- **Overall score:** `S = (1/N) * Σ(all verdicts)` — primary metric
- **Dimension-balanced:** Mean of dimension scores (engineering extension)
- **Affine rescaling:** `S' = S*(b-a) + a` — output layer only, internal stays [0,1]

## Optional Update Loops

- **Cross-model evaluator update:** Align a target evaluator to a reference by analyzing disagreements, extracting lessons, and revising the evaluator prompt.
- **Self prompt update:** Improve extraction quality by collecting failures, generating lessons, and updating the generator prompt.
- **Guardrails:** Early stopping, max prompt length, rollback on regression, semantic dedup of lessons, version history.

## Benchmark Metrics

- **Spearman's ρ** (primary): Rank correlation with human labels
- **Kendall's τ**: Concordance measure
- **Pearson's r**: Linear correlation

## Running Tests

```bash
pytest
```

## Acceptance Criteria

- [x] Auto-generates requirements + binary questions from task prompts
- [x] Evaluator yields one verdict + explanation per question
- [x] Dimension/overall scores are exact means of binary verdicts
- [x] Scores rescale via affine transform
- [x] Outputs stored in traceable schema linking scores → questions → explanations
- [x] Optional update loops with lesson extraction and prompt rewriting
- [x] Benchmark metrics (Spearman/Kendall/Pearson) available
- [x] Evaluation deterministic at temperature 0
- [x] Prompt updates versioned and reversible
- [x] `pytest` passes
- [x] CLI emits JSON with extracted data, per-field/section scores, and overall confidence
