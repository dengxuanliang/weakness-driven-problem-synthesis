# Weakness-Driven Problem Synthesis

A skill and Python pipeline for turning model evaluation logs into high-difficulty synthesized coding problems that target observed weakness patterns.

## What It Does

Given an evaluation `jsonl` log, this pipeline:

1. Filters failed records
2. Attributes each failure to concrete error tags and root causes
3. Clusters repeated error patterns into weakness groups
4. Allocates synthesis quota across weaknesses
5. Generates diversified coding problems for each weakness
6. Exports a solver-facing view for downstream code-generation evaluation

The output is designed for weakness-targeted benchmark expansion rather than generic problem generation.

## Pipeline Overview

The pipeline has four main stages:

- `attribute`: classify each failed sample into structured error tags, root cause, and ability dimensions
- `cluster`: group recurring failure tags into higher-level weaknesses
- `allocate`: distribute total synthesis budget across weaknesses
- `synthesize`: generate new coding problems that target each weakness while enforcing anti-homogeneity constraints

CLI entrypoints currently support starting from:

- `attribute`: full pipeline from eval log
- `cluster`: reuse an existing `error_attributions.jsonl`
- `synthesize`: reuse an existing `error_attributions.jsonl` and `weaknesses.json`

A final export step writes `solver_view.jsonl`, which is the solver-facing artifact intended for downstream model solving.

## Input

The pipeline expects an evaluation log in `jsonl` format.

Each record should contain fields compatible with the current schema, including:

- `id` or `question_id`
- `content`
- `canonical_solution`
- `completion`
- `test`
- `labels.category`
- `labels.programming_language`
- `labels.difficulty`
- `pass_at_1`

Only failed records are used for weakness analysis.

## Output Artifacts

A successful run may produce:

- `error_attributions.jsonl`
  - per-failure structured attribution results
- `weaknesses.json`
  - clustered weakness definitions and supporting question ids
- `synthesized_problems.jsonl`
  - full internal synthesis records, including diversity-control metadata
- `solver_view.jsonl`
  - solver-facing problem records with a composed `solver_prompt`
- `report.md`
  - run summary, weakness counts, allocation, and completion statistics

### Solver-Facing Artifact

Use `solver_view.jsonl` for downstream solving.

Each solver record contains:

- `id`
- `language`
- `difficulty`
- `problem_statement`
- `function_signature`
- `input_format`
- `output_format`
- `constraints`
- `solver_prompt`

Internal synthesis-only fields such as `edge_cases_hinted` are intentionally excluded from this view.

## Installation

Python 3.11+ is required.

Install dependencies:

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Configuration

Set provider credentials through environment variables.

For OpenAI-compatible gateways:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=...
```

Optional debug dump path:

```bash
export WEAKNESS_SYNTH_DEBUG_PATH=debug_invalid_json.txt
```

The loader checks environment variables first. If a required value is missing, it falls back to the repository-root `.env` file. `.env` never overrides an already-set environment variable.

Model selection priority is:

1. `--model`
2. provider-specific environment variable or repository-root `.env`
3. error

There is no built-in default model name.

You can also store local secrets in `.env`, but that file must remain uncommitted.

## Usage

Run from the repo root:

```bash
python -m weakness_driven_problem_synthesis.run \
  --eval-log small_eval_500.jsonl \
  --total-questions 50 \
  --output-dir smoke_output_500_50 \
  --provider openai \
  --model your-model-name \
  --yes
```

Or use the script wrapper:

```bash
python scripts/run.py \
  --eval-log small_eval_500.jsonl \
  --total-questions 50 \
  --output-dir smoke_output_500_50 \
  --provider openai \
  --model your-model-name \
  --yes
```

Start from the cluster stage with an existing attribution artifact:

```bash
python -m weakness_driven_problem_synthesis.run \
  --eval-log small_eval_500.jsonl \
  --total-questions 50 \
  --output-dir smoke_output_500_50 \
  --start-stage cluster \
  --attributions-file previous_run/error_attributions.jsonl \
  --provider openai \
  --model your-model-name \
  --yes
```

Start directly from synthesis with existing attribution and weakness artifacts:

```bash
python -m weakness_driven_problem_synthesis.run \
  --total-questions 50 \
  --output-dir smoke_output_500_50 \
  --start-stage synthesize \
  --attributions-file previous_run/error_attributions.jsonl \
  --weaknesses-file previous_run/weaknesses.json \
  --provider openai \
  --model your-model-name \
  --yes
```

## Resume and Restart Behavior

By default, the pipeline runs with resume enabled.

- `--resume` reuses existing stage artifacts when possible
- `--no-resume` clears stage artifacts before the run
- `--restart` removes the entire output directory before starting

If a run exits through an empty-analysis path, stale downstream synthesis artifacts are cleaned automatically.

## Notes on Diversity Control

The synthesis stage uses both prompt-level and post-generation filtering to reduce homogeneity.

Current controls include:

- per-weakness history summaries in the synthesis prompt
- coverage memory over scale, data shape, and pitfall classes
- duplicate-key filtering
- statement similarity filtering
- soft rejection of reused shape/scale combinations with similar statements

The target behavior is diversified weakness-aligned generation, not exact template variation.

## Testing

Run the test suite with:

```bash
pytest -q
```

## Repository Notes

Ignored local-only artifacts include:

- `.env`
- sampled eval logs such as `small_eval*.jsonl`
- smoke output directories such as `smoke_output*/`

These are intentionally kept out of version control because they may contain local configuration, large logs, or generated artifacts.
