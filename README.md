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
- `cluster`: reuse an existing `error_attributions.jsonl` plus the original eval log for representative context
- `synthesize`: reuse an existing `error_attributions.jsonl` and `weaknesses.json`

A final export step writes `solver_view.jsonl`, which is the solver-facing artifact intended for downstream model solving.

Recent hardening includes:

- one-command local bootstrap through `scripts/bootstrap.sh`
- resumable entrypoints for `cluster` and `synthesize`
- artifact consistency checks before synthesis-only runs
- global LLM throttling and burst cooldowns
- retry handling for retryable gateway block pages
- proxy-aware OpenAI/Anthropic clients
- oversized-record skipping and failed-attribution logging

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
- `failed_attribution_records.jsonl`
  - failed attribution attempts that were skipped without stopping the run
- `skipped_failed_records.jsonl`
  - failed eval records skipped before attribution because the raw JSONL line was too large
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

One-command bootstrap:

```bash
bash scripts/bootstrap.sh
```

If `.env` does not already exist, the bootstrap script creates an empty `.env`
template containing:

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

Existing `.env` files are never overwritten.

Manual install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
source .venv/bin/activate
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

Optional proxy configuration is supported through standard proxy environment variables or `.env` entries:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

Optional LLM request throttling controls:

```bash
export WEAKNESS_SYNTH_MAX_IN_FLIGHT=8
export WEAKNESS_SYNTH_MIN_INTERVAL_MS=150
export WEAKNESS_SYNTH_BURST_LIMIT=12
export WEAKNESS_SYNTH_BURST_COOLDOWN_MS=1200
```

The loader checks environment variables first. If a required value is missing, it falls back to the repository-root `.env` file. `.env` never overrides an already-set environment variable.

The default provider is `openai`. Pass `--provider anthropic` if you want to use the Anthropic client path instead.

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

If you use the default OpenAI-compatible path, `--provider` may be omitted.

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

`cluster` still requires `--eval-log` because weakness clustering uses representative question context from the original evaluation records.

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

The synthesis-only entrypoint validates the supplied artifacts before generating:

- every `weaknesses.json` evidence question id must appear in truly failed attributions
- each evidence attribution must share at least one tag with the weakness `covered_tags`
- mismatched artifacts fail fast instead of silently producing misaligned problems

## Resume and Restart Behavior

By default, the pipeline runs with resume enabled.

- `--resume` reuses existing stage artifacts when possible
- `--no-resume` clears stage artifacts before the run
- `--restart` removes the entire output directory before starting

If a run exits through an empty-analysis path, stale downstream synthesis artifacts are cleaned automatically.

Input records larger than 1 MB are skipped before attribution and recorded in `skipped_failed_records.jsonl`. Single attribution failures are recorded in `failed_attribution_records.jsonl` and do not stop the whole run.

## Notes on Diversity Control

The synthesis stage uses both prompt-level and post-generation filtering to reduce homogeneity.

Current controls include:

- per-weakness history summaries in the synthesis prompt
- coverage memory over scale, data shape, and pitfall classes
- duplicate-key filtering
- statement similarity filtering
- soft rejection of reused shape/scale combinations with similar statements

The target behavior is diversified weakness-aligned generation, not exact template variation.

## Skill Usage

This repository is both a Python package and a Codex/Claude-style skill.

- `SKILL.md` describes when the skill should be used
- `references/` contains prompt and vocabulary assets
- `weakness_driven_problem_synthesis/` contains the implementation package
- `scripts/run.py` is a thin CLI wrapper

After bootstrap or install, run the CLI from the repository root or from any environment where the package is installed.

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
