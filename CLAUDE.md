# Weakness-Driven Problem Synthesis

This repository provides a weakness-driven problem synthesis pipeline that turns evaluation `jsonl` logs into synthesized coding problems targeting observed model weaknesses.

## Core Entry Point

Run the pipeline with:

```bash
python -m weakness_driven_problem_synthesis.run \
  --eval-log <eval_log.jsonl> \
  --total-questions <count> \
  --output-dir <output_dir> \
  --provider <openai|anthropic> \
  --model <model_name> \
  --yes
```

An equivalent wrapper is also available:

```bash
python scripts/run.py ...
```

## Main Pipeline Stages

- `load_filter`: load eval records and keep failed samples
- `attribute`: classify failures into tags, root causes, and ability dimensions
- `cluster`: group recurring tags into weakness clusters
- `allocate`: distribute synthesis quota across weaknesses
- `synthesize`: generate diversified weakness-targeted problems
- `solver_view`: export a solver-facing artifact for downstream solving

## Important Outputs

- `error_attributions.jsonl`
- `weaknesses.json`
- `synthesized_problems.jsonl`
- `solver_view.jsonl`
- `report.md`

Use `solver_view.jsonl` for downstream solver-model evaluation. Do not use internal-only synthesis fields such as `edge_cases_hinted` as solver input.

## Configuration

Expected environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` for OpenAI-compatible gateways
- `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`
- `WEAKNESS_SYNTH_DEBUG_PATH` for optional invalid-JSON dumps

Environment variables take priority. If a required value is missing, the loader falls back to the repository-root `.env` file. `.env` never overrides an already-set environment variable.

Model selection priority is:

1. `--model`
2. provider-specific environment variable or repository-root `.env`
3. error

There is no built-in default model name.

Local secrets may be stored in `.env`, but `.env` must never be committed.

## Repository Notes

- Smoke outputs and sampled eval logs are intentionally gitignored.
- The repository root is the current directory. Do not recreate an extra nested project directory.
- See `README.md` for full usage and output documentation.
