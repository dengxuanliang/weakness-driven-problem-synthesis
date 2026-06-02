Run the weakness-driven problem synthesis pipeline in this repository.

Before running:

- Confirm the evaluation log path exists.
- Confirm the total question count.
- Confirm the provider and model.
- Confirm the required API environment variables are configured.
- If the output directory already exists, decide whether to use `--resume`, `--no-resume`, or `--restart`.

Preferred command:

```bash
python -m weakness_driven_problem_synthesis.run $ARGUMENTS
```

Equivalent wrapper:

```bash
python scripts/run.py $ARGUMENTS
```

Expected artifacts:

- `error_attributions.jsonl`
- `weaknesses.json`
- `synthesized_problems.jsonl`
- `solver_view.jsonl`
- `report.md`

If the user asks for solver-facing output, point them to `solver_view.jsonl`.
