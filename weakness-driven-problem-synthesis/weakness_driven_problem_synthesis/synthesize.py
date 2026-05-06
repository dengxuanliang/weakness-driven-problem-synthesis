"""Problem synthesis stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.dedup import duplicate_key
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.schemas import SynthesisSummary, SynthProblem, WeaknessSet

MIN_STATEMENT_CHARS = 200


def _load_existing_problems(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    results: list[dict] = []
    with output_path.open() as handle:
        for raw_line in handle:
            if raw_line.strip():
                results.append(json.loads(raw_line))
    return results


async def synthesize_for_weaknesses(
    weakness_set: WeaknessSet,
    *,
    allocations: dict[str, int],
    output_path: Path,
    provider: str,
    model: str | None,
    provider_client: Any | None = None,
) -> SynthesisSummary:
    existing = _load_existing_problems(output_path)
    completed = len(existing)
    retry_count = 0
    existing_by_weakness: dict[str, list[dict]] = {}
    seen_keys = set()
    for problem in existing:
        existing_by_weakness.setdefault(problem["weakness_id"], []).append(problem)
        seen_keys.add(duplicate_key(problem))

    for weakness in weakness_set.weaknesses:
        target = allocations.get(weakness.id, 0)
        current = len(existing_by_weakness.get(weakness.id, []))
        if current >= target:
            continue

        while current < target:
            attempted_keys: set[tuple[str, str]] = set()
            while current < target:
                payload = await complete_json(
                    f"Synthesize one problem for {weakness.id}",
                    {"type": "array"},
                    provider=provider,
                    model=model,
                    provider_client=provider_client,
                )
                candidate = SynthProblem.model_validate(payload[0])
                key = duplicate_key(candidate.model_dump())
                attempted_duplicate = key in attempted_keys
                attempted_keys.add(key)

                is_short = len(candidate.problem_statement) < MIN_STATEMENT_CHARS
                is_duplicate = key in seen_keys or attempted_duplicate
                if is_short or is_duplicate:
                    retry_count += 1
                    continue

                record = candidate.model_dump()
                record["batch_index"] = current
                with output_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")

                existing_by_weakness.setdefault(weakness.id, []).append(record)
                seen_keys.add(key)
                current += 1
                completed += 1
                break

    return SynthesisSummary(completed=completed, retry_count=retry_count, dropped=0)
