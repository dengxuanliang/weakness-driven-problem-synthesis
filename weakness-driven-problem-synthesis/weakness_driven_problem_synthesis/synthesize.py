"""Problem synthesis stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.dedup import duplicate_key, ngram_jaccard
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import SynthesisSummary, SynthProblem, WeaknessSet

BASE_BATCH_SIZE = 10
MIN_STATEMENT_CHARS = 200
NGRAM_N = 4
SIMILARITY_THRESHOLD = 0.6
PER_SLOT_RETRY_LIMIT = 3
MAX_EXTRA_BATCHES = 2


def _load_existing_problems(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    results: list[dict] = []
    with output_path.open() as handle:
        for raw_line in handle:
            if raw_line.strip():
                results.append(json.loads(raw_line))
    return results


def _prior_summary(problems: list[dict]) -> str:
    if not problems:
        return "Prior problems summary: none"
    items = [f"{problem['id']}: {problem['scenario']}" for problem in problems]
    return "Prior problems summary: " + "; ".join(items)


def has_high_similarity(candidate_statement: str, existing_problems: list[dict]) -> bool:
    for problem in existing_problems:
        if ngram_jaccard(candidate_statement, problem["problem_statement"], n=NGRAM_N) >= SIMILARITY_THRESHOLD:
            return True
    return False


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
    dropped = 0
    skipped = 0
    extra_batches = 0
    existing_by_weakness: dict[str, list[dict]] = {}
    completed_by_weakness: dict[str, int] = {}
    seen_keys = set()
    for problem in existing:
        existing_by_weakness.setdefault(problem["weakness_id"], []).append(problem)
        completed_by_weakness[problem["weakness_id"]] = completed_by_weakness.get(problem["weakness_id"], 0) + 1
        seen_keys.add(duplicate_key(problem))

    for weakness in weakness_set.weaknesses:
        target = allocations.get(weakness.id, 0)
        current = len(existing_by_weakness.get(weakness.id, []))
        if current >= target:
            skipped += current
            completed_by_weakness.setdefault(weakness.id, current)
            continue

        batch_index = current // BASE_BATCH_SIZE
        extra_batches_used = 0

        while current < target:
            batch_size = min(BASE_BATCH_SIZE, target - current)
            prompt_template = load_prompt("synthesize.txt")
            prior_summary = _prior_summary(existing_by_weakness.get(weakness.id, []))
            payload = await complete_json(
                (
                    f"{prompt_template}\n\n"
                    f"Weakness ID: {weakness.id}\n"
                    f"Weakness name: {weakness.name}\n"
                    f"Description: {weakness.description}\n"
                    f"Language: {weakness.dominant_language}\n"
                    f"Batch size: {batch_size}\n"
                    f"{prior_summary}\n"
                ),
                {"type": "array"},
                provider=provider,
                model=model,
                provider_client=provider_client,
            )

            candidates = [SynthProblem.model_validate(item) for item in payload]
            slot_index = 0
            while slot_index < batch_size:
                if slot_index >= len(candidates):
                    break

                candidate = candidates[slot_index]
                attempt_count = 0
                attempted_keys: set[tuple[str, str]] = set()

                while True:
                    key = duplicate_key(candidate.model_dump())
                    attempted_duplicate = key in attempted_keys
                    attempted_keys.add(key)

                    is_short = len(candidate.problem_statement) < MIN_STATEMENT_CHARS
                    same_weakness_existing = existing_by_weakness.get(weakness.id, [])
                    is_duplicate = key in seen_keys or attempted_duplicate
                    is_similar = has_high_similarity(candidate.problem_statement, same_weakness_existing)
                    if not is_short and not is_duplicate and not is_similar:
                        record = candidate.model_dump()
                        record["batch_index"] = batch_index
                        with output_path.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")

                        existing_by_weakness.setdefault(weakness.id, []).append(record)
                        completed_by_weakness[weakness.id] = completed_by_weakness.get(weakness.id, 0) + 1
                        seen_keys.add(key)
                        current += 1
                        completed += 1
                        slot_index += 1
                        break

                    retry_count += 1
                    attempt_count += 1
                    if attempt_count >= PER_SLOT_RETRY_LIMIT:
                        dropped += 1
                        slot_index += 1
                        if current < target and extra_batches_used < MAX_EXTRA_BATCHES:
                            extra_batches_used += 1
                            extra_batches += 1
                        break

                    refill_payload = await complete_json(
                        (
                            f"{prompt_template}\n\n"
                            f"Weakness ID: {weakness.id}\n"
                            f"Weakness name: {weakness.name}\n"
                            f"Description: {weakness.description}\n"
                            f"Language: {weakness.dominant_language}\n"
                            f"Batch size: 1\n"
                            f"{prior_summary}\n"
                        ),
                        {"type": "array"},
                        provider=provider,
                        model=model,
                        provider_client=provider_client,
                    )
                    candidate = SynthProblem.model_validate(refill_payload[0])

            if current >= target:
                break

            batch_index += 1
            if extra_batches_used >= MAX_EXTRA_BATCHES and current < target:
                break

    return SynthesisSummary(
        completed=completed,
        retry_count=retry_count,
        dropped=dropped,
        skipped=skipped,
        extra_batches=extra_batches,
        completed_by_weakness=completed_by_weakness,
    )
