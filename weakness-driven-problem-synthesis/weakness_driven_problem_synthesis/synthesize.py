"""Problem synthesis stage."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.dedup import duplicate_key, ngram_jaccard
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import SynthesisSummary, SynthProblem, Weakness, WeaknessSet

BASE_BATCH_SIZE = 10
MIN_STATEMENT_CHARS = 200
NGRAM_N = 4
SIMILARITY_THRESHOLD = 0.6
PER_SLOT_RETRY_LIMIT = 3
MAX_EXTRA_BATCHES = 2
RECENT_SUMMARY_LIMIT = 20
MAX_COVERAGE_BUCKETS = 12


def _build_progress_bar(*, total: int, initial: int, desc: str, unit: str) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return _NullProgressBar()
    return tqdm(total=total, initial=initial, desc=desc, unit=unit)


class _NullProgressBar:
    def update(self, value: int) -> None:
        return None

    def close(self) -> None:
        return None


def _load_existing_problems(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []

    results: list[dict] = []
    with output_path.open() as handle:
        for raw_line in handle:
            if raw_line.strip():
                results.append(json.loads(raw_line))
    return results


def _prior_summary(problems: list[dict], *, limit: int = RECENT_SUMMARY_LIMIT) -> str:
    if not problems:
        return "Recent generated problems: none"

    recent = problems[-limit:]
    lines = [f"Recent generated problems (latest {len(recent)}):"]
    for problem in recent:
        lines.append(
            "- "
            f"{problem['id']} | "
            f"scenario={problem['scenario']} | "
            f"scale={problem.get('input_scale_class', '')} | "
            f"shape={problem.get('data_shape_class', '')} | "
            f"pitfall={problem.get('primary_pitfall', '')} | "
            f"novelty={problem.get('novelty_reason', '')}"
        )
    return "\n".join(lines)


def _coverage_summary(problems: list[dict], *, max_buckets: int = MAX_COVERAGE_BUCKETS) -> str:
    if not problems:
        return "Coverage memory: none"

    def format_counter(name: str, key: str) -> list[str]:
        counts = Counter(problem.get(key, "") for problem in problems if problem.get(key, ""))
        lines = [f"- {name} counts:"]
        for label, count in counts.most_common(max_buckets):
            lines.append(f"  - {label}: {count}")
        return lines

    lines = ["Coverage memory:"]
    lines.extend(format_counter("input_scale_class", "input_scale_class"))
    lines.extend(format_counter("data_shape_class", "data_shape_class"))
    lines.extend(format_counter("primary_pitfall", "primary_pitfall"))
    return "\n".join(lines)


def _expect_array_payload(payload: Any, *, stage: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    preview = repr(payload)
    if len(preview) > 200:
        preview = preview[:200] + "..."
    raise ValueError(f"{stage} expected JSON array payload, got {type(payload).__name__}: {preview}")


def _expect_non_empty_array_payload(payload: Any, *, stage: str) -> list[dict[str, Any]]:
    items = _expect_array_payload(payload, stage=stage)
    if not items:
        raise ValueError(f"{stage} expected non-empty JSON array payload, got empty list")
    return items


def _build_synthesis_prompt(
    *,
    prompt_template: str,
    weakness: Weakness,
    batch_size: int,
    weakness_history: list[dict],
) -> str:
    prior_summary = _prior_summary(weakness_history)
    coverage_summary = _coverage_summary(weakness_history)
    return (
        f"{prompt_template}\n\n"
        f"Weakness ID: {weakness.id}\n"
        f"Weakness name: {weakness.name}\n"
        f"Description: {weakness.description}\n"
        f"Language: {weakness.dominant_language}\n"
        f"Batch size: {batch_size}\n"
        f"{prior_summary}\n"
        f"{coverage_summary}\n"
    )


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
    progress = _build_progress_bar(
        total=sum(allocations.values()),
        initial=len(existing),
        desc="Synthesis",
        unit="problem",
    )
    try:
        for weakness in weakness_set.weaknesses:
            target = allocations.get(weakness.id, 0)
            current = len(existing_by_weakness.get(weakness.id, []))
            skipped += min(current, target)
            if current >= target:
                completed_by_weakness.setdefault(weakness.id, current)
                continue

            batch_index = current // BASE_BATCH_SIZE
            extra_batches_used = 0

            while current < target:
                batch_size = min(BASE_BATCH_SIZE, target - current)
                prompt_template = load_prompt("synthesize.txt")
                weakness_history = existing_by_weakness.get(weakness.id, [])
                payload = await complete_json(
                    _build_synthesis_prompt(
                        prompt_template=prompt_template,
                        weakness=weakness,
                        batch_size=batch_size,
                        weakness_history=weakness_history,
                    ),
                    {"type": "array"},
                    provider=provider,
                    model=model,
                    provider_client=provider_client,
                )

                candidate_payload = _expect_non_empty_array_payload(payload, stage="synthesize_for_weaknesses")
                candidates = [SynthProblem.model_validate(item) for item in candidate_payload]
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
                            progress.update(1)
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

                        weakness_history = existing_by_weakness.get(weakness.id, [])
                        refill_payload = await complete_json(
                            _build_synthesis_prompt(
                                prompt_template=prompt_template,
                                weakness=weakness,
                                batch_size=1,
                                weakness_history=weakness_history,
                            ),
                            {"type": "array"},
                            provider=provider,
                            model=model,
                            provider_client=provider_client,
                        )
                        refill_candidates = _expect_non_empty_array_payload(
                            refill_payload,
                            stage="synthesize_for_weaknesses",
                        )
                        candidate = SynthProblem.model_validate(refill_candidates[0])

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
            shortfall_by_weakness={
                weakness.id: max(allocations.get(weakness.id, 0) - completed_by_weakness.get(weakness.id, 0), 0)
                for weakness in weakness_set.weaknesses
            },
        )
    finally:
        progress.close()
