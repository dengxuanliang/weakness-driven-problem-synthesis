"""Problem synthesis stage."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.dedup import duplicate_key, ngram_jaccard
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import EvalRecord, SynthesisSummary, SynthProblem, Weakness, WeaknessSet

BASE_BATCH_SIZE = 10
MIN_STATEMENT_CHARS = 200
NGRAM_N = 4
SIMILARITY_THRESHOLD = 0.6
SHAPE_COMBO_SIMILARITY_THRESHOLD = 0.35
PER_SLOT_RETRY_LIMIT = 3
MAX_EXTRA_BATCHES = 2
RECENT_SUMMARY_LIMIT = 20
MAX_COVERAGE_BUCKETS = 12
SYNTHESIS_MAX_TOKENS = 12_000
MAX_REPRESENTATIVE_TAGS = 3
MAX_REPRESENTATIVE_TAG_CHARS = 200
MAX_REPRESENTATIVE_SKETCHES = 2
MAX_REPRESENTATIVE_SKETCH_CHARS = 300
MAX_SINGLE_SKETCH_CHARS = 120


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


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]


def _render_representative_tags(weakness: Weakness) -> str:
    if not weakness.covered_tags:
        return "Representative tags: none"

    selected: list[str] = []
    total_chars = 0
    for raw_tag in weakness.covered_tags:
        tag = raw_tag
        remaining = MAX_REPRESENTATIVE_TAG_CHARS - total_chars
        if remaining <= 0 or len(selected) >= MAX_REPRESENTATIVE_TAGS:
            break
        line_overhead = 2 if selected else 0
        allowed = max(remaining - line_overhead, 0)
        if allowed <= 0:
            break
        if len(tag) > allowed:
            tag = _truncate_text(tag, allowed)
        candidate_chars = total_chars + line_overhead + len(tag)
        if candidate_chars > MAX_REPRESENTATIVE_TAG_CHARS:
            break
        selected.append(tag)
        total_chars = candidate_chars

    if not selected:
        return "Representative tags: none"
    return "Representative tags:\n" + "\n".join(f"- {tag}" for tag in selected)


def _render_failure_sketches(
    *,
    evidence_question_ids: list[int],
    eval_records_by_id: dict[int, EvalRecord] | None,
) -> str:
    if not evidence_question_ids or not eval_records_by_id:
        return "Representative failure sketches: none"

    selected: list[str] = []
    total_chars = 0
    for question_id in evidence_question_ids:
        record = eval_records_by_id.get(question_id)
        if record is None:
            continue
        sketch = _truncate_text(record.content.strip().splitlines()[0], MAX_SINGLE_SKETCH_CHARS)
        if not sketch:
            continue
        line_overhead = 2 if selected else 0
        candidate_chars = total_chars + line_overhead + len(sketch)
        if candidate_chars > MAX_REPRESENTATIVE_SKETCH_CHARS:
            break
        selected.append(sketch)
        total_chars = candidate_chars
        if len(selected) >= MAX_REPRESENTATIVE_SKETCHES:
            break

    if not selected:
        return "Representative failure sketches: none"
    return "Representative failure sketches:\n" + "\n".join(f"- {sketch}" for sketch in selected)


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
    evidence_question_ids: list[int],
    eval_records_by_id: dict[int, EvalRecord] | None,
) -> str:
    prior_summary = _prior_summary(weakness_history)
    coverage_summary = _coverage_summary(weakness_history)
    representative_tags = _render_representative_tags(weakness)
    representative_sketches = _render_failure_sketches(
        evidence_question_ids=evidence_question_ids,
        eval_records_by_id=eval_records_by_id,
    )
    return (
        f"{prompt_template}\n\n"
        f"Weakness ID: {weakness.id}\n"
        f"Weakness name: {weakness.name}\n"
        f"Description: {weakness.description}\n"
        f"Language: {weakness.dominant_language}\n"
        f"Batch size: {batch_size}\n"
        f"{representative_tags}\n"
        f"{representative_sketches}\n"
        f"{prior_summary}\n"
        f"{coverage_summary}\n"
    )


def has_high_similarity(candidate_statement: str, existing_problems: list[dict]) -> bool:
    for problem in existing_problems:
        if ngram_jaccard(candidate_statement, problem["problem_statement"], n=NGRAM_N) >= SIMILARITY_THRESHOLD:
            return True
    return False


def has_similar_shape_combo(candidate: dict, existing_problems: list[dict]) -> bool:
    candidate_combo = (candidate.get("input_scale_class", ""), candidate.get("data_shape_class", ""))
    candidate_statement = candidate.get("problem_statement", "")
    for problem in existing_problems:
        if candidate_combo != (problem.get("input_scale_class", ""), problem.get("data_shape_class", "")):
            continue
        if ngram_jaccard(candidate_statement, problem["problem_statement"], n=NGRAM_N) >= SHAPE_COMBO_SIMILARITY_THRESHOLD:
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
    eval_records_by_id: dict[int, EvalRecord] | None = None,
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
                evidence_question_ids = weakness_set.evidence_question_ids.get(weakness.id, [])
                payload = await complete_json(
                    _build_synthesis_prompt(
                        prompt_template=prompt_template,
                        weakness=weakness,
                        batch_size=batch_size,
                        weakness_history=weakness_history,
                        evidence_question_ids=evidence_question_ids,
                        eval_records_by_id=eval_records_by_id,
                    ),
                    {"type": "array"},
                    provider=provider,
                    model=model,
                    max_tokens=SYNTHESIS_MAX_TOKENS,
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
                        reused_shape_combo = has_similar_shape_combo(candidate.model_dump(), same_weakness_existing)
                        if not is_short and not is_duplicate and not is_similar and not reused_shape_combo:
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
                                evidence_question_ids=evidence_question_ids,
                                eval_records_by_id=eval_records_by_id,
                            ),
                            {"type": "array"},
                            provider=provider,
                            model=model,
                            max_tokens=SYNTHESIS_MAX_TOKENS,
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
