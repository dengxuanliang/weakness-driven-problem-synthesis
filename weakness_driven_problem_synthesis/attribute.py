"""Failure attribution stage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt, load_reference
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord

ATTRIBUTE_PROMPT_MAX_CHARS = 25_000
ATTRIBUTE_CONTENT_MAX_CHARS = 6_000
ATTRIBUTE_CANONICAL_SOLUTION_MAX_CHARS = 5_000
ATTRIBUTE_COMPLETION_MAX_CHARS = 5_000
ATTRIBUTE_TEST_MAX_CHARS = 6_000
ATTRIBUTE_SEEN_TAGS_MAX_CHARS = 1_000


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


def _load_existing_attributions(output_path: Path) -> list[Attribution]:
    if not output_path.exists():
        return []

    results: list[Attribution] = []
    with output_path.open() as handle:
        for raw_line in handle:
            if raw_line.strip():
                results.append(Attribution.model_validate_json(raw_line))
    return results


def _truncate_text(text: str, *, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[truncated {omitted} chars from {label}]"


def _trim_seen_tags_section(text: str, *, max_chars: int = ATTRIBUTE_SEEN_TAGS_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[truncated {omitted} chars from seen_tags]"


def _build_attribute_prompt(
    *,
    prompt_template: str,
    vocabulary: str,
    seen_tags_section: str,
    record: EvalRecord,
) -> str:
    content_text = _truncate_text(record.content, max_chars=ATTRIBUTE_CONTENT_MAX_CHARS, label="content")
    canonical_solution_text = _truncate_text(
        record.canonical_solution,
        max_chars=ATTRIBUTE_CANONICAL_SOLUTION_MAX_CHARS,
        label="canonical_solution",
    )
    completion_text = _truncate_text(
        record.completion,
        max_chars=ATTRIBUTE_COMPLETION_MAX_CHARS,
        label="completion",
    )
    test_text = _truncate_text(record.test_text, max_chars=ATTRIBUTE_TEST_MAX_CHARS, label="test")
    seen_tags_text = _trim_seen_tags_section(seen_tags_section)

    prompt = (
        f"{prompt_template}\n\n"
        f"Vocabulary:\n{vocabulary}\n\n"
        f"Seen tags:\n{seen_tags_text}\n\n"
        f"Question ID: {record.question_id}\n"
        f"Content:\n{content_text}\n\n"
        f"Canonical solution:\n{canonical_solution_text}\n\n"
        f"Completion:\n{completion_text}\n\n"
        f"Labels: category={record.labels.category}, language={record.labels.programming_language}, difficulty={record.labels.difficulty}\n"
        f"Test:\n{test_text}\n"
    )
    if len(prompt) <= ATTRIBUTE_PROMPT_MAX_CHARS:
        return prompt

    overflow = len(prompt) - ATTRIBUTE_PROMPT_MAX_CHARS
    content_text = _truncate_text(
        record.content,
        max_chars=max(500, ATTRIBUTE_CONTENT_MAX_CHARS - overflow // 3),
        label="content",
    )
    canonical_solution_text = _truncate_text(
        record.canonical_solution,
        max_chars=max(500, ATTRIBUTE_CANONICAL_SOLUTION_MAX_CHARS - overflow // 3),
        label="canonical_solution",
    )
    completion_text = _truncate_text(
        record.completion,
        max_chars=max(500, ATTRIBUTE_COMPLETION_MAX_CHARS - overflow // 3),
        label="completion",
    )
    test_text = _truncate_text(
        record.test_text,
        max_chars=max(500, ATTRIBUTE_TEST_MAX_CHARS - overflow // 3),
        label="test",
    )
    seen_tags_text = _trim_seen_tags_section(seen_tags_section, max_chars=500)

    prompt = (
        f"{prompt_template}\n\n"
        f"Vocabulary:\n{vocabulary}\n\n"
        f"Seen tags:\n{seen_tags_text}\n\n"
        f"Question ID: {record.question_id}\n"
        f"Content:\n{content_text}\n\n"
        f"Canonical solution:\n{canonical_solution_text}\n\n"
        f"Completion:\n{completion_text}\n\n"
        f"Labels: category={record.labels.category}, language={record.labels.programming_language}, difficulty={record.labels.difficulty}\n"
        f"Test:\n{test_text}\n"
    )
    return prompt[:ATTRIBUTE_PROMPT_MAX_CHARS]


def _write_failed_attribution_record(*, output_path: Path, record: EvalRecord, exc: Exception) -> None:
    payload = {
        "question_id": record.question_id,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "content_chars": len(record.content),
        "canonical_solution_chars": len(record.canonical_solution),
        "completion_chars": len(record.completion),
        "test_chars": len(record.test_text),
    }
    import json

    with output_path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_attribution_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    for field_name in ("error_tags", "ability_dimensions"):
        value = normalized.get(field_name)
        if isinstance(value, str):
            normalized[field_name] = [value]
    return normalized


async def _attribute_record(
    record: EvalRecord,
    *,
    provider: str,
    model: str | None,
    provider_client: Any | None,
    seen_tags: set[str],
) -> Attribution:
    vocabulary = load_reference("error_tag_vocabulary.md")
    prompt_template = load_prompt("attribute.txt")
    seen_tags_section = ", ".join(sorted(seen_tags)) if seen_tags else "none"
    prompt = _build_attribute_prompt(
        prompt_template=prompt_template,
        vocabulary=vocabulary,
        seen_tags_section=seen_tags_section,
        record=record,
    )
    payload = await complete_json(
        prompt,
        {"type": "object"},
        provider=provider,
        model=model,
        provider_client=provider_client,
    )
    return Attribution.model_validate(_normalize_attribution_payload(payload))


async def attribute_failures(
    records: list[EvalRecord],
    *,
    output_path: Path,
    failed_output_path: Path | None = None,
    provider: str,
    model: str | None,
    concurrency: int,
    provider_client: Any | None = None,
) -> list[Attribution]:
    failed_output_path = failed_output_path or output_path.with_name("failed_attribution_records.jsonl")
    existing = _load_existing_attributions(output_path)
    processed_ids = {item.question_id for item in existing}
    results = list(existing)
    seen_tags = {tag for item in existing for tag in item.error_tags}
    active_tasks: set[asyncio.Task[tuple[int, EvalRecord, Attribution]]] = set()
    progress = _build_progress_bar(
        total=len(records),
        initial=len(existing),
        desc="Attribution",
        unit="record",
    )

    async def run_record(index: int, record: EvalRecord, prompt_seen_tags: set[str]) -> tuple[int, EvalRecord, Attribution]:
        attribution = await _attribute_record(
            record,
            provider=provider,
            model=model,
            provider_client=provider_client,
            seen_tags=prompt_seen_tags,
        )
        return index, record, attribution

    records_to_process = [
        (index, record)
        for index, record in enumerate(records)
        if record.question_id is not None and record.question_id not in processed_ids
    ]
    next_index = 0

    def dispatch_next() -> None:
        nonlocal next_index
        if next_index >= len(records_to_process):
            return
        index, record = records_to_process[next_index]
        next_index += 1
        prompt_seen_tags = set(seen_tags)
        task = asyncio.create_task(run_record(index, record, prompt_seen_tags))
        setattr(task, "_record", record)
        active_tasks.add(task)

    try:
        for _ in range(min(concurrency, len(records_to_process))):
            dispatch_next()

        while active_tasks:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            completed: list[tuple[int, Attribution]] = []
            round_attributions: list[Attribution] = []
            for task in done:
                try:
                    index, _, attribution = task.result()
                except Exception as exc:
                    record = getattr(task, "_record", None)
                    if record is not None:
                        _write_failed_attribution_record(output_path=failed_output_path, record=record, exc=exc)
                    progress.update(1)
                    continue
                completed.append((index, attribution))

            for _, attribution in sorted(completed, key=lambda item: item[0]):
                with output_path.open("a") as handle:
                    handle.write(attribution.model_dump_json() + "\n")
                round_attributions.append(attribution)
                results.append(attribution)
                progress.update(1)

            for attribution in round_attributions:
                seen_tags.update(attribution.error_tags)

            for _ in range(len(done)):
                dispatch_next()

        results.sort(key=lambda item: item.question_id)
        return results
    finally:
        progress.close()
