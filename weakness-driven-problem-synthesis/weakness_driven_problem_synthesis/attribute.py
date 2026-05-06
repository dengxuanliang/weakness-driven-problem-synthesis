"""Failure attribution stage."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt, load_reference
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord


def _load_existing_attributions(output_path: Path) -> list[Attribution]:
    if not output_path.exists():
        return []

    results: list[Attribution] = []
    with output_path.open() as handle:
        for raw_line in handle:
            if raw_line.strip():
                results.append(Attribution.model_validate_json(raw_line))
    return results


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
    prompt = (
        f"{prompt_template}\n\n"
        f"Vocabulary:\n{vocabulary}\n\n"
        f"Seen tags:\n{seen_tags_section}\n\n"
        f"Question ID: {record.question_id}\n"
        f"Content:\n{record.content}\n\n"
        f"Canonical solution:\n{record.canonical_solution}\n\n"
        f"Completion:\n{record.completion}\n\n"
        f"Labels: category={record.labels.category}, language={record.labels.programming_language}, difficulty={record.labels.difficulty}\n"
        f"Test:\n{record.test}\n"
    )
    payload = await complete_json(
        prompt,
        {"type": "object"},
        provider=provider,
        model=model,
        provider_client=provider_client,
    )
    return Attribution.model_validate(payload)


async def attribute_failures(
    records: list[EvalRecord],
    *,
    output_path: Path,
    provider: str,
    model: str | None,
    concurrency: int,
    provider_client: Any | None = None,
) -> list[Attribution]:
    existing = _load_existing_attributions(output_path)
    processed_ids = {item.question_id for item in existing}
    results = list(existing)
    seen_tags = {tag for item in existing for tag in item.error_tags}
    active_tasks: set[asyncio.Task[tuple[int, Attribution]]] = set()

    async def run_record(index: int, record: EvalRecord, prompt_seen_tags: set[str]) -> tuple[int, Attribution]:
        attribution = await _attribute_record(
            record,
            provider=provider,
            model=model,
            provider_client=provider_client,
            seen_tags=prompt_seen_tags,
        )
        return index, attribution

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
        active_tasks.add(asyncio.create_task(run_record(index, record, prompt_seen_tags)))

    for _ in range(min(concurrency, len(records_to_process))):
        dispatch_next()

    while active_tasks:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        completed = sorted(done, key=lambda task: task.result()[0])
        round_attributions: list[Attribution] = []
        for task in completed:
            _, attribution = task.result()
            with output_path.open("a") as handle:
                handle.write(attribution.model_dump_json() + "\n")
            round_attributions.append(attribution)
            results.append(attribution)

        for attribution in round_attributions:
            seen_tags.update(attribution.error_tags)

        for _ in range(len(completed)):
            dispatch_next()

    results.sort(key=lambda item: item.question_id)
    return results
