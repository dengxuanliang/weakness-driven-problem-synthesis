"""Failure attribution stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
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
) -> Attribution:
    payload = await complete_json(
        record.content,
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
    semaphore = asyncio.Semaphore(concurrency)

    async def run_record(record: EvalRecord) -> Attribution | None:
        if record.question_id is None or record.question_id in processed_ids:
            return None
        async with semaphore:
            attribution = await _attribute_record(
                record,
                provider=provider,
                model=model,
                provider_client=provider_client,
            )
            with output_path.open("a") as handle:
                handle.write(attribution.model_dump_json() + "\n")
            return attribution

    pending = [run_record(record) for record in records]
    for attribution in await asyncio.gather(*pending):
        if attribution is not None:
            results.append(attribution)

    results.sort(key=lambda item: item.question_id)
    return results
