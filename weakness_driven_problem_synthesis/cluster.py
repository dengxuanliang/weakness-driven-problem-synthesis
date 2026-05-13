"""Weakness clustering stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord, Weakness, WeaknessSet

CLUSTER_PROMPT_MAX_CHARS = 25_000


def _expect_array_payload(payload: Any, *, stage: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    preview = repr(payload)
    if len(preview) > 200:
        preview = preview[:200] + "..."
    raise ValueError(f"{stage} expected JSON array payload, got {type(payload).__name__}: {preview}")


def _expect_array_payload_with_min_items(payload: Any, *, stage: str, min_items: int) -> list[dict[str, Any]]:
    items = _expect_array_payload(payload, stage=stage)
    if len(items) < min_items:
        if min_items == 1:
            raise ValueError(f"{stage} expected non-empty JSON array payload, got empty list")
        raise ValueError(f"{stage} expected at least {min_items} JSON array items, got {len(items)}")
    return items


def _validate_weakness_set(weakness_set: WeaknessSet, *, attributions: list[Attribution]) -> WeaknessSet:
    _expect_array_payload_with_min_items(
        [item.model_dump() for item in weakness_set.weaknesses],
        stage="cluster_weaknesses",
        min_items=1 if attributions else 0,
    )
    return weakness_set


def _render_tag_summary_blocks(tag_summaries: dict[str, list[dict[str, object]]]) -> list[str]:
    return [f"- {tag}: {json.dumps(representatives, ensure_ascii=False)}" for tag, representatives in tag_summaries.items()]


def _build_cluster_prompt_from_blocks(*, prompt_template: str, blocks: list[str]) -> str:
    return f"{prompt_template}\n\nRepresentative question summaries:\n" + "\n".join(blocks)


def _chunk_blocks_by_char_budget(*, prompt_template: str, blocks: list[str], budget_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        if len(_build_cluster_prompt_from_blocks(prompt_template=prompt_template, blocks=[block])) > budget_chars:
            raise ValueError("single tag summary block exceeds cluster prompt budget")

        candidate = current + [block]
        if current and len(_build_cluster_prompt_from_blocks(prompt_template=prompt_template, blocks=candidate)) > budget_chars:
            chunks.append(current)
            current = [block]
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


async def _cluster_chunk(
    *,
    prompt_template: str,
    blocks: list[str],
    provider: str,
    model: str | None,
    provider_client: Any | None,
) -> list[Weakness]:
    payload = await complete_json(
        _build_cluster_prompt_from_blocks(prompt_template=prompt_template, blocks=blocks),
        {"type": "array"},
        provider=provider,
        model=model,
        provider_client=provider_client,
    )
    weaknesses_payload = _expect_array_payload_with_min_items(
        payload,
        stage="cluster_weaknesses",
        min_items=1 if blocks else 0,
    )
    return [Weakness.model_validate(item) for item in weaknesses_payload]


def _render_weakness_merge_blocks(weaknesses: list[Weakness]) -> list[str]:
    blocks = []
    for weakness in weaknesses:
        blocks.append(
            "- "
            + json.dumps(
                {
                    "name": weakness.name,
                    "description": weakness.description,
                    "dominant_language": weakness.dominant_language,
                    "dominant_category": weakness.dominant_category,
                },
                ensure_ascii=False,
            )
        )
    return blocks


def _renumber_weaknesses(weaknesses: list[Weakness]) -> list[Weakness]:
    renumbered = []
    for index, weakness in enumerate(weaknesses, start=1):
        renumbered.append(weakness.model_copy(update={"id": f"W{index:03d}"}))
    return renumbered


async def _merge_chunked_weaknesses(
    *,
    chunked_weaknesses: list[list[Weakness]],
    prompt_template: str,
    provider: str,
    model: str | None,
    provider_client: Any | None,
) -> list[Weakness]:
    merged_input = [weakness for chunk in chunked_weaknesses for weakness in chunk]
    merge_blocks = _render_weakness_merge_blocks(merged_input)
    merge_chunks = _chunk_blocks_by_char_budget(
        prompt_template=prompt_template,
        blocks=merge_blocks,
        budget_chars=CLUSTER_PROMPT_MAX_CHARS,
    )
    if len(merge_chunks) != 1:
        raise ValueError("merged weakness summaries still exceed cluster prompt budget")
    return await _cluster_chunk(
        prompt_template=prompt_template,
        blocks=merge_chunks[0],
        provider=provider,
        model=model,
        provider_client=provider_client,
    )


def map_questions_to_clusters(
    attributions: list[Attribution],
    weaknesses: list[Weakness],
) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {weakness.id: [] for weakness in weaknesses}
    for attribution in attributions:
        tags = set(attribution.error_tags)
        for weakness in weaknesses:
            if tags.intersection(weakness.covered_tags):
                mapping[weakness.id].append(attribution.question_id)
    return mapping


async def cluster_weaknesses(
    attributions: list[Attribution],
    *,
    eval_records: list[EvalRecord],
    output_path: Path,
    provider: str,
    model: str | None,
    provider_client: Any | None = None,
) -> WeaknessSet:
    if output_path.exists():
        weakness_set = WeaknessSet.model_validate_json(output_path.read_text())
        return _validate_weakness_set(weakness_set, attributions=attributions)

    prompt_template = load_prompt("cluster.txt")
    records_by_id = {record.question_id: record for record in eval_records if record.question_id is not None}
    tag_summaries: dict[str, list[dict[str, object]]] = {}
    for attribution in attributions:
        for tag in attribution.error_tags:
            tag_summaries.setdefault(tag, [])
            if len(tag_summaries[tag]) < 3:
                source_record = records_by_id.get(attribution.question_id)
                one_line_content = ""
                category = ""
                language = ""
                if source_record is not None:
                    one_line_content = source_record.content.strip().splitlines()[0][:120]
                    category = source_record.labels.category
                    language = source_record.labels.programming_language
                tag_summaries[tag].append(
                    {
                        "id": attribution.question_id,
                        "category": category,
                        "language": language,
                        "one_line_content": one_line_content,
                    }
                )

    blocks = _render_tag_summary_blocks(tag_summaries)
    chunks = _chunk_blocks_by_char_budget(
        prompt_template=prompt_template,
        blocks=blocks,
        budget_chars=CLUSTER_PROMPT_MAX_CHARS,
    )
    min_items = 1 if attributions else 0
    if not chunks:
        weaknesses: list[Weakness] = []
    elif len(chunks) == 1:
        weaknesses = await _cluster_chunk(
            prompt_template=prompt_template,
            blocks=chunks[0],
            provider=provider,
            model=model,
            provider_client=provider_client,
        )
    else:
        chunked_weaknesses = []
        for chunk_blocks in chunks:
            chunked_weaknesses.append(
                await _cluster_chunk(
                    prompt_template=prompt_template,
                    blocks=chunk_blocks,
                    provider=provider,
                    model=model,
                    provider_client=provider_client,
                )
            )
        weaknesses = await _merge_chunked_weaknesses(
            chunked_weaknesses=chunked_weaknesses,
            prompt_template=prompt_template,
            provider=provider,
            model=model,
            provider_client=provider_client,
        )

    weaknesses = _renumber_weaknesses(weaknesses)
    _expect_array_payload_with_min_items(
        [item.model_dump() for item in weaknesses],
        stage="cluster_weaknesses",
        min_items=min_items,
    )
    weakness_set = WeaknessSet(
        weaknesses=weaknesses,
        evidence_question_ids=map_questions_to_clusters(attributions, weaknesses),
    )
    _validate_weakness_set(weakness_set, attributions=attributions)
    output_path.write_text(weakness_set.model_dump_json(indent=2))
    return weakness_set
