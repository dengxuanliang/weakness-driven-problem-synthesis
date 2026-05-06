"""Weakness clustering stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord, Weakness, WeaknessSet


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
        return WeaknessSet.model_validate_json(output_path.read_text())

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

    prompt_sections = ["Representative question summaries:"]
    for tag, representatives in tag_summaries.items():
        prompt_sections.append(f"- {tag}: {representatives}")

    payload = await complete_json(
        f"{prompt_template}\n\n" + "\n".join(prompt_sections),
        {"type": "array"},
        provider=provider,
        model=model,
        provider_client=provider_client,
    )
    weaknesses = [Weakness.model_validate(item) for item in payload]
    weakness_set = WeaknessSet(
        weaknesses=weaknesses,
        evidence_question_ids=map_questions_to_clusters(attributions, weaknesses),
    )
    output_path.write_text(weakness_set.model_dump_json(indent=2))
    return weakness_set
