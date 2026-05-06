"""Weakness clustering stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.schemas import Attribution, Weakness, WeaknessSet


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
    output_path: Path,
    provider: str,
    model: str | None,
    provider_client: Any | None = None,
) -> WeaknessSet:
    if output_path.exists():
        return WeaknessSet.model_validate_json(output_path.read_text())

    payload = await complete_json(
        "Cluster weakness tags",
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
