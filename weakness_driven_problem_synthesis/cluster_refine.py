"""LLM-backed candidate-cluster refinement."""

from __future__ import annotations

import json
from typing import Any

from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit, RefinedCluster
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import Weakness


REFINE_PROMPT_MAX_CHARS = 20_000
REFINE_TAG_LIMIT_STEPS = (16, 12, 8, 6, 4, 2, 1)
REFINE_REPRESENTATIVE_LIMIT_STEPS = (3, 2, 1)
REFINE_TEXT_LIMIT_STEPS = (2000, 1200, 800, 400, 200, 120, 80)


def _expect_array_payload_with_min_items(payload: Any, *, stage: str, min_items: int) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        preview = repr(payload)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        raise ValueError(f"{stage} expected JSON array payload, got {type(payload).__name__}: {preview}")
    if len(payload) < min_items:
        if min_items == 1:
            raise ValueError(f"{stage} expected non-empty JSON array payload, got empty list")
        raise ValueError(f"{stage} expected at least {min_items} JSON array items, got {len(payload)}")
    return payload


def _render_candidate_cluster(candidate: CandidateCluster) -> str:
    representatives = []
    for unit in candidate.representative_units:
        representatives.append(
            {
                "question_id": unit.question_id,
                "root_cause": unit.root_cause,
                "ability_dimensions": unit.ability_dimensions,
                "language": unit.language,
                "category": unit.category,
                "one_line_content": unit.one_line_content,
            }
        )
    return (
        f"Candidate ID: {candidate.candidate_id}\n"
        f"Dominant language: {candidate.dominant_language}\n"
        f"Dominant category: {candidate.dominant_category}\n"
        f"Candidate tags: {json.dumps(candidate.member_tags, ensure_ascii=False)}\n"
        f"Representative evidence: {json.dumps(representatives, ensure_ascii=False)}"
    )


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]


def _compact_candidate(candidate: CandidateCluster, *, tag_limit: int, representative_limit: int, text_limit: int) -> CandidateCluster:
    compacted_units = []
    for unit in candidate.representative_units[:representative_limit]:
        compacted_units.append(
            ClusterUnit(
                question_id=unit.question_id,
                error_tags=list(unit.error_tags[:1]),
                root_cause=_truncate_text(unit.root_cause, text_limit),
                ability_dimensions=list(unit.ability_dimensions[:2]),
                language=unit.language,
                category=unit.category,
                one_line_content=_truncate_text(unit.one_line_content, text_limit),
            )
        )
    compacted_tags = [_truncate_text(tag, text_limit) for tag in candidate.member_tags[:tag_limit]]
    return CandidateCluster(
        candidate_id=candidate.candidate_id,
        member_question_ids=candidate.member_question_ids,
        member_tags=compacted_tags,
        representative_units=compacted_units,
        dominant_language=candidate.dominant_language,
        dominant_category=candidate.dominant_category,
    )


def _select_child_units(candidate: CandidateCluster, *, covered_tags: list[str]) -> list[ClusterUnit]:
    covered = set(covered_tags)
    matched = [unit for unit in candidate.representative_units if covered.intersection(unit.error_tags)]
    if matched:
        return matched[:3]

    covered_prefixes = {tag.split(":", 1)[0] for tag in covered}
    prefix_matched = [
        unit
        for unit in candidate.representative_units
        if {tag.split(":", 1)[0] for tag in unit.error_tags}.intersection(covered_prefixes)
    ]
    if prefix_matched:
        return prefix_matched[:1]

    return candidate.representative_units[:1]


def _select_child_question_ids(candidate: CandidateCluster, *, child_units: list[ClusterUnit]) -> list[int]:
    question_ids = sorted({unit.question_id for unit in child_units})
    if question_ids:
        return question_ids
    return candidate.member_question_ids[:1]


def _build_refine_prompt(*, prompt_template: str, candidate: CandidateCluster) -> str:
    prompt = f"{prompt_template}\n\n{_render_candidate_cluster(candidate)}"
    if len(prompt) <= REFINE_PROMPT_MAX_CHARS:
        return prompt

    for representative_limit in REFINE_REPRESENTATIVE_LIMIT_STEPS:
        for text_limit in REFINE_TEXT_LIMIT_STEPS:
            for tag_limit in REFINE_TAG_LIMIT_STEPS:
                compacted = _compact_candidate(
                    candidate,
                    tag_limit=tag_limit,
                    representative_limit=representative_limit,
                    text_limit=text_limit,
                )
                prompt = f"{prompt_template}\n\n{_render_candidate_cluster(compacted)}"
                if len(prompt) <= REFINE_PROMPT_MAX_CHARS:
                    return prompt

    raise ValueError("cluster refine prompt exceeds budget after compaction")


async def refine_candidate_clusters(
    candidates: list[CandidateCluster],
    *,
    provider: str,
    model: str | None,
    provider_client: Any | None = None,
    progress: Any | None = None,
) -> list[RefinedCluster]:
    if not candidates:
        return []

    prompt_template = load_prompt("cluster_refine.txt")
    refined: list[RefinedCluster] = []
    for candidate in candidates:
        payload = await complete_json(
            _build_refine_prompt(prompt_template=prompt_template, candidate=candidate),
            {"type": "array"},
            provider=provider,
            model=model,
            provider_client=provider_client,
        )
        refined_payload = _expect_array_payload_with_min_items(
            payload,
            stage="refine_candidate_clusters",
            min_items=1,
        )
        for index, item in enumerate(refined_payload, start=1):
            weakness = Weakness.model_validate(item)
            child_units = _select_child_units(candidate, covered_tags=list(weakness.covered_tags))
            refined.append(
                RefinedCluster(
                    refined_id=f"{candidate.candidate_id}-R{index:02d}",
                    name=weakness.name,
                    description=weakness.description,
                    covered_tags=list(weakness.covered_tags),
                    member_question_ids=_select_child_question_ids(candidate, child_units=child_units),
                    representative_units=child_units,
                    dominant_language=weakness.dominant_language,
                    dominant_category=weakness.dominant_category,
                )
            )
        if progress is not None:
            progress.update(1)
    return refined
