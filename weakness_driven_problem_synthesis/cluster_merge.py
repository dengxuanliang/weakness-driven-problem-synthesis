"""Controlled merge stage for refined weakness clusters."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from weakness_driven_problem_synthesis.cluster_types import ClusterUnit, RefinedCluster
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import Weakness


MERGE_NEIGHBOR_LIMIT = 2
MERGE_SIMILARITY_THRESHOLD = 0.25
MERGE_PROMPT_MAX_CHARS = 20_000
MERGE_TAG_LIMIT_STEPS = (8, 6, 4, 2, 1)
MERGE_REPRESENTATIVE_LIMIT_STEPS = (2, 1)
MERGE_TEXT_LIMIT_STEPS = (1600, 1000, 600, 300, 160, 100, 80)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp_path.replace(path)


def _serialize_merge_state(
    *,
    current: list[RefinedCluster],
    rejected_pairs: set[tuple[str, str]],
    merge_index: int,
    round_index: int,
) -> dict[str, object]:
    return {
        "current": [asdict(cluster) for cluster in current],
        "rejected_pairs": [list(pair) for pair in sorted(rejected_pairs)],
        "merge_index": merge_index,
        "round_index": round_index,
    }


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _compact_tags_for_merge(tags: list[str], *, max_tags: int, max_chars: int) -> list[str]:
    compacted: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for tag in tags:
        if tag in seen:
            continue
        candidate_chars = total_chars + len(tag)
        if compacted and (len(compacted) >= max_tags or candidate_chars > max_chars):
            break
        if not compacted and len(tag) > max_chars:
            compacted.append(tag[:max_chars])
            break
        compacted.append(tag)
        seen.add(tag)
        total_chars = candidate_chars
    return compacted


def _cluster_tokens(cluster: RefinedCluster) -> set[str]:
    tokens = set()
    tokens.update(_tokenize(cluster.name))
    tokens.update(_tokenize(cluster.description))
    for tag in cluster.covered_tags:
        tokens.update(_tokenize(tag.replace(":", " ")))
    for unit in cluster.representative_units:
        tokens.update(_tokenize(unit.root_cause))
        tokens.update(_tokenize(unit.one_line_content))
        tokens.update(item.lower() for item in unit.ability_dimensions)
    return tokens


def _pair_similarity(left: RefinedCluster, right: RefinedCluster) -> float:
    score = 0.0
    left_prefixes = {tag.split(":", 1)[0] for tag in left.covered_tags}
    right_prefixes = {tag.split(":", 1)[0] for tag in right.covered_tags}
    if left_prefixes & right_prefixes:
        score += 0.2
    left_tokens = _cluster_tokens(left)
    right_tokens = _cluster_tokens(right)
    union = left_tokens | right_tokens
    if union:
        score += len(left_tokens & right_tokens) / len(union)
    if left.dominant_language and left.dominant_language == right.dominant_language:
        score += 0.05
    if left.dominant_category and left.dominant_category == right.dominant_category:
        score += 0.05
    return score


def _render_merge_cluster(cluster: RefinedCluster) -> dict[str, object]:
    representatives = []
    for unit in cluster.representative_units[:2]:
        representatives.append(
            {
                "question_id": unit.question_id,
                "root_cause": unit.root_cause,
                "ability_dimensions": unit.ability_dimensions,
                "one_line_content": unit.one_line_content,
            }
        )
    return {
        "id": cluster.refined_id,
        "name": cluster.name,
        "description": cluster.description,
        "covered_tags": _compact_tags_for_merge(cluster.covered_tags, max_tags=8, max_chars=8_000),
        "dominant_language": cluster.dominant_language,
        "dominant_category": cluster.dominant_category,
        "representative_evidence": representatives,
    }


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]


def _compact_refined_cluster(cluster: RefinedCluster, *, tag_limit: int, representative_limit: int, text_limit: int) -> RefinedCluster:
    compacted_units = []
    for unit in cluster.representative_units[:representative_limit]:
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
    return RefinedCluster(
        refined_id=cluster.refined_id,
        name=_truncate_text(cluster.name, text_limit),
        description=_truncate_text(cluster.description, text_limit),
        covered_tags=[_truncate_text(tag, text_limit) for tag in cluster.covered_tags[:tag_limit]],
        member_question_ids=cluster.member_question_ids,
        representative_units=compacted_units,
        dominant_language=cluster.dominant_language,
        dominant_category=cluster.dominant_category,
    )


def _build_merge_prompt(*, prompt_template: str, left: RefinedCluster, right: RefinedCluster) -> str:
    prompt = (
        f"{prompt_template}\n\n"
        f"Left cluster: {json.dumps(_render_merge_cluster(left), ensure_ascii=False)}\n"
        f"Right cluster: {json.dumps(_render_merge_cluster(right), ensure_ascii=False)}\n"
    )
    if len(prompt) <= MERGE_PROMPT_MAX_CHARS:
        return prompt

    for representative_limit in MERGE_REPRESENTATIVE_LIMIT_STEPS:
        for text_limit in MERGE_TEXT_LIMIT_STEPS:
            for tag_limit in MERGE_TAG_LIMIT_STEPS:
                left_compact = _compact_refined_cluster(
                    left,
                    tag_limit=tag_limit,
                    representative_limit=representative_limit,
                    text_limit=text_limit,
                )
                right_compact = _compact_refined_cluster(
                    right,
                    tag_limit=tag_limit,
                    representative_limit=representative_limit,
                    text_limit=text_limit,
                )
                prompt = (
                    f"{prompt_template}\n\n"
                    f"Left cluster: {json.dumps(_render_merge_cluster(left_compact), ensure_ascii=False)}\n"
                    f"Right cluster: {json.dumps(_render_merge_cluster(right_compact), ensure_ascii=False)}\n"
                )
                if len(prompt) <= MERGE_PROMPT_MAX_CHARS:
                    return prompt

    raise ValueError("cluster merge prompt exceeds budget after compaction")


def _build_merge_pairs(clusters: list[RefinedCluster], *, rejected_pairs: set[tuple[str, str]]) -> list[tuple[float, int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for index, cluster in enumerate(clusters):
        scored_neighbors = []
        for other_index, other in enumerate(clusters):
            if index >= other_index:
                continue
            pair_key = tuple(sorted((cluster.refined_id, other.refined_id)))
            if pair_key in rejected_pairs:
                continue
            score = _pair_similarity(cluster, other)
            if score < MERGE_SIMILARITY_THRESHOLD:
                continue
            scored_neighbors.append((score, index, other_index))
        scored_neighbors.sort(reverse=True)
        pairs.extend(scored_neighbors[:MERGE_NEIGHBOR_LIMIT])
    pairs.sort(reverse=True)
    deduped: list[tuple[float, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for score, left_index, right_index in pairs:
        key = tuple(sorted((left_index, right_index)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append((score, left_index, right_index))
    return deduped


def _merge_refined_pair(left: RefinedCluster, right: RefinedCluster, weakness: Weakness, *, merge_index: int) -> RefinedCluster:
    representative_units = list(left.representative_units)
    for unit in right.representative_units:
        if unit.question_id not in {item.question_id for item in representative_units}:
            representative_units.append(unit)
        if len(representative_units) >= 3:
            break
    return RefinedCluster(
        refined_id=f"M{merge_index:03d}",
        name=weakness.name,
        description=weakness.description,
        covered_tags=sorted(set(left.covered_tags) | set(right.covered_tags) | set(weakness.covered_tags)),
        member_question_ids=sorted(set(left.member_question_ids) | set(right.member_question_ids)),
        representative_units=representative_units[:3],
        dominant_language=weakness.dominant_language,
        dominant_category=weakness.dominant_category,
    )


async def merge_refined_clusters(
    refined_clusters: list[RefinedCluster],
    *,
    provider: str,
    model: str | None,
    provider_client: Any | None = None,
    progress_factory: Callable[..., Any] | None = None,
    resume_state: dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
) -> list[Weakness]:
    if not refined_clusters:
        return []

    prompt_template = load_prompt("cluster_merge.txt")
    if resume_state is not None:
        current = list(refined_clusters)
        rejected_pairs = {tuple(pair) for pair in resume_state.get("rejected_pairs", [])}
        merge_index = int(resume_state.get("merge_index", 1))
        round_index = int(resume_state.get("round_index", 1))
    else:
        current = list(refined_clusters)
        rejected_pairs = set()
        merge_index = 1
        round_index = 1
    while True:
        merge_pairs = _build_merge_pairs(current, rejected_pairs=rejected_pairs)
        if not merge_pairs:
            break

        progress = None
        if progress_factory is not None:
            progress = progress_factory(total=len(merge_pairs), initial=0, desc=f"Cluster merge r{round_index}", unit="pair")

        consumed: set[int] = set()
        next_clusters: list[RefinedCluster] = []
        merged_any = False
        try:
            for _, left_index, right_index in merge_pairs:
                if left_index in consumed or right_index in consumed:
                    continue
                left = current[left_index]
                right = current[right_index]
                payload = await complete_json(
                    _build_merge_prompt(prompt_template=prompt_template, left=left, right=right),
                    {
                        "type": "object",
                        "properties": {
                            "should_merge": {"type": "boolean"},
                            "merged_weakness": {"type": ["object", "null"]},
                        },
                        "required": ["should_merge", "merged_weakness"],
                    },
                    provider=provider,
                    model=model,
                    provider_client=provider_client,
                )
                if progress is not None:
                    progress.update(1)
                pair_key = tuple(sorted((left.refined_id, right.refined_id)))
                if not payload.get("should_merge"):
                    rejected_pairs.add(pair_key)
                    continue
                weakness = Weakness.model_validate(payload["merged_weakness"])
                next_clusters.append(_merge_refined_pair(left, right, weakness, merge_index=merge_index))
                merge_index += 1
                consumed.add(left_index)
                consumed.add(right_index)
                merged_any = True
        finally:
            if progress is not None:
                progress.close()

        for index, cluster in enumerate(current):
            if index not in consumed:
                next_clusters.append(cluster)
        if not merged_any:
            exhausted_pairs = len(rejected_pairs) > 0 and not _build_merge_pairs(current, rejected_pairs=rejected_pairs)
            if exhausted_pairs:
                break
            if checkpoint_path is not None:
                _write_json_atomic(
                    checkpoint_path,
                    _serialize_merge_state(
                        current=current,
                        rejected_pairs=rejected_pairs,
                        merge_index=merge_index,
                        round_index=round_index + 1,
                    ),
                )
            round_index += 1
            continue
        if len(next_clusters) == len(current):
            break
        current = next_clusters
        if checkpoint_path is not None:
            _write_json_atomic(
                checkpoint_path,
                _serialize_merge_state(
                    current=current,
                    rejected_pairs=rejected_pairs,
                    merge_index=merge_index,
                    round_index=round_index + 1,
                ),
            )
        round_index += 1

    return [
        Weakness(
            id=cluster.refined_id,
            name=cluster.name,
            description=cluster.description,
            covered_tags=list(cluster.covered_tags),
            dominant_language=cluster.dominant_language,
            dominant_category=cluster.dominant_category,
        )
        for cluster in current
    ]
