"""Pre-clustering helpers for weakness clustering."""

from __future__ import annotations

import re
from collections import Counter

from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord


TOKEN_RE = re.compile(r"[a-z0-9]+")
REPRESENTATIVE_UNIT_LIMIT = 3
TAG_SIMILARITY_THRESHOLD = 0.52
PREFIX_WEIGHT = 0.40
ABILITY_WEIGHT = 0.20
GENERIC_ABILITY_WEIGHT = 0.05
ROOT_OVERLAP_CAP = 0.25
TAG_OVERLAP_CAP = 0.12
CONTENT_OVERLAP_CAP = 0.10
LANGUAGE_WEIGHT = 0.03
CATEGORY_WEIGHT = 0.02
CROSS_PREFIX_BRIDGE_BONUS = 0.10
CROSS_PREFIX_ROOT_OVERLAP_MIN = 0.35
CROSS_PREFIX_CONTENT_OVERLAP_MIN = 0.10
SAME_PREFIX_WEAK_SEMANTIC_PENALTY = 0.12
SAME_PREFIX_ROOT_OVERLAP_MIN = 0.05
SAME_PREFIX_TAG_OVERLAP_RESCUE_MIN = 0.30
GENERIC_ABILITY_DIMENSIONS = {"reasoning", "generalization"}


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _tag_prefix(tag: str) -> str:
    return tag.split(":", 1)[0].strip().lower()


def _dominant_value(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


def build_cluster_units(*, attributions: list[Attribution], eval_records: list[EvalRecord]) -> list[ClusterUnit]:
    records_by_id = {record.question_id: record for record in eval_records if record.question_id is not None}
    units: list[ClusterUnit] = []
    for attribution in attributions:
        record = records_by_id.get(attribution.question_id)
        one_line_content = ""
        language = ""
        category = ""
        if record is not None:
            one_line_content = record.content.strip().splitlines()[0][:120]
            language = record.labels.programming_language
            category = record.labels.category
        units.append(
            ClusterUnit(
                question_id=attribution.question_id,
                error_tags=list(attribution.error_tags),
                root_cause=attribution.root_cause,
                ability_dimensions=list(attribution.ability_dimensions),
                language=language,
                category=category,
                one_line_content=one_line_content,
            )
        )
    return units


def _build_tag_stats(units: list[ClusterUnit]) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for unit in units:
        for tag in unit.error_tags:
            entry = stats.setdefault(
                tag,
                {
                    "question_ids": [],
                    "units": [],
                    "ability_dimensions": set(),
                    "root_tokens": set(),
                    "content_tokens": set(),
                    "languages": [],
                    "categories": [],
                },
            )
            entry["question_ids"].append(unit.question_id)
            if len(entry["units"]) < REPRESENTATIVE_UNIT_LIMIT:
                entry["units"].append(unit)
            entry["ability_dimensions"].update(item.lower() for item in unit.ability_dimensions)
            entry["root_tokens"].update(_tokenize(unit.root_cause))
            entry["content_tokens"].update(_tokenize(unit.one_line_content))
            if unit.language:
                entry["languages"].append(unit.language)
            if unit.category:
                entry["categories"].append(unit.category)
    return stats


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _has_specific_ability_overlap(left: set[str], right: set[str]) -> bool:
    overlap = left & right
    return bool(overlap - GENERIC_ABILITY_DIMENSIONS)


def _tag_similarity_components(left_tag: str, right_tag: str, stats: dict[str, dict[str, object]]) -> dict[str, object]:
    left = stats[left_tag]
    right = stats[right_tag]
    same_prefix = _tag_prefix(left_tag) == _tag_prefix(right_tag)
    ability_overlap = bool(left["ability_dimensions"] & right["ability_dimensions"])
    specific_ability_overlap = _has_specific_ability_overlap(left["ability_dimensions"], right["ability_dimensions"])
    root_overlap = _jaccard(left["root_tokens"], right["root_tokens"])
    left_tag_tokens = _tokenize(left_tag.replace(":", " ").replace("-", " "))
    right_tag_tokens = _tokenize(right_tag.replace(":", " ").replace("-", " "))
    tag_overlap = _jaccard(left_tag_tokens, right_tag_tokens)
    content_overlap = _jaccard(left["content_tokens"], right["content_tokens"])
    same_language = bool(
        _dominant_value(left["languages"]) and _dominant_value(left["languages"]) == _dominant_value(right["languages"])
    )
    same_category = bool(
        _dominant_value(left["categories"]) and _dominant_value(left["categories"]) == _dominant_value(right["categories"])
    )
    bridge_applied = bool(
        (not same_prefix)
        and ability_overlap
        and root_overlap >= CROSS_PREFIX_ROOT_OVERLAP_MIN
        and content_overlap >= CROSS_PREFIX_CONTENT_OVERLAP_MIN
    )
    same_prefix_penalty_applied = bool(
        same_prefix
        and (not specific_ability_overlap)
        and root_overlap < SAME_PREFIX_ROOT_OVERLAP_MIN
        and tag_overlap < SAME_PREFIX_TAG_OVERLAP_RESCUE_MIN
    )
    return {
        "same_prefix": same_prefix,
        "ability_overlap": ability_overlap,
        "specific_ability_overlap": specific_ability_overlap,
        "root_overlap": root_overlap,
        "tag_overlap": tag_overlap,
        "content_overlap": content_overlap,
        "same_language": same_language,
        "same_category": same_category,
        "bridge_applied": bridge_applied,
        "same_prefix_penalty_applied": same_prefix_penalty_applied,
    }


def _tag_similarity(left_tag: str, right_tag: str, stats: dict[str, dict[str, object]]) -> float:
    components = _tag_similarity_components(left_tag, right_tag, stats)
    score = 0.0
    if components["same_prefix"]:
        score += PREFIX_WEIGHT
    if components["specific_ability_overlap"]:
        score += ABILITY_WEIGHT
    elif components["ability_overlap"]:
        score += GENERIC_ABILITY_WEIGHT
    score += min(components["root_overlap"], ROOT_OVERLAP_CAP)
    score += min(components["tag_overlap"], TAG_OVERLAP_CAP)
    if not components["same_prefix_penalty_applied"]:
        score += min(components["content_overlap"], CONTENT_OVERLAP_CAP)
    if components["same_language"]:
        score += LANGUAGE_WEIGHT
    if components["same_category"]:
        score += CATEGORY_WEIGHT
    if components["bridge_applied"]:
        score += CROSS_PREFIX_BRIDGE_BONUS
    if components["same_prefix_penalty_applied"]:
        score -= SAME_PREFIX_WEAK_SEMANTIC_PENALTY
    return score


def _connected_components(tags: list[str], stats: dict[str, dict[str, object]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {tag: set() for tag in tags}
    for index, left_tag in enumerate(tags):
        for right_tag in tags[index + 1 :]:
            if _tag_similarity(left_tag, right_tag, stats) >= TAG_SIMILARITY_THRESHOLD:
                adjacency[left_tag].add(right_tag)
                adjacency[right_tag].add(left_tag)

    seen: set[str] = set()
    components: list[list[str]] = []
    for tag in tags:
        if tag in seen:
            continue
        stack = [tag]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        components.append(sorted(component))
    return components


def propose_candidate_clusters(units: list[ClusterUnit]) -> list[CandidateCluster]:
    if not units:
        return []

    tag_stats = _build_tag_stats(units)
    tags = sorted(tag_stats)
    components = _connected_components(tags, tag_stats)
    candidates: list[CandidateCluster] = []
    for index, component in enumerate(components, start=1):
        question_ids: list[int] = []
        representative_units: list[ClusterUnit] = []
        seen_question_ids: set[int] = set()
        languages: list[str] = []
        categories: list[str] = []
        for tag in component:
            entry = tag_stats[tag]
            question_ids.extend(entry["question_ids"])
            languages.extend(entry["languages"])
            categories.extend(entry["categories"])
            for unit in entry["units"]:
                if unit.question_id in seen_question_ids:
                    continue
                representative_units.append(unit)
                seen_question_ids.add(unit.question_id)
                if len(representative_units) >= REPRESENTATIVE_UNIT_LIMIT:
                    break
        candidates.append(
            CandidateCluster(
                candidate_id=f"C{index:03d}",
                member_question_ids=sorted(set(question_ids)),
                member_tags=component,
                representative_units=representative_units,
                dominant_language=_dominant_value(languages),
                dominant_category=_dominant_value(categories),
            )
        )
    return candidates
