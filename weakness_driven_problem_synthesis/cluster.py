"""Weakness clustering stage."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit, RefinedCluster
from weakness_driven_problem_synthesis.cluster_merge import merge_refined_clusters
from weakness_driven_problem_synthesis.cluster_precluster import build_cluster_units, propose_candidate_clusters
from weakness_driven_problem_synthesis.cluster_refine import refine_candidate_clusters
from weakness_driven_problem_synthesis.llm_client import complete_json
from weakness_driven_problem_synthesis.prompts import load_prompt
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord, Weakness, WeaknessSet

CLUSTER_PROMPT_MAX_CHARS = 25_000
MERGE_BLOCK_MAX_TAGS = 8
MERGE_BLOCK_MAX_TAG_CHARS = 8_000
LARGE_INPUT_TAG_THRESHOLD = 12
CLUSTER_CANDIDATES_ARTIFACT = "cluster_candidates.json"
CLUSTER_REFINED_ARTIFACT = "cluster_refined.json"
CLUSTER_MERGE_STATE_ARTIFACT = "cluster_merge_state.json"


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


def _write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp_path.replace(path)


def _cluster_unit_from_payload(payload: dict[str, object]) -> ClusterUnit:
    return ClusterUnit(
        question_id=int(payload["question_id"]),
        error_tags=list(payload["error_tags"]),
        root_cause=str(payload["root_cause"]),
        ability_dimensions=list(payload["ability_dimensions"]),
        language=str(payload["language"]),
        category=str(payload["category"]),
        one_line_content=str(payload["one_line_content"]),
    )


def _candidate_cluster_from_payload(payload: dict[str, object]) -> CandidateCluster:
    return CandidateCluster(
        candidate_id=str(payload["candidate_id"]),
        member_question_ids=[int(item) for item in payload["member_question_ids"]],
        member_tags=list(payload["member_tags"]),
        representative_units=[_cluster_unit_from_payload(item) for item in payload["representative_units"]],
        dominant_language=str(payload["dominant_language"]),
        dominant_category=str(payload["dominant_category"]),
    )


def _refined_cluster_from_payload(payload: dict[str, object]) -> RefinedCluster:
    return RefinedCluster(
        refined_id=str(payload["refined_id"]),
        name=str(payload["name"]),
        description=str(payload["description"]),
        covered_tags=list(payload["covered_tags"]),
        member_question_ids=[int(item) for item in payload["member_question_ids"]],
        representative_units=[_cluster_unit_from_payload(item) for item in payload["representative_units"]],
        dominant_language=str(payload["dominant_language"]),
        dominant_category=str(payload["dominant_category"]),
    )


def _load_candidate_clusters(path: Path) -> list[CandidateCluster]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("invalid cluster candidates artifact")
    return [_candidate_cluster_from_payload(item) for item in payload]


def _load_refined_checkpoint(path: Path) -> dict[str, list[RefinedCluster]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("invalid cluster refined artifact")
    refined_by_candidate_id: dict[str, list[RefinedCluster]] = {}
    for item in payload:
        candidate_id = str(item["candidate_id"])
        refined_by_candidate_id[candidate_id] = [_refined_cluster_from_payload(cluster) for cluster in item["refined_clusters"]]
    return refined_by_candidate_id


def _load_merge_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("invalid cluster merge state artifact")
    return {
        "current": [_refined_cluster_from_payload(item) for item in payload.get("current", [])],
        "rejected_pairs": [list(pair) for pair in payload.get("rejected_pairs", [])],
        "merge_index": payload.get("merge_index", 1),
        "round_index": payload.get("round_index", 1),
    }


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


def _render_weakness_merge_blocks(weaknesses: list[Weakness]) -> list[str]:
    blocks = []
    for weakness in weaknesses:
        blocks.append(
            "- "
            + json.dumps(
                {
                    "name": weakness.name,
                    "description": weakness.description,
                    "covered_tags": _compact_tags_for_merge(
                        weakness.covered_tags,
                        max_tags=MERGE_BLOCK_MAX_TAGS,
                        max_chars=MERGE_BLOCK_MAX_TAG_CHARS,
                    ),
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


def _max_merge_rounds_for(initial_weakness_count: int) -> int:
    return max(6, initial_weakness_count)


async def _merge_chunked_weaknesses(
    *,
    chunked_weaknesses: list[list[Weakness]],
    prompt_template: str,
    provider: str,
    model: str | None,
    provider_client: Any | None,
) -> list[Weakness]:
    # Hierarchical merge keeps each merge prompt under budget even when the
    # first-pass weakness summaries are still too large to combine at once.
    current_weaknesses = [weakness for chunk in chunked_weaknesses for weakness in chunk]
    max_rounds = _max_merge_rounds_for(len(current_weaknesses))
    for _round_index in range(max_rounds):
        merge_blocks = _render_weakness_merge_blocks(current_weaknesses)
        merge_chunks = _chunk_blocks_by_char_budget(
            prompt_template=prompt_template,
            blocks=merge_blocks,
            budget_chars=CLUSTER_PROMPT_MAX_CHARS,
        )
        if not merge_chunks:
            return []
        if len(merge_chunks) == 1:
            return await _cluster_chunk(
                prompt_template=prompt_template,
                blocks=merge_chunks[0],
                provider=provider,
                model=model,
                provider_client=provider_client,
            )

        next_weaknesses: list[Weakness] = []
        for chunk_blocks in merge_chunks:
            next_weaknesses.extend(
                await _cluster_chunk(
                    prompt_template=prompt_template,
                    blocks=chunk_blocks,
                    provider=provider,
                    model=model,
                    provider_client=provider_client,
                )
            )
        next_merge_blocks = _render_weakness_merge_blocks(next_weaknesses)
        next_merge_chunks = _chunk_blocks_by_char_budget(
            prompt_template=prompt_template,
            blocks=next_merge_blocks,
            budget_chars=CLUSTER_PROMPT_MAX_CHARS,
        )
        chunk_count_improved = len(next_merge_chunks) < len(merge_chunks)
        weakness_count_improved = len(next_weaknesses) < len(current_weaknesses)
        if not chunk_count_improved and not weakness_count_improved:
            raise ValueError("cluster merge made no progress under prompt budget")
        current_weaknesses = next_weaknesses

    raise ValueError("cluster merge exceeded max rounds under prompt budget")


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

    candidates_path = output_path.with_name(CLUSTER_CANDIDATES_ARTIFACT)
    refined_path = output_path.with_name(CLUSTER_REFINED_ARTIFACT)
    merge_state_path = output_path.with_name(CLUSTER_MERGE_STATE_ARTIFACT)

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

    min_items = 1 if attributions else 0
    if len(tag_summaries) > LARGE_INPUT_TAG_THRESHOLD:
        merge_state = _load_merge_state(merge_state_path) if merge_state_path.exists() else None
        if merge_state is not None:
            print("Cluster: resume merge candidates")
            refined_clusters = merge_state["current"]
            weaknesses = await merge_refined_clusters(
                refined_clusters,
                provider=provider,
                model=model,
                provider_client=provider_client,
                progress_factory=_build_progress_bar,
                resume_state=merge_state,
                checkpoint_path=merge_state_path,
            )
        else:
            resume_refined_by_candidate_id = _load_refined_checkpoint(refined_path) if refined_path.exists() else None
            if resume_refined_by_candidate_id is not None:
                print("Cluster: resume refine candidates")
            if candidates_path.exists():
                candidates = _load_candidate_clusters(candidates_path)
            else:
                print("Cluster: build units")
                units = build_cluster_units(attributions=attributions, eval_records=eval_records)
                print("Cluster: precluster")
                candidates = propose_candidate_clusters(units)
                _write_json_atomic(candidates_path, [asdict(candidate) for candidate in candidates])
            if resume_refined_by_candidate_id is not None and set(resume_refined_by_candidate_id) >= {candidate.candidate_id for candidate in candidates}:
                refined_clusters = [
                    cluster
                    for candidate in candidates
                    for cluster in resume_refined_by_candidate_id.get(candidate.candidate_id, [])
                ]
            else:
                print("Cluster: refine candidates")
                refine_progress = _build_progress_bar(total=len(candidates), initial=0, desc="Cluster refine", unit="cluster")
                try:
                    refined_clusters = await refine_candidate_clusters(
                        candidates,
                        provider=provider,
                        model=model,
                        provider_client=provider_client,
                        progress=refine_progress,
                        resume_refined_by_candidate_id=resume_refined_by_candidate_id,
                        checkpoint_path=refined_path,
                    )
                finally:
                    refine_progress.close()
            print("Cluster: merge candidates")
            weaknesses = await merge_refined_clusters(
                refined_clusters,
                provider=provider,
                model=model,
                provider_client=provider_client,
                progress_factory=_build_progress_bar,
                checkpoint_path=merge_state_path,
            )
    else:
        prompt_template = load_prompt("cluster.txt")
        blocks = _render_tag_summary_blocks(tag_summaries)
        chunks = _chunk_blocks_by_char_budget(
            prompt_template=prompt_template,
            blocks=blocks,
            budget_chars=CLUSTER_PROMPT_MAX_CHARS,
        )
        if not chunks:
            weaknesses = []
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
    if not attributions and not tag_summaries:
        weaknesses: list[Weakness] = []

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
    if merge_state_path.exists():
        merge_state_path.unlink()
    return weakness_set
