import pytest

from weakness_driven_problem_synthesis.cluster_merge import MERGE_PROMPT_MAX_CHARS, _build_merge_prompt, merge_refined_clusters
from weakness_driven_problem_synthesis.cluster_types import ClusterUnit, RefinedCluster


def make_refined(refined_id: str, *, tag_count: int = 20, text_size: int = 10_000) -> RefinedCluster:
    long_text = "y" * text_size
    representative_units = [
        ClusterUnit(
            question_id=index,
            error_tags=[f"tag:{index}"],
            root_cause=long_text,
            ability_dimensions=["reasoning", "generalization"],
            language="python",
            category="algorithms",
            one_line_content=long_text,
        )
        for index in range(1, 4)
    ]
    return RefinedCluster(
        refined_id=refined_id,
        name=f"Weakness {refined_id} {long_text}",
        description=long_text,
        covered_tags=[f"{refined_id}:tag:{index}:{long_text}" for index in range(tag_count)],
        member_question_ids=[1, 2, 3],
        representative_units=representative_units,
        dominant_language="python",
        dominant_category="algorithms",
    )


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        return self.outputs.pop(0)


def test_build_merge_prompt_stays_within_budget_for_large_pair():
    prompt = _build_merge_prompt(
        prompt_template="cluster merge prompt",
        left=make_refined("R001"),
        right=make_refined("R002"),
    )

    assert len(prompt) <= MERGE_PROMPT_MAX_CHARS


def test_build_merge_prompt_raises_when_even_minimal_pair_exceeds_budget(monkeypatch):
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster_merge.MERGE_PROMPT_MAX_CHARS", 140)

    with pytest.raises(ValueError, match="cluster merge prompt exceeds budget after compaction"):
        _build_merge_prompt(
            prompt_template="cluster merge prompt",
            left=make_refined("R001", tag_count=2, text_size=500),
            right=make_refined("R002", tag_count=2, text_size=500),
        )


@pytest.mark.asyncio
async def test_merge_refined_clusters_tries_lower_ranked_pairs_after_top_pair_rejection(monkeypatch):
    left = make_refined("R001", tag_count=1, text_size=20)
    middle = make_refined("R002", tag_count=1, text_size=20)
    right = make_refined("R003", tag_count=1, text_size=20)

    left.covered_tags = ["recursion:base-case-missing"]
    middle.covered_tags = ["recursion:termination-condition-missing"]
    right.covered_tags = ["recursion:state-leak"]

    left.description = "recursion branch termination"
    middle.name = "Greedy local choice trap"
    middle.description = "greedy local choice trap"
    right.description = "recursion stack state bug"

    client = FakeProvider(
        outputs=[
            '{"should_merge":false,"merged_weakness":null}',
            '{"should_merge":true,"merged_weakness":{"id":"W001","name":"Recursion robustness","description":"shared recursion weakness","covered_tags":["recursion:base-case-missing","recursion:state-leak"],"dominant_language":"python","dominant_category":"algorithms"}}',
        ]
    )

    def fake_build_merge_pairs(clusters, *, rejected_pairs):
        if len(clusters) != 3:
            return []
        pair_map = {item.refined_id: index for index, item in enumerate(clusters)}
        first = (0.95, pair_map["R001"], pair_map["R002"])
        second = (0.70, pair_map["R001"], pair_map["R003"])
        if tuple(sorted(("R001", "R002"))) in rejected_pairs:
            return [second]
        pairs = [first]
        return pairs

    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster_merge._build_merge_pairs", fake_build_merge_pairs)

    merged = await merge_refined_clusters(
        [left, middle, right],
        provider="openai",
        model="test-model",
        provider_client=client,
        progress_factory=None,
    )

    names = sorted(item.name for item in merged)
    assert names == ["Greedy local choice trap", "Recursion robustness"]
