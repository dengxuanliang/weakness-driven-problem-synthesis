import pytest

from weakness_driven_problem_synthesis.cluster_refine import REFINE_PROMPT_MAX_CHARS, _build_refine_prompt, refine_candidate_clusters
from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit


def make_candidate(*, tag_count: int = 20, text_size: int = 10_000) -> CandidateCluster:
    long_text = "x" * text_size
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
    return CandidateCluster(
        candidate_id="C001",
        member_question_ids=[1, 2, 3],
        member_tags=[f"tag:{index}:{long_text}" for index in range(tag_count)],
        representative_units=representative_units,
        dominant_language="python",
        dominant_category="algorithms",
    )


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        return self.outputs.pop(0)


def test_build_refine_prompt_stays_within_budget_for_large_candidate():
    prompt = _build_refine_prompt(
        prompt_template="cluster refine prompt",
        candidate=make_candidate(),
    )

    assert len(prompt) <= REFINE_PROMPT_MAX_CHARS


def test_build_refine_prompt_raises_when_even_minimal_candidate_exceeds_budget(monkeypatch):
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster_refine.REFINE_PROMPT_MAX_CHARS", 120)

    with pytest.raises(ValueError, match="cluster refine prompt exceeds budget after compaction"):
        _build_refine_prompt(
            prompt_template="cluster refine prompt",
            candidate=make_candidate(tag_count=2, text_size=500),
        )


@pytest.mark.asyncio
async def test_refine_candidate_clusters_reassigns_child_evidence_by_covered_tags():
    candidate = CandidateCluster(
        candidate_id="C001",
        member_question_ids=[1, 2, 3],
        member_tags=[
            "recursion:base-case-missing",
            "greedy:wrong-local-choice",
            "recursion:state-leak",
        ],
        representative_units=[
            ClusterUnit(
                question_id=1,
                error_tags=["recursion:base-case-missing"],
                root_cause="misses base case",
                ability_dimensions=["reasoning"],
                language="python",
                category="algorithms",
                one_line_content="case 1",
            ),
            ClusterUnit(
                question_id=2,
                error_tags=["greedy:wrong-local-choice"],
                root_cause="uses greedy shortcut",
                ability_dimensions=["optimization"],
                language="python",
                category="algorithms",
                one_line_content="case 2",
            ),
            ClusterUnit(
                question_id=3,
                error_tags=["recursion:state-leak"],
                root_cause="forgets to isolate state",
                ability_dimensions=["reasoning"],
                language="python",
                category="algorithms",
                one_line_content="case 3",
            ),
        ],
        dominant_language="python",
        dominant_category="algorithms",
    )
    client = FakeProvider(
        outputs=[
            '[{"id":"W001","name":"Recursion robustness","description":"recursion issues","covered_tags":["recursion:base-case-missing","recursion:state-leak"],"dominant_language":"python","dominant_category":"algorithms"},{"id":"W002","name":"Greedy shortcut","description":"greedy issue","covered_tags":["greedy:wrong-local-choice"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )

    refined = await refine_candidate_clusters(
        [candidate],
        provider="openai",
        model="test-model",
        provider_client=client,
        progress=None,
    )

    by_name = {item.name: item for item in refined}
    assert by_name["Recursion robustness"].member_question_ids == [1, 3]
    assert [unit.question_id for unit in by_name["Recursion robustness"].representative_units] == [1, 3]
    assert by_name["Greedy shortcut"].member_question_ids == [2]
    assert [unit.question_id for unit in by_name["Greedy shortcut"].representative_units] == [2]
