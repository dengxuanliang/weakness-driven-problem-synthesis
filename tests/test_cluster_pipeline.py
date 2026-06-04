import pytest

from weakness_driven_problem_synthesis.cluster_merge import merge_refined_clusters
from weakness_driven_problem_synthesis.cluster_precluster import _build_tag_stats, _tag_similarity_components, build_cluster_units, propose_candidate_clusters
from weakness_driven_problem_synthesis.cluster_refine import refine_candidate_clusters
from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit, RefinedCluster
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        return self.outputs.pop(0)


class ProgressSpy:
    def __init__(self):
        self.updates = []
        self.closed = False

    def update(self, value: int) -> None:
        self.updates.append(value)

    def close(self) -> None:
        self.closed = True


def make_attribution(question_id: int, error_tags: list[str], *, root_cause: str = "misses base case") -> Attribution:
    return Attribution.model_validate(
        {
            "question_id": question_id,
            "is_truly_failed": True,
            "error_tags": error_tags,
            "root_cause": root_cause,
            "ability_dimensions": ["reasoning"],
            "evidence_snippet": "snippet",
        }
    )


def make_eval_record(question_id: int, content: str, language: str = "python", category: str = "algorithms") -> EvalRecord:
    return EvalRecord.model_validate(
        {
            "question_id": question_id,
            "content": content,
            "canonical_solution": "def solve(): pass",
            "completion": "def solve(): return None",
            "test": "assert True",
            "labels": {
                "category": category,
                "programming_language": language,
                "difficulty": "hard",
            },
            "pass_at_1": 0,
        }
    )


def make_candidate(candidate_id: str, tags: list[str], question_ids: list[int]) -> CandidateCluster:
    representative_units = [
        ClusterUnit(
            question_id=question_id,
            error_tags=[tags[0]],
            root_cause="misses base case",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content=f"case {question_id}",
        )
        for question_id in question_ids[:2]
    ]
    return CandidateCluster(
        candidate_id=candidate_id,
        member_question_ids=question_ids,
        member_tags=tags,
        representative_units=representative_units,
        dominant_language="python",
        dominant_category="algorithms",
    )


def make_refined(refined_id: str, tags: list[str], question_ids: list[int], name: str, *, root_cause: str = "misses base case") -> RefinedCluster:
    representative_units = [
        ClusterUnit(
            question_id=question_id,
            error_tags=[tags[0]],
            root_cause=root_cause,
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content=f"case {question_id}",
        )
        for question_id in question_ids[:2]
    ]
    return RefinedCluster(
        refined_id=refined_id,
        name=name,
        description="shared weakness",
        covered_tags=tags,
        member_question_ids=question_ids,
        representative_units=representative_units,
        dominant_language="python",
        dominant_category="algorithms",
    )


def test_build_cluster_units_captures_eval_context():
    units = build_cluster_units(
        attributions=[make_attribution(1, ["recursion:base-case-missing"])],
        eval_records=[make_eval_record(1, "Recursive traversal over nested arrays")],
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.question_id == 1
    assert unit.error_tags == ["recursion:base-case-missing"]
    assert unit.language == "python"
    assert unit.category == "algorithms"
    assert unit.one_line_content == "Recursive traversal over nested arrays"


def test_propose_candidate_clusters_groups_related_tags():
    units = [
        ClusterUnit(
            question_id=1,
            error_tags=["recursion:base-case-missing"],
            root_cause="misses base case",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="nested traversal",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["recursion:termination-condition-missing"],
            root_cause="forgets termination branch",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="tree dfs",
        ),
        ClusterUnit(
            question_id=3,
            error_tags=["greedy:wrong-local-choice"],
            root_cause="uses greedy shortcut",
            ability_dimensions=["optimization"],
            language="python",
            category="algorithms",
            one_line_content="interval schedule",
        ),
    ]

    candidates = propose_candidate_clusters(units)

    candidate_tags = sorted(sorted(candidate.member_tags) for candidate in candidates)
    assert ["greedy:wrong-local-choice"] in candidate_tags
    assert sorted(["recursion:base-case-missing", "recursion:termination-condition-missing"]) in candidate_tags


def test_propose_candidate_clusters_links_cross_prefix_tags_with_shared_root_cause():
    units = [
        ClusterUnit(
            question_id=1,
            error_tags=["recursion:state-leak"],
            root_cause="shared mutable state leaks across recursive frames",
            ability_dimensions=["reasoning", "state_management"],
            language="python",
            category="algorithms",
            one_line_content="recursive search with shared path state",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["backtracking:shared-mutable-state"],
            root_cause="shared mutable state leaks across search branches",
            ability_dimensions=["reasoning", "state_management"],
            language="python",
            category="algorithms",
            one_line_content="backtracking search with shared path state",
        ),
        ClusterUnit(
            question_id=3,
            error_tags=["greedy:wrong-local-choice"],
            root_cause="uses greedy shortcut",
            ability_dimensions=["optimization"],
            language="python",
            category="algorithms",
            one_line_content="interval schedule",
        ),
    ]

    candidates = propose_candidate_clusters(units)

    candidate_tags = sorted(sorted(candidate.member_tags) for candidate in candidates)
    assert sorted(["backtracking:shared-mutable-state", "recursion:state-leak"]) in candidate_tags


def test_propose_candidate_clusters_keeps_same_prefix_tags_separate_when_root_cause_differs():
    units = [
        ClusterUnit(
            question_id=1,
            error_tags=["dp:transition-missing"],
            root_cause="omits recurrence transition case",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="dynamic programming over prefixes",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["dp:memory-overflow"],
            root_cause="allocates quadratic table without compression",
            ability_dimensions=["complexity"],
            language="python",
            category="algorithms",
            one_line_content="memory-heavy dynamic programming table",
        ),
    ]

    candidates = propose_candidate_clusters(units)

    candidate_tags = sorted(sorted(candidate.member_tags) for candidate in candidates)
    assert ["dp:memory-overflow"] in candidate_tags
    assert ["dp:transition-missing"] in candidate_tags


def test_propose_candidate_clusters_does_not_treat_generic_ability_overlap_as_same_prefix_exemption():
    units = [
        ClusterUnit(
            question_id=1,
            error_tags=["dp:transition-missing"],
            root_cause="omits recurrence transition case",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="dynamic programming over prefixes",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["dp:memory-overflow"],
            root_cause="allocates quadratic table without compression",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="memory-heavy dynamic programming table",
        ),
    ]

    candidates = propose_candidate_clusters(units)

    candidate_tags = sorted(sorted(candidate.member_tags) for candidate in candidates)
    assert ["dp:memory-overflow"] in candidate_tags
    assert ["dp:transition-missing"] in candidate_tags


def test_propose_candidate_clusters_does_not_bridge_cross_prefix_tags_with_only_topic_similarity():
    units = [
        ClusterUnit(
            question_id=1,
            error_tags=["graph:state-update-order"],
            root_cause="updates distance state in the wrong relaxation order",
            ability_dimensions=["reasoning", "graph_search"],
            language="python",
            category="algorithms",
            one_line_content="grid shortest path graph traversal",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["dfs:premature-termination"],
            root_cause="returns early before exploring all reachable branches",
            ability_dimensions=["reasoning", "graph_search"],
            language="python",
            category="algorithms",
            one_line_content="grid shortest path graph traversal",
        ),
    ]

    candidates = propose_candidate_clusters(units)

    candidate_tags = sorted(sorted(candidate.member_tags) for candidate in candidates)
    assert ["dfs:premature-termination"] in candidate_tags
    assert ["graph:state-update-order"] in candidate_tags


def test_tag_similarity_components_expose_bridge_and_penalty_flags():
    bridge_units = [
        ClusterUnit(
            question_id=1,
            error_tags=["recursion:state-leak"],
            root_cause="shared mutable state leaks across recursive frames",
            ability_dimensions=["reasoning", "state_management"],
            language="python",
            category="algorithms",
            one_line_content="recursive search with shared path state",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["backtracking:shared-mutable-state"],
            root_cause="shared mutable state leaks across search branches",
            ability_dimensions=["reasoning", "state_management"],
            language="python",
            category="algorithms",
            one_line_content="backtracking search with shared path state",
        ),
    ]
    bridge_stats = _build_tag_stats(bridge_units)
    bridge_components = _tag_similarity_components(
        "recursion:state-leak",
        "backtracking:shared-mutable-state",
        bridge_stats,
    )
    assert bridge_components["bridge_applied"] is True
    assert bridge_components["same_prefix_penalty_applied"] is False

    penalty_units = [
        ClusterUnit(
            question_id=1,
            error_tags=["dp:transition-missing"],
            root_cause="omits recurrence transition case",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="dynamic programming over prefixes",
        ),
        ClusterUnit(
            question_id=2,
            error_tags=["dp:memory-overflow"],
            root_cause="allocates quadratic table without compression",
            ability_dimensions=["reasoning"],
            language="python",
            category="algorithms",
            one_line_content="memory-heavy dynamic programming table",
        ),
    ]
    penalty_stats = _build_tag_stats(penalty_units)
    penalty_components = _tag_similarity_components(
        "dp:transition-missing",
        "dp:memory-overflow",
        penalty_stats,
    )
    assert penalty_components["bridge_applied"] is False
    assert penalty_components["same_prefix_penalty_applied"] is True


@pytest.mark.asyncio
async def test_refine_candidate_clusters_updates_progress_and_preserves_members():
    candidate = make_candidate(
        "C001",
        ["recursion:base-case-missing", "recursion:termination-condition-missing"],
        [1, 2],
    )
    client = FakeProvider(
        outputs=[
            '[{"id":"R001","name":"Recursion termination","description":"shared weakness","covered_tags":["recursion:base-case-missing","recursion:termination-condition-missing"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )
    progress = ProgressSpy()

    refined = await refine_candidate_clusters(
        [candidate],
        provider="openai",
        model="test-model",
        provider_client=client,
        progress=progress,
    )

    assert len(refined) == 1
    assert refined[0].member_question_ids == [1, 2]
    assert refined[0].covered_tags == [
        "recursion:base-case-missing",
        "recursion:termination-condition-missing",
    ]
    assert progress.updates == [1]


@pytest.mark.asyncio
async def test_merge_refined_clusters_updates_progress_and_merges_neighbors():
    refined_clusters = [
        make_refined("R001", ["recursion:base-case-missing"], [1], "Recursion base case"),
        make_refined("R002", ["recursion:termination-condition-missing"], [2], "Recursion termination"),
        make_refined("R003", ["greedy:wrong-local-choice"], [3], "Greedy shortcut", root_cause="uses greedy shortcut"),
    ]
    client = FakeProvider(
        outputs=[
            '{"should_merge":true,"merged_weakness":{"id":"W999","name":"Recursion termination","description":"shared recursion weakness","covered_tags":["recursion:base-case-missing","recursion:termination-condition-missing"],"dominant_language":"python","dominant_category":"algorithms"}}',
            '{"should_merge":false,"merged_weakness":null}',
        ]
    )
    progress = ProgressSpy()

    merged = await merge_refined_clusters(
        refined_clusters,
        provider="openai",
        model="test-model",
        provider_client=client,
        progress_factory=lambda **_: progress,
    )

    assert len(merged) == 2
    names = sorted(item.name for item in merged)
    assert names == ["Greedy shortcut", "Recursion termination"]
    assert sum(progress.updates) >= 1
    assert progress.closed is True
