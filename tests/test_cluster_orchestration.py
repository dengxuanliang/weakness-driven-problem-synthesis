import pytest

from weakness_driven_problem_synthesis.cluster import cluster_weaknesses
from weakness_driven_problem_synthesis.cluster_types import CandidateCluster, ClusterUnit, RefinedCluster
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord, Weakness


class ProgressSpy:
    def __init__(self):
        self.updates = []
        self.closed = False

    def update(self, value: int) -> None:
        self.updates.append(value)

    def close(self) -> None:
        self.closed = True


def make_attribution(question_id: int, error_tags: list[str]) -> Attribution:
    return Attribution.model_validate(
        {
            "question_id": question_id,
            "is_truly_failed": True,
            "error_tags": error_tags,
            "root_cause": "root cause",
            "ability_dimensions": ["reasoning"],
            "evidence_snippet": "snippet",
        }
    )


def make_eval_record(question_id: int, content: str) -> EvalRecord:
    return EvalRecord.model_validate(
        {
            "question_id": question_id,
            "content": content,
            "canonical_solution": "def solve(): pass",
            "completion": "def solve(): return None",
            "test": "assert True",
            "labels": {
                "category": "algorithms",
                "programming_language": "python",
                "difficulty": "hard",
            },
            "pass_at_1": 0,
        }
    )


@pytest.mark.asyncio
async def test_cluster_weaknesses_uses_candidate_pipeline_for_large_tag_sets(tmp_path, monkeypatch):
    output_path = tmp_path / "weaknesses.json"
    calls = []

    def fake_build_cluster_units(*, attributions, eval_records):
        calls.append("units")
        return [
            ClusterUnit(
                question_id=1,
                error_tags=["recursion:base-case-missing"],
                root_cause="root cause",
                ability_dimensions=["reasoning"],
                language="python",
                category="algorithms",
                one_line_content="case 1",
            )
        ]

    def fake_propose_candidate_clusters(units):
        calls.append("precluster")
        return [
            CandidateCluster(
                candidate_id="C001",
                member_question_ids=[1, 2, 3],
                member_tags=["recursion:base-case-missing", "recursion:termination-condition-missing"],
                representative_units=units,
                dominant_language="python",
                dominant_category="algorithms",
            )
        ]

    async def fake_refine_candidate_clusters(candidates, **kwargs):
        calls.append("refine")
        progress = kwargs["progress"]
        progress.update(1)
        return [
            RefinedCluster(
                refined_id="R001",
                name="Recursion termination",
                description="shared weakness",
                covered_tags=["recursion:base-case-missing", "recursion:termination-condition-missing"],
                member_question_ids=[1, 2, 3],
                representative_units=candidates[0].representative_units,
                dominant_language="python",
                dominant_category="algorithms",
            )
        ]

    async def fake_merge_refined_clusters(refined_clusters, **kwargs):
        calls.append("merge")
        return [
            Weakness.model_validate(
                {
                    "id": "W777",
                    "name": "Recursion termination",
                    "description": "shared weakness",
                    "covered_tags": ["recursion:base-case-missing", "recursion:termination-condition-missing"],
                    "dominant_language": "python",
                    "dominant_category": "algorithms",
                }
            )
        ]

    progress_bars = []

    def fake_build_progress_bar(**kwargs):
        progress = ProgressSpy()
        progress_bars.append((kwargs["desc"], progress))
        return progress

    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster.LARGE_INPUT_TAG_THRESHOLD", 2)
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster.build_cluster_units", fake_build_cluster_units)
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster.propose_candidate_clusters", fake_propose_candidate_clusters)
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster.refine_candidate_clusters", fake_refine_candidate_clusters)
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster.merge_refined_clusters", fake_merge_refined_clusters)
    monkeypatch.setattr("weakness_driven_problem_synthesis.cluster._build_progress_bar", fake_build_progress_bar)

    attributions = [
        make_attribution(1, ["recursion:base-case-missing"]),
        make_attribution(2, ["recursion:termination-condition-missing"]),
        make_attribution(3, ["greedy:wrong-local-choice"]),
    ]
    eval_records = [
        make_eval_record(1, "case 1"),
        make_eval_record(2, "case 2"),
        make_eval_record(3, "case 3"),
    ]

    result = await cluster_weaknesses(
        attributions,
        eval_records=eval_records,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=object(),
    )

    assert calls == ["units", "precluster", "refine", "merge"]
    assert [item.id for item in result.weaknesses] == ["W001"]
    assert result.evidence_question_ids["W001"] == [1, 2]
    assert [desc for desc, _ in progress_bars] == ["Cluster refine"]
    assert progress_bars[0][1].updates == [1]
