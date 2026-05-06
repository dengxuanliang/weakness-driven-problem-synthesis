import pytest

from weakness_driven_problem_synthesis.cluster import (
    cluster_weaknesses,
    map_questions_to_clusters,
)
from weakness_driven_problem_synthesis.schemas import Attribution, Weakness


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


@pytest.mark.asyncio
async def test_cluster_weaknesses_writes_resume_artifact(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    attributions = [
        make_attribution(1, ["recursion:base-case-missing"]),
        make_attribution(2, ["edge-case:empty-input"]),
    ]
    client = FakeProvider(
        outputs=[
            '[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"},{"id":"W002","name":"Empty input handling","description":"edge input bugs","covered_tags":["edge-case:empty-input"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )

    result = await cluster_weaknesses(
        attributions,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert output_path.exists()
    assert result.weaknesses[0].id == "W001"
    reloaded = await cluster_weaknesses(
        attributions,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )
    assert reloaded == result
    assert len(client.calls) == 1


def test_map_questions_to_clusters_counts_multi_cluster_membership():
    attributions = [
        make_attribution(1, ["recursion:base-case-missing"]),
        make_attribution(3, ["recursion:base-case-missing", "edge-case:empty-input"]),
    ]
    weaknesses = [
        Weakness.model_validate(
            {
                "id": "W001",
                "name": "Recursion termination",
                "description": "recursion bugs",
                "covered_tags": ["recursion:base-case-missing"],
                "dominant_language": "python",
                "dominant_category": "algorithms",
            }
        ),
        Weakness.model_validate(
            {
                "id": "W002",
                "name": "Empty input handling",
                "description": "edge input bugs",
                "covered_tags": ["edge-case:empty-input"],
                "dominant_language": "python",
                "dominant_category": "algorithms",
            }
        ),
    ]

    mapping = map_questions_to_clusters(attributions, weaknesses)
    assert mapping["W001"] == [1, 3]
    assert mapping["W002"] == [3]
