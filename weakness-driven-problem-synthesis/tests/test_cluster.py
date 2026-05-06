import pytest

from weakness_driven_problem_synthesis.cluster import (
    cluster_weaknesses,
    map_questions_to_clusters,
)
from weakness_driven_problem_synthesis.schemas import Attribution, EvalRecord, Weakness


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


@pytest.mark.asyncio
async def test_cluster_weaknesses_writes_resume_artifact(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    attributions = [
        make_attribution(1, ["recursion:base-case-missing"]),
        make_attribution(2, ["edge-case:empty-input"]),
    ]
    eval_records = [
        make_eval_record(1, "recursive traversal on nested arrays"),
        make_eval_record(2, "handle null records in event stream"),
    ]
    client = FakeProvider(
        outputs=[
            '[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"},{"id":"W002","name":"Empty input handling","description":"edge input bugs","covered_tags":["edge-case:empty-input"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )

    result = await cluster_weaknesses(
        attributions,
        eval_records=eval_records,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert output_path.exists()
    assert result.weaknesses[0].id == "W001"
    reloaded = await cluster_weaknesses(
        attributions,
        eval_records=eval_records,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )
    assert reloaded == result
    assert len(client.calls) == 1
    prompt = client.calls[0]["prompt"]
    assert "Representative question summaries" in prompt
    assert "recursion:base-case-missing" in prompt
    assert "edge-case:empty-input" in prompt
    assert "category" in prompt
    assert "language" in prompt
    assert "one_line_content" in prompt


@pytest.mark.asyncio
async def test_cluster_weaknesses_deduplicates_tags_and_limits_representatives(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    attributions = [
        make_attribution(1, ["recursion:base-case-missing"]),
        make_attribution(2, ["recursion:base-case-missing"]),
        make_attribution(3, ["recursion:base-case-missing"]),
        make_attribution(4, ["recursion:base-case-missing"]),
    ]
    eval_records = [
        make_eval_record(1, "case one"),
        make_eval_record(2, "case two"),
        make_eval_record(3, "case three"),
        make_eval_record(4, "case four"),
    ]
    client = FakeProvider(
        outputs=[
            '[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )

    await cluster_weaknesses(
        attributions,
        eval_records=eval_records,
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    prompt = client.calls[0]["prompt"]
    assert prompt.count("recursion:base-case-missing") == 1
    assert prompt.count("'id':") <= 3


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
