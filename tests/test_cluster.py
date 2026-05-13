import pytest

from weakness_driven_problem_synthesis.cluster import (
    CLUSTER_PROMPT_MAX_CHARS,
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


@pytest.mark.asyncio
async def test_cluster_weaknesses_rejects_non_array_payload_with_clear_error(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    client = FakeProvider(
        outputs=[
            {"weaknesses": []}
        ]
    )

    with pytest.raises(ValueError, match=r"cluster_weaknesses expected JSON array payload"):
        await cluster_weaknesses(
            [make_attribution(1, ["recursion:base-case-missing"])],
            eval_records=[make_eval_record(1, "recursive traversal on nested arrays")],
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=client,
        )


@pytest.mark.asyncio
async def test_cluster_weaknesses_rejects_empty_payload_when_attributions_exist(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    client = FakeProvider(outputs=[[]])

    with pytest.raises(ValueError, match=r"cluster_weaknesses expected non-empty JSON array payload"):
        await cluster_weaknesses(
            [make_attribution(1, ["recursion:base-case-missing"])],
            eval_records=[make_eval_record(1, "recursive traversal on nested arrays")],
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=client,
        )


@pytest.mark.asyncio
async def test_cluster_weaknesses_allows_empty_payload_when_attributions_are_empty(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    client = FakeProvider(outputs=[[]])

    result = await cluster_weaknesses(
        [],
        eval_records=[],
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.weaknesses == []
    assert result.evidence_question_ids == {}
    assert output_path.exists()


@pytest.mark.asyncio
async def test_cluster_weaknesses_rejects_empty_resumed_artifact_when_attributions_exist(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    output_path.write_text('{"weaknesses":[],"evidence_question_ids":{}}')

    with pytest.raises(ValueError, match=r"cluster_weaknesses expected non-empty JSON array payload"):
        await cluster_weaknesses(
            [make_attribution(1, ["recursion:base-case-missing"])],
            eval_records=[make_eval_record(1, "recursive traversal on nested arrays")],
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=FakeProvider(outputs=[]),
        )


@pytest.mark.asyncio
async def test_cluster_weaknesses_allows_empty_resumed_artifact_when_attributions_are_empty(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    output_path.write_text('{"weaknesses":[],"evidence_question_ids":{}}')

    result = await cluster_weaknesses(
        [],
        eval_records=[],
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=FakeProvider(outputs=[]),
    )

    assert result.weaknesses == []
    assert result.evidence_question_ids == {}


@pytest.mark.asyncio
async def test_cluster_weaknesses_uses_single_call_for_small_inputs(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    client = FakeProvider(
        outputs=[
            '[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}]'
        ]
    )

    result = await cluster_weaknesses(
        [make_attribution(1, ["recursion:base-case-missing"])],
        eval_records=[make_eval_record(1, "small case")],
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert len(client.calls) == 1
    assert result.weaknesses[0].id == "W001"


@pytest.mark.asyncio
async def test_cluster_weaknesses_chunks_large_inputs_and_keeps_each_prompt_under_budget(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    tag_prefix = "t" * 7_000
    attributions = [make_attribution(index, [f"{tag_prefix}:{index}"]) for index in range(1, 7)]
    eval_records = [make_eval_record(index, "small case") for index in range(1, 7)]
    client = FakeProvider(
        outputs=[
            f'[{{"id":"WA","name":"Chunk A","description":"d","covered_tags":["{tag_prefix}:1","{tag_prefix}:2","{tag_prefix}:3"],"dominant_language":"python","dominant_category":"algorithms"}}]',
            f'[{{"id":"WB","name":"Chunk B","description":"d","covered_tags":["{tag_prefix}:4","{tag_prefix}:5","{tag_prefix}:6"],"dominant_language":"python","dominant_category":"algorithms"}}]',
            f'[{{"id":"W999","name":"Merged weakness","description":"merged","covered_tags":["{tag_prefix}:1","{tag_prefix}:2","{tag_prefix}:3","{tag_prefix}:4","{tag_prefix}:5","{tag_prefix}:6"],"dominant_language":"python","dominant_category":"algorithms"}}]',
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

    assert len(client.calls) == 3
    assert all(len(call["prompt"]) <= CLUSTER_PROMPT_MAX_CHARS for call in client.calls)
    assert [weakness.id for weakness in result.weaknesses] == ["W001"]
    assert result.weaknesses[0].name == "Merged weakness"
    assert result.evidence_question_ids["W001"] == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_cluster_weaknesses_includes_covered_tags_in_merge_prompt_under_budget(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    tag_prefix = "t" * 7_000
    attributions = [make_attribution(index, [f"{tag_prefix}:{index}"]) for index in range(1, 7)]
    eval_records = [make_eval_record(index, "small case") for index in range(1, 7)]
    client = FakeProvider(
        outputs=[
            f'[{{"id":"WA","name":"Chunk A","description":"d","covered_tags":["{tag_prefix}:1","{tag_prefix}:2","{tag_prefix}:3"],"dominant_language":"python","dominant_category":"algorithms"}}]',
            f'[{{"id":"WB","name":"Chunk B","description":"d","covered_tags":["{tag_prefix}:4","{tag_prefix}:5","{tag_prefix}:6"],"dominant_language":"python","dominant_category":"algorithms"}}]',
            f'[{{"id":"W999","name":"Merged weakness","description":"merged","covered_tags":["{tag_prefix}:1","{tag_prefix}:2","{tag_prefix}:3","{tag_prefix}:4","{tag_prefix}:5","{tag_prefix}:6"],"dominant_language":"python","dominant_category":"algorithms"}}]',
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

    merge_prompt = client.calls[-1]["prompt"]
    assert '"covered_tags"' in merge_prompt
    assert f"{tag_prefix}:1" in merge_prompt
    assert len(merge_prompt) <= CLUSTER_PROMPT_MAX_CHARS


@pytest.mark.asyncio
async def test_cluster_weaknesses_fails_fast_when_single_block_exceeds_budget(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    huge_tag = "t" * 30_000
    attribution = make_attribution(1, [huge_tag])
    eval_record = make_eval_record(1, "small case")

    with pytest.raises(ValueError, match="single tag summary block exceeds cluster prompt budget"):
        await cluster_weaknesses(
            [attribution],
            eval_records=[eval_record],
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=FakeProvider(outputs=[]),
        )


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
