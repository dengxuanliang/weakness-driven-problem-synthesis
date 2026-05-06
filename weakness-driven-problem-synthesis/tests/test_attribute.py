import pytest
import asyncio

from weakness_driven_problem_synthesis.attribute import attribute_failures
from weakness_driven_problem_synthesis.llm_client import (
    build_provider_client,
    complete_json,
)
from weakness_driven_problem_synthesis.schemas import EvalRecord


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


class ControlledProvider:
    def __init__(self):
        self.calls = []
        self._gates = {
            1: asyncio.Event(),
            2: asyncio.Event(),
            3: asyncio.Event(),
        }

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        question_id_line = next(line for line in prompt.splitlines() if line.startswith("Question ID:"))
        question_id = int(question_id_line.split(":", 1)[1].strip())
        self.calls.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        if question_id == 3:
            assert self._gates[1].is_set()
        await self._gates[question_id].wait()
        return {
            "question_id": question_id,
            "is_truly_failed": True,
            "error_tags": [f"tag:{question_id}"],
            "root_cause": f"root {question_id}",
            "ability_dimensions": [f"ability {question_id}"],
            "evidence_snippet": f"evidence {question_id}",
        }


@pytest.mark.asyncio
async def test_complete_json_retries_invalid_json():
    client = FakeProvider(outputs=["not json", '{"ok": true}'])
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )
    assert result == {"ok": True}
    assert len(client.calls) == 2


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_provider_client(provider="openai", model=None)


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
async def test_attribute_failures_appends_one_json_line_per_record(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 1, "is_truly_failed": true, "error_tags": ["api-misuse:parser"], "root_cause": "wrong parser", "ability_dimensions": ["library API details"], "evidence_snippet": "parse(x)"}',
            '{"question_id": 2, "is_truly_failed": true, "error_tags": ["edge-case:empty-input"], "root_cause": "missed empty input", "ability_dimensions": ["edge case handling"], "evidence_snippet": "if not data"}',
        ]
    )
    records = [make_eval_record(1, "first"), make_eval_record(2, "second")]

    result = await attribute_failures(
        records,
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=2,
    )

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert [item.question_id for item in result] == [1, 2]


@pytest.mark.asyncio
async def test_attribute_failures_skips_already_processed_question_ids(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    output_path.write_text(
        '{"question_id": 7, "is_truly_failed": true, "error_tags": ["x:y"], "root_cause": "r", "ability_dimensions": ["a"], "evidence_snippet": "e"}\n'
    )
    client = FakeProvider(
        outputs=[
            '{"question_id": 8, "is_truly_failed": true, "error_tags": ["edge-case:null"], "root_cause": "missed null", "ability_dimensions": ["edge case handling"], "evidence_snippet": "value is None"}',
        ]
    )
    records = [make_eval_record(7, "skip me"), make_eval_record(8, "process me")]

    result = await attribute_failures(
        records,
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=2,
    )

    assert [item.question_id for item in result] == [7, 8]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_attribute_failures_includes_existing_seen_tags_in_prompt(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    output_path.write_text(
        '{"question_id": 7, "is_truly_failed": true, "error_tags": ["recursion:base-case-missing"], "root_cause": "r", "ability_dimensions": ["a"], "evidence_snippet": "e"}\n'
    )
    client = FakeProvider(
        outputs=[
            '{"question_id": 8, "is_truly_failed": true, "error_tags": ["edge-case:null"], "root_cause": "missed null", "ability_dimensions": ["edge case handling"], "evidence_snippet": "value is None"}',
        ]
    )

    await attribute_failures(
        [make_eval_record(8, "process me")],
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    assert "Seen tags:" in client.calls[0]["prompt"]
    assert "recursion:base-case-missing" in client.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_attribute_failures_updates_seen_tags_during_same_run(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 1, "is_truly_failed": true, "error_tags": ["recursion:base-case-missing"], "root_cause": "r1", "ability_dimensions": ["a1"], "evidence_snippet": "e1"}',
            '{"question_id": 2, "is_truly_failed": true, "error_tags": ["recursion:base-case-missing"], "root_cause": "r2", "ability_dimensions": ["a2"], "evidence_snippet": "e2"}',
        ]
    )

    await attribute_failures(
        [make_eval_record(1, "first"), make_eval_record(2, "second")],
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    assert "Seen tags:" in client.calls[1]["prompt"]
    assert "recursion:base-case-missing" in client.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_attribute_failures_uses_latest_completed_seen_tags_when_refilling_concurrency(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = ControlledProvider()
    records = [
        make_eval_record(1, "first"),
        make_eval_record(2, "second"),
        make_eval_record(3, "third"),
    ]

    task = asyncio.create_task(
        attribute_failures(
            records,
            output_path=output_path,
            provider_client=client,
            provider="openai",
            model="test-model",
            concurrency=2,
        )
    )

    while len(client.calls) < 2:
        await asyncio.sleep(0)

    prompt_by_id = {call["question_id"]: call["prompt"] for call in client.calls}
    assert "tag:1" not in prompt_by_id[2]

    client._gates[1].set()

    while len(client.calls) < 3:
        await asyncio.sleep(0)

    prompt_by_id = {call["question_id"]: call["prompt"] for call in client.calls}
    assert "tag:1" in prompt_by_id[3]

    client._gates[2].set()
    client._gates[3].set()
    await task
