import pytest

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
