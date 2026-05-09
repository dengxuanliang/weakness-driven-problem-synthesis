import pytest
import asyncio
from pathlib import Path
from pydantic import ValidationError

from weakness_driven_problem_synthesis.attribute import attribute_failures
from weakness_driven_problem_synthesis.llm_client import (
    _openai_completion_mode,
    _load_env_file_if_needed,
    build_provider_client,
    complete_json,
)
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


class SimultaneousCompletionProvider:
    def __init__(self):
        self.calls = []
        self._started = {
            1: asyncio.Event(),
            2: asyncio.Event(),
            3: asyncio.Event(),
        }
        self._release_first_wave = asyncio.Event()

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
        self._started[question_id].set()
        if question_id in (1, 2):
            await self._release_first_wave.wait()
        else:
            assert self._started[1].is_set()
            assert self._started[2].is_set()
        return {
            "question_id": question_id,
            "is_truly_failed": True,
            "error_tags": [f"tag:{question_id}"],
            "root_cause": f"root {question_id}",
            "ability_dimensions": [f"ability {question_id}"],
            "evidence_snippet": f"evidence {question_id}",
        }


class FakeOpenAIChatCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Message:
            def __init__(self, content: str):
                self.content = content

        class _Choice:
            def __init__(self, content: str):
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str):
                self.choices = [_Choice(content)]

        return _Response(self.content)


class FakeOpenAIClient:
    def __init__(self, content: str):
        self.chat = type("ChatNamespace", (), {"completions": FakeOpenAIChatCompletions(content)})()


class RecordingProgressBar:
    def __init__(self, total=None, initial=0, desc=None, unit=None):
        self.total = total
        self.initial = initial
        self.desc = desc
        self.unit = unit
        self.updates = []
        self.closed = False

    def update(self, value):
        self.updates.append(value)

    def close(self):
        self.closed = True


def make_progress_factory():
    holder = {}

    def factory(**kwargs):
        progress = RecordingProgressBar(**kwargs)
        holder["progress"] = progress
        return progress

    return holder, factory


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


@pytest.mark.asyncio
async def test_complete_json_includes_last_raw_output_preview_when_retries_exhausted():
    client = FakeProvider(outputs=["not json", "still not json", "final not json"])

    with pytest.raises(ValueError, match=r"final not json"):
        await complete_json(
            "prompt",
            {"type": "object"},
            provider_client=client,
            model="test-model",
        )


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_provider_client(provider="openai", model=None)


def test_load_env_file_if_needed_does_not_override_existing_environment(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from_dotenv\nOPENAI_BASE_URL=https://dotenv.example\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from_env")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _load_env_file_if_needed(env_path=env_path)

    assert Path(env_path).exists()
    assert __import__("os").environ["OPENAI_API_KEY"] == "from_env"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://dotenv.example"


def test_load_env_file_if_needed_uses_dotenv_as_fallback(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from_dotenv\nOPENAI_BASE_URL=https://dotenv.example\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _load_env_file_if_needed(env_path=env_path)

    assert __import__("os").environ["OPENAI_API_KEY"] == "from_dotenv"
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://dotenv.example"


def test_missing_api_key_still_fails_when_env_and_dotenv_are_both_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "weakness_driven_problem_synthesis.llm_client._repo_root",
        lambda: tmp_path,
    )

    with pytest.raises(RuntimeError, match="Missing required environment variable: OPENAI_API_KEY"):
        build_provider_client(provider="openai", model=None)


def test_attribution_schema_rejects_scalar_ability_dimensions_without_normalization():
    with pytest.raises(ValidationError):
        Attribution.model_validate(
            {
                "question_id": 1,
                "is_truly_failed": True,
                "error_tags": ["edge-case:null"],
                "root_cause": "r",
                "ability_dimensions": "generalization",
                "evidence_snippet": "e",
            }
        )


@pytest.mark.asyncio
async def test_attribute_failures_normalizes_scalar_list_fields_from_model(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 1, "is_truly_failed": true, "error_tags": "edge-case:null", "root_cause": "missed null", "ability_dimensions": "generalization", "evidence_snippet": "value is None"}',
        ]
    )

    result = await attribute_failures(
        [make_eval_record(1, "first")],
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    assert result[0].error_tags == ["edge-case:null"]
    assert result[0].ability_dimensions == ["generalization"]


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


def test_eval_record_accepts_real_log_test_payload_as_dict():
    record = EvalRecord.model_validate(
        {
            "question_id": 1,
            "content": "problem",
            "canonical_solution": "def solve(): pass",
            "completion": "def solve(): return None",
            "test": {"code": "assert True"},
            "labels": {
                "category": "algorithms",
                "programming_language": "python",
                "difficulty": "hard",
            },
            "pass_at_1": 0,
        }
    )

    assert isinstance(record.test, dict)
    assert record.test["code"] == "assert True"


def test_openai_completion_mode_uses_json_object_for_object_schema():
    assert _openai_completion_mode({"type": "object"}) == "json_object"


def test_openai_completion_mode_uses_plain_text_array_for_array_schema():
    assert _openai_completion_mode({"type": "array"}) == "plain_text_array"


@pytest.mark.asyncio
async def test_complete_json_uses_openai_chat_completions_json_mode():
    provider_client = type(
        "ProviderStub",
        (),
        {
            "model": "test-model",
            "complete_json": None,
        },
    )()
    chat_client = FakeOpenAIClient('{"ok": true}')

    from weakness_driven_problem_synthesis.llm_client import OpenAIProviderClient

    client = OpenAIProviderClient(provider="openai", model="test-model", client=chat_client)
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )

    assert result == {"ok": True}
    call = chat_client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][-1]["content"] == "prompt"


@pytest.mark.asyncio
async def test_complete_json_uses_plain_text_mode_for_openai_array_schema():
    chat_client = FakeOpenAIClient('[{"id":"W1","name":"n","description":"d","covered_tags":["t"],"dominant_language":"python","dominant_category":"algorithms"}]')

    from weakness_driven_problem_synthesis.llm_client import OpenAIProviderClient

    client = OpenAIProviderClient(provider="openai", model="test-model", client=chat_client)
    result = await complete_json(
        "prompt",
        {"type": "array"},
        provider_client=client,
        model="test-model",
    )

    assert isinstance(result, list)
    call = chat_client.chat.completions.calls[0]
    assert "response_format" not in call
    assert "JSON array" in call["messages"][0]["content"]


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
async def test_attribute_failures_updates_progress_for_new_records(tmp_path, monkeypatch):
    output_path = tmp_path / "error_attributions.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 1, "is_truly_failed": true, "error_tags": ["tag:1"], "root_cause": "r1", "ability_dimensions": ["a1"], "evidence_snippet": "e1"}',
            '{"question_id": 2, "is_truly_failed": true, "error_tags": ["tag:2"], "root_cause": "r2", "ability_dimensions": ["a2"], "evidence_snippet": "e2"}',
        ]
    )
    holder, factory = make_progress_factory()
    monkeypatch.setattr("weakness_driven_problem_synthesis.attribute._build_progress_bar", factory)

    await attribute_failures(
        [make_eval_record(1, "first"), make_eval_record(2, "second")],
        output_path=output_path,
        provider="openai",
        model="test-model",
        concurrency=2,
        provider_client=client,
    )

    progress = holder["progress"]
    assert progress.total == 2
    assert progress.initial == 0
    assert progress.updates == [1, 1]
    assert progress.closed is True


@pytest.mark.asyncio
async def test_attribute_failures_progress_initializes_from_existing_records(tmp_path, monkeypatch):
    output_path = tmp_path / "error_attributions.jsonl"
    output_path.write_text(
        '{"question_id": 7, "is_truly_failed": true, "error_tags": ["x:y"], "root_cause": "r", "ability_dimensions": ["a"], "evidence_snippet": "e"}\n'
    )
    client = FakeProvider(
        outputs=[
            '{"question_id": 8, "is_truly_failed": true, "error_tags": ["tag:8"], "root_cause": "r8", "ability_dimensions": ["a8"], "evidence_snippet": "e8"}',
        ]
    )
    holder, factory = make_progress_factory()
    monkeypatch.setattr("weakness_driven_problem_synthesis.attribute._build_progress_bar", factory)

    await attribute_failures(
        [make_eval_record(7, "skip me"), make_eval_record(8, "process me")],
        output_path=output_path,
        provider="openai",
        model="test-model",
        concurrency=1,
        provider_client=client,
    )

    progress = holder["progress"]
    assert progress.total == 2
    assert progress.initial == 1
    assert progress.updates == [1]


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
async def test_attribute_failures_truncates_oversized_test_payload_in_prompt(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 9, "is_truly_failed": true, "error_tags": ["tag:9"], "root_cause": "r9", "ability_dimensions": ["a9"], "evidence_snippet": "e9"}',
        ]
    )
    record = EvalRecord.model_validate(
        {
            "question_id": 9,
            "content": "problem",
            "canonical_solution": "def solve(): pass",
            "completion": "def solve(): return None",
            "test": "x" * 200_000,
            "labels": {
                "category": "algorithms",
                "programming_language": "python",
                "difficulty": "hard",
            },
            "pass_at_1": 0,
        }
    )

    await attribute_failures(
        [record],
        output_path=output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    prompt = client.calls[0]["prompt"]
    assert "Test:\n" in prompt
    assert "[truncated " in prompt
    assert " chars]" in prompt
    assert len(prompt) < 100_000


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


@pytest.mark.asyncio
async def test_attribute_failures_merges_all_completed_tags_before_refilling_concurrency(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    client = SimultaneousCompletionProvider()
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

    await client._started[1].wait()
    await client._started[2].wait()
    client._release_first_wave.set()

    while len(client.calls) < 3:
        await asyncio.sleep(0)

    prompt_by_id = {call["question_id"]: call["prompt"] for call in client.calls}
    assert "tag:1" in prompt_by_id[3]
    assert "tag:2" in prompt_by_id[3]

    await task
