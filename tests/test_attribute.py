import pytest
import asyncio
import sys
import types
import httpx
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


@pytest.mark.asyncio
async def test_complete_json_accepts_markdown_fenced_object_json():
    client = FakeProvider(outputs=['```json\n{"ok": true}\n```'])
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_complete_json_accepts_markdown_fenced_array_json():
    client = FakeProvider(
        outputs=[
            '```\n[{"id":"W1","name":"n","description":"d","covered_tags":["t"],"dominant_language":"python","dominant_category":"algorithms"}]\n```'
        ]
    )
    result = await complete_json(
        "prompt",
        {"type": "array"},
        provider_client=client,
        model="test-model",
    )
    assert isinstance(result, list)
    assert result[0]["id"] == "W1"


@pytest.mark.asyncio
async def test_complete_json_retries_retryable_html_403_gateway_page():
    class Retryable403Provider:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "403 PermissionDenied <html><title>Access Denied</title><body>firewall blocked</body></html>"
                )
            return '{"ok": true}'

    client = Retryable403Provider()
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )

    assert result == {"ok": True}
    assert client.calls == 2


@pytest.mark.asyncio
async def test_complete_json_does_not_retry_non_retryable_403_auth_error():
    class Auth403Provider:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            self.calls += 1
            raise RuntimeError("403 PermissionDenied invalid api key")

    client = Auth403Provider()

    with pytest.raises(RuntimeError, match="invalid api key"):
        await complete_json(
            "prompt",
            {"type": "object"},
            provider_client=client,
            model="test-model",
        )

    assert client.calls == 1


@pytest.mark.asyncio
async def test_complete_json_retries_403_html_from_response_object(monkeypatch):
    class _Response:
        def __init__(self):
            self.text = "<html><body>Access Denied</body></html>"
            self.content = b"<html><body>Access Denied</body></html>"

    class Response403Provider:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            self.calls += 1
            if self.calls == 1:
                exc = RuntimeError("403 PermissionDenied")
                exc.response = _Response()
                raise exc
            return '{"ok": true}'

    client = Response403Provider()
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )

    assert result == {"ok": True}
    assert client.calls == 2


@pytest.mark.asyncio
async def test_complete_json_retries_403_from_status_code_attribute():
    class Status403Provider:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            self.calls += 1
            if self.calls == 1:
                exc = RuntimeError("gateway blocked")
                exc.status_code = 403
                exc.response = type("Resp", (), {"text": "<html><body>Forbidden</body></html>"})()
                raise exc
            return '{"ok": true}'

    client = Status403Provider()
    result = await complete_json(
        "prompt",
        {"type": "object"},
        provider_client=client,
        model="test-model",
    )

    assert result == {"ok": True}
    assert client.calls == 2


@pytest.mark.asyncio
async def test_complete_json_global_throttler_limits_concurrent_provider_calls(monkeypatch):
    starts: list[int] = []
    finishes: list[int] = []
    current_in_flight = 0
    max_in_flight = 0

    class ConcurrencyProbeProvider:
        def __init__(self):
            self.calls = 0

        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            nonlocal current_in_flight, max_in_flight
            self.calls += 1
            starts.append(self.calls)
            current_in_flight += 1
            max_in_flight = max(max_in_flight, current_in_flight)
            await asyncio.sleep(0)
            current_in_flight -= 1
            finishes.append(self.calls)
            return '{"ok": true}'

    monkeypatch.setattr("weakness_driven_problem_synthesis.llm_client._REQUEST_THROTTLER", None)
    monkeypatch.setenv("WEAKNESS_SYNTH_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("WEAKNESS_SYNTH_MIN_INTERVAL_MS", "0")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_LIMIT", "100")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_COOLDOWN_MS", "0")

    client = ConcurrencyProbeProvider()
    await asyncio.gather(
        complete_json("p1", {"type": "object"}, provider_client=client, model="test-model"),
        complete_json("p2", {"type": "object"}, provider_client=client, model="test-model"),
        complete_json("p3", {"type": "object"}, provider_client=client, model="test-model"),
    )

    assert max_in_flight == 1
    assert starts == [1, 2, 3]
    assert finishes == [1, 2, 3]


@pytest.mark.asyncio
async def test_request_throttler_honors_burst_cooldown(monkeypatch):
    import weakness_driven_problem_synthesis.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_REQUEST_THROTTLER", None)
    monkeypatch.setenv("WEAKNESS_SYNTH_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("WEAKNESS_SYNTH_MIN_INTERVAL_MS", "0")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_LIMIT", "1")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_COOLDOWN_MS", "200")

    throttler = llm_client._get_request_throttler()
    wait_times: list[float] = []
    clock = {"now": 0.0}

    class FakeLoop:
        def time(self):
            return clock["now"]

    async def fake_sleep(seconds):
        wait_times.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr("weakness_driven_problem_synthesis.llm_client.asyncio.get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr("weakness_driven_problem_synthesis.llm_client.asyncio.sleep", fake_sleep)

    async with throttler:
        pass
    clock["now"] = 0.01
    async with throttler:
        pass

    assert wait_times == [0.2]


def test_request_throttler_reads_env_configuration(monkeypatch):
    monkeypatch.setenv("WEAKNESS_SYNTH_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("WEAKNESS_SYNTH_MIN_INTERVAL_MS", "250")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_LIMIT", "7")
    monkeypatch.setenv("WEAKNESS_SYNTH_BURST_COOLDOWN_MS", "900")

    from weakness_driven_problem_synthesis.llm_client import _REQUEST_THROTTLER, _get_request_throttler

    monkeypatch.setattr("weakness_driven_problem_synthesis.llm_client._REQUEST_THROTTLER", None)
    throttler = _get_request_throttler()
    assert throttler.max_in_flight == 1
    assert throttler.min_interval_seconds == 0.25
    assert throttler.burst_limit == 7
    assert throttler.burst_cooldown_seconds == 0.9


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_provider_client(provider="openai", model=None)


def test_load_env_file_if_needed_reads_from_project_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=from_dotenv\nOPENAI_BASE_URL=https://dotenv.example\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = _load_env_file_if_needed(env_path=env_path)

    assert config["OPENAI_API_KEY"] == "from_dotenv"
    assert config["OPENAI_BASE_URL"] == "https://dotenv.example"
    assert config["OPENAI_MODEL"] == "dotenv-model"


def test_missing_api_key_still_fails_when_env_and_project_dotenv_are_both_absent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing required environment variable: OPENAI_API_KEY"):
        build_provider_client(provider="openai", model=None)


def test_explicit_model_overrides_project_dotenv_model(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL=https://dotenv.example\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url):
            calls["api_key"] = api_key
            calls["base_url"] = base_url

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    client = build_provider_client(provider="openai", model="cli-model")
    assert client.model == "cli-model"
    assert calls["api_key"] == "dotenv-key"
    assert calls["base_url"] == "https://dotenv.example"


def test_model_name_uses_project_dotenv_when_cli_model_missing(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL=https://dotenv.example\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url):
            pass

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    client = build_provider_client(provider="openai", model=None)
    assert client.model == "dotenv-model"


def test_model_name_uses_project_cwd_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url):
            pass

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    client = build_provider_client(provider="openai", model=None)
    assert client.model == "dotenv-model"


def test_environment_overrides_project_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_MODEL=dotenv-model\nOPENAI_BASE_URL=https://dotenv.example\n")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")
    monkeypatch.chdir(tmp_path)

    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key, base_url):
            calls["api_key"] = api_key
            calls["base_url"] = base_url

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    client = build_provider_client(provider="openai", model=None)
    assert client.model == "env-model"
    assert calls["api_key"] == "env-key"
    assert calls["base_url"] == "https://env.example"


def test_missing_model_name_fails_when_cli_env_and_project_dotenv_are_all_absent(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL=https://dotenv.example\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="Missing required model configuration: OPENAI_MODEL"):
        build_provider_client(provider="openai", model=None)


def test_openai_client_does_not_inject_http_client_without_proxy(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL=https://dotenv.example\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    build_provider_client(provider="openai", model=None)
    assert "http_client" not in calls


def test_openai_client_injects_http_client_from_environment_proxy(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_BASE_URL=https://dotenv.example\nOPENAI_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")

    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    build_provider_client(provider="openai", model=None)
    assert isinstance(calls["http_client"], httpx.AsyncClient)


def test_openai_client_injects_http_client_from_project_dotenv_proxy(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=dotenv-key\n"
        "OPENAI_BASE_URL=https://dotenv.example\n"
        "OPENAI_MODEL=dotenv-model\n"
        "HTTP_PROXY=http://127.0.0.1:7890\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    build_provider_client(provider="openai", model=None)
    assert isinstance(calls["http_client"], httpx.AsyncClient)


def test_anthropic_client_injects_http_client_from_environment_proxy(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=dotenv-key\nANTHROPIC_MODEL=dotenv-model\nANTHROPIC_BASE_URL=https://anthropic.example\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")

    calls = {}

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic))

    build_provider_client(provider="anthropic", model=None)
    assert isinstance(calls["http_client"], httpx.AsyncClient)


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
    assert " chars from test]" in prompt
    assert len(prompt) < 100_000


@pytest.mark.asyncio
async def test_attribute_failures_truncates_large_core_fields_and_caps_prompt_size(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    failed_output_path = tmp_path / "failed_attribution_records.jsonl"
    client = FakeProvider(
        outputs=[
            '{"question_id": 10, "is_truly_failed": true, "error_tags": ["tag:10"], "root_cause": "r10", "ability_dimensions": ["a10"], "evidence_snippet": "e10"}',
        ]
    )
    record = EvalRecord.model_validate(
        {
            "question_id": 10,
            "content": "C" * 50_000,
            "canonical_solution": "S" * 50_000,
            "completion": "M" * 50_000,
            "test": "T" * 50_000,
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
        failed_output_path=failed_output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    prompt = client.calls[0]["prompt"]
    assert "[truncated " in prompt
    assert "from content" in prompt
    assert "from canonical_solution" in prompt
    assert "from completion" in prompt
    assert "from test" in prompt
    assert len(prompt) <= 25_000


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


class FailingOnceProvider:
    def __init__(self):
        self.calls = []

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        question_id_line = next(line for line in prompt.splitlines() if line.startswith("Question ID:"))
        question_id = int(question_id_line.split(":", 1)[1].strip())
        self.calls.append(question_id)
        if question_id == 1:
            raise RuntimeError("simulated attribution failure")
        return {
            "question_id": question_id,
            "is_truly_failed": True,
            "error_tags": [f"tag:{question_id}"],
            "root_cause": f"root {question_id}",
            "ability_dimensions": [f"ability {question_id}"],
            "evidence_snippet": f"evidence {question_id}",
        }


@pytest.mark.asyncio
async def test_attribute_failures_skips_failed_record_and_continues(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    failed_output_path = tmp_path / "failed_attribution_records.jsonl"
    client = FailingOnceProvider()

    result = await attribute_failures(
        [make_eval_record(1, "first"), make_eval_record(2, "second"), make_eval_record(3, "third")],
        output_path=output_path,
        failed_output_path=failed_output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=2,
    )

    assert [item.question_id for item in result] == [2, 3]
    assert output_path.read_text().count("\n") == 2
    failed_lines = failed_output_path.read_text().strip().splitlines()
    assert len(failed_lines) == 1
    assert '"question_id": 1' in failed_lines[0]
    assert "simulated attribution failure" in failed_lines[0]


@pytest.mark.asyncio
async def test_attribute_failures_writes_failed_records_metadata(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    failed_output_path = tmp_path / "failed_attribution_records.jsonl"
    client = FailingOnceProvider()

    await attribute_failures(
        [make_eval_record(1, "first")],
        output_path=output_path,
        failed_output_path=failed_output_path,
        provider_client=client,
        provider="openai",
        model="test-model",
        concurrency=1,
    )

    payload = failed_output_path.read_text()
    assert '"question_id": 1' in payload
    assert '"error_type": "RuntimeError"' in payload
    assert '"error_message": "simulated attribution failure"' in payload
    assert '"content_chars":' in payload
    assert '"canonical_solution_chars":' in payload
    assert '"completion_chars":' in payload
    assert '"test_chars":' in payload


@pytest.mark.asyncio
async def test_attribute_failures_all_fail_without_raising(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    failed_output_path = tmp_path / "failed_attribution_records.jsonl"

    class AlwaysFailProvider:
        async def complete_json(self, *, prompt, schema, system, max_tokens, model):
            raise RuntimeError("always fail")

    result = await attribute_failures(
        [make_eval_record(1, "first"), make_eval_record(2, "second")],
        output_path=output_path,
        failed_output_path=failed_output_path,
        provider_client=AlwaysFailProvider(),
        provider="openai",
        model="test-model",
        concurrency=2,
    )

    assert result == []
    assert not output_path.exists() or output_path.read_text() == ""
    assert len(failed_output_path.read_text().strip().splitlines()) == 2
