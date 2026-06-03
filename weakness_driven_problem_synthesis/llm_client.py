"""Async provider abstraction for JSON LLM completions."""

from __future__ import annotations

import asyncio
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_PROVIDERS = {"anthropic", "openai"}

OPENAI_JSON_OBJECT_MODE = "json_object"
OPENAI_PLAIN_TEXT_ARRAY_MODE = "plain_text_array"
_REQUEST_THROTTLER: "_RequestThrottler | None" = None


@dataclass
class ProviderClient:
    provider: str
    model: str

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: dict,
        system: str | None,
        max_tokens: int,
        model: str,
    ) -> str:
        raise NotImplementedError("Network provider calls are not implemented yet")


@dataclass
class OpenAIProviderClient(ProviderClient):
    client: Any

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: dict,
        system: str | None,
        max_tokens: int,
        model: str,
    ) -> str:
        completion_mode = _openai_completion_mode(schema)
        effective_system = system
        request_kwargs: dict[str, Any] = {}

        # Some OpenAI-compatible gateways reliably support structured JSON mode only
        # for object-root outputs. Array-root outputs stay in plain-text JSON mode and
        # are parsed locally after completion.
        if completion_mode == OPENAI_JSON_OBJECT_MODE:
            request_kwargs["response_format"] = {"type": "json_object"}
        else:
            array_instruction = _openai_array_instruction()
            effective_system = array_instruction if system is None else f"{system}\n\n{array_instruction}"

        response = await self.client.chat.completions.create(
            model=model,
            messages=_build_openai_messages(prompt=prompt, system=effective_system),
            max_tokens=max_tokens,
            **request_kwargs,
        )
        return response.choices[0].message.content


@dataclass
class AnthropicProviderClient(ProviderClient):
    client: Any

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: dict,
        system: str | None,
        max_tokens: int,
        model: str,
    ) -> str:
        format_instruction = (
            "Return valid JSON only. Match this JSON schema exactly:\n"
            f"{json.dumps(schema, ensure_ascii=True)}"
        )
        system_prompt = format_instruction if system is None else f"{system}\n\n{format_instruction}"
        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise ValueError("Anthropic response did not contain any text blocks")
        return "".join(text_blocks)


@dataclass
class _RequestThrottler:
    max_in_flight: int
    min_interval_seconds: float
    burst_limit: int
    burst_cooldown_seconds: float

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_in_flight)
        self._lock = asyncio.Lock()
        self._last_started_at = 0.0
        self._recent_starts: list[float] = []
        self._cooldown_until = 0.0

    async def __aenter__(self) -> "_RequestThrottler":
        await self._semaphore.acquire()
        await self._acquire_start_slot()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._semaphore.release()

    async def _acquire_start_slot(self) -> None:
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                self._recent_starts = [ts for ts in self._recent_starts if now - ts < 60.0]
                cooldown_wait = max(0.0, self._cooldown_until - now)
                interval_wait = max(0.0, self.min_interval_seconds - (now - self._last_started_at))
                if self.burst_limit > 0 and len(self._recent_starts) >= self.burst_limit and cooldown_wait <= 0:
                    self._cooldown_until = max(self._cooldown_until, now + self.burst_cooldown_seconds)
                    self._recent_starts = []
                    cooldown_wait = max(cooldown_wait, self.burst_cooldown_seconds)
                wait_for = max(interval_wait, cooldown_wait)
                if wait_for <= 0:
                    now = asyncio.get_running_loop().time()
                    self._last_started_at = now
                    self._recent_starts.append(now)
                    return
            await asyncio.sleep(wait_for)


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_request_throttler() -> _RequestThrottler:
    global _REQUEST_THROTTLER
    if _REQUEST_THROTTLER is None:
        _REQUEST_THROTTLER = _RequestThrottler(
            max_in_flight=max(1, _read_int_env("WEAKNESS_SYNTH_MAX_IN_FLIGHT", 8)),
            min_interval_seconds=max(0.0, _read_float_env("WEAKNESS_SYNTH_MIN_INTERVAL_MS", 150.0) / 1000.0),
            burst_limit=max(1, _read_int_env("WEAKNESS_SYNTH_BURST_LIMIT", 12)),
            burst_cooldown_seconds=max(0.0, _read_float_env("WEAKNESS_SYNTH_BURST_COOLDOWN_MS", 1200.0) / 1000.0),
        )
    return _REQUEST_THROTTLER


def _openai_completion_mode(schema: dict[str, Any]) -> str:
    return OPENAI_PLAIN_TEXT_ARRAY_MODE if schema.get("type") == "array" else OPENAI_JSON_OBJECT_MODE


def _openai_array_instruction() -> str:
    return (
        "Return valid JSON only. "
        "The top-level value must be a JSON array. "
        "The first character must be '[' and the last character must be ']'."
    )


def _build_openai_messages(*, prompt: str, system: str | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if system:
        items.append(
            {
                "role": "system",
                "content": system,
            }
        )
    items.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    return items


def _extract_fenced_code_block(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None

    first_newline = stripped.find("\n")
    if first_newline == -1:
        return None

    closing_index = stripped.rfind("\n```")
    if closing_index <= first_newline:
        return None

    return stripped[first_newline + 1 : closing_index].strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_env_path(*, cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()) / ".env"


def _load_env_file_if_needed(*, env_path: Path | None = None) -> dict[str, str]:
    path = env_path or _project_env_path()
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        loaded[key] = value
    return loaded


def _resolve_config_value(*, key: str, config: dict[str, str]) -> str | None:
    env_value = os.getenv(key)
    if env_value:
        return env_value
    return config.get(key)


def _resolve_proxy_url(config: dict[str, str]) -> str | None:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = os.getenv(key)
        if value:
            return value
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = config.get(key)
        if value:
            return value
    return None


def _build_optional_httpx_client(config: dict[str, str]) -> Any | None:
    proxy_url = _resolve_proxy_url(config)
    if not proxy_url:
        return None

    import httpx

    return httpx.AsyncClient(
        proxy=proxy_url,
        timeout=httpx.Timeout(120.0),
        trust_env=False,
    )


def _model_env_var_for_provider(provider: str) -> str:
    return "ANTHROPIC_MODEL" if provider == "anthropic" else "OPENAI_MODEL"


def _resolve_model_name(provider: str, model: str | None, *, config: dict[str, str]) -> str:
    if model:
        return model
    env_var = _model_env_var_for_provider(provider)
    env_model = _resolve_config_value(key=env_var, config=config)
    if env_model:
        return env_model
    raise RuntimeError(f"Missing required model configuration: {env_var}")


def _parse_json_with_fence_fallback(raw_output: Any) -> dict | list[dict]:
    if isinstance(raw_output, (dict, list)):
        return raw_output

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        fenced = _extract_fenced_code_block(raw_output) if isinstance(raw_output, str) else None
        if fenced is None:
            raise
        return json.loads(fenced)


def build_provider_client(provider: str, model: str | None) -> ProviderClient:
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(f"Unsupported provider: {provider}")

    config = _load_env_file_if_needed()

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = _resolve_config_value(key=env_var, config=config)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {env_var}")

    resolved_model = _resolve_model_name(provider, model, config=config)
    http_client = _build_optional_httpx_client(config)
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for provider=openai") from exc

        openai_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": _resolve_config_value(key="OPENAI_BASE_URL", config=config),
        }
        if http_client is not None:
            openai_kwargs["http_client"] = http_client

        return OpenAIProviderClient(
            provider=provider,
            model=resolved_model,
            client=AsyncOpenAI(**openai_kwargs),
        )

    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for provider=anthropic") from exc

    anthropic_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": _resolve_config_value(key="ANTHROPIC_BASE_URL", config=config),
    }
    if http_client is not None:
        anthropic_kwargs["http_client"] = http_client

    return AnthropicProviderClient(
        provider=provider,
        model=resolved_model,
        client=AsyncAnthropic(**anthropic_kwargs),
    )


async def complete_json(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    provider: str = "anthropic",
    model: str | None = None,
    provider_client: Any | None = None,
) -> dict | list[dict]:
    client = provider_client or build_provider_client(provider=provider, model=model)
    resolved_model = model or getattr(client, "model", None)
    if resolved_model is None:
        raise RuntimeError("Provider client did not expose a resolved model")
    repair_prompt = prompt
    last_raw_output: Any | None = None

    for attempt in range(3):
        raw_output = await _complete_with_backoff(
            client=client,
            prompt=repair_prompt,
            schema=schema,
            system=system,
            max_tokens=max_tokens,
            model=resolved_model,
        )
        last_raw_output = raw_output
        try:
            return _parse_json_with_fence_fallback(raw_output)
        except json.JSONDecodeError:
            if attempt == 2:
                break
            repair_prompt = (
                f"{prompt}\n\nPrevious output was not valid JSON. "
                f"Repair it to valid JSON only:\n{raw_output}"
            )

    _maybe_write_invalid_json_debug_dump(last_raw_output)
    raise ValueError(
        "Model did not return valid JSON after 3 attempts. "
        f"Last output preview: {_preview_text(last_raw_output)}"
    )


async def _complete_with_backoff(
    *,
    client: Any,
    prompt: str,
    schema: dict,
    system: str | None,
    max_tokens: int,
    model: str,
) -> Any:
    delay = 1
    last_error: Exception | None = None
    for _ in range(5):
        try:
            async with _get_request_throttler():
                return await client.complete_json(
                    prompt=prompt,
                    schema=schema,
                    system=system,
                    max_tokens=max_tokens,
                    model=model,
                )
        except Exception as exc:
            last_error = exc
            if not _looks_retryable(exc):
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
    if last_error is None:
        raise RuntimeError("LLM completion failed without an error")
    raise last_error


def _looks_retryable(error: Exception) -> bool:
    response_text = _extract_error_response_text(error)
    message_text = str(error).lower()
    response_message = response_text.lower() if response_text else ""
    combined_message = f"{message_text}\n{response_message}".strip()
    retry_markers = ("rate limit", "429", "500", "502", "503", "504", "timeout", "temporarily unavailable")
    if any(marker in combined_message for marker in retry_markers):
        return True
    status_code = _extract_error_status_code(error)
    if status_code == 403 or "403" in message_text:
        if _looks_like_gateway_block_page(response_message or message_text):
            return True
    return False


def _extract_error_status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    if response is not None:
        response_status_code = getattr(response, "status_code", None)
        if isinstance(response_status_code, int):
            return response_status_code

    resp = getattr(error, "resp", None)
    if resp is not None:
        resp_status_code = getattr(resp, "status_code", None)
        if isinstance(resp_status_code, int):
            return resp_status_code

    return None


def _extract_error_response_text(error: Exception) -> str | None:
    for attr_name in ("response", "resp", "body", "content", "text"):
        value = getattr(error, attr_name, None)
        if value is None:
            continue
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                continue
        if isinstance(value, str):
            return value
        nested_text = getattr(value, "text", None)
        if isinstance(nested_text, str):
            return nested_text
        nested_content = getattr(value, "content", None)
        if isinstance(nested_content, bytes):
            return nested_content.decode("utf-8", errors="ignore")
        if isinstance(nested_content, str):
            return nested_content
    return None


def _looks_like_gateway_block_page(message: str) -> bool:
    lowered = html.unescape(message.lower())
    block_markers = (
        "<html",
        "<!doctype html",
        "permission denied",
        "access denied",
        "forbidden",
        "firewall",
        "blocked",
        "captcha",
    )
    return any(marker in lowered for marker in block_markers)


def _preview_text(value: Any, *, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _maybe_write_invalid_json_debug_dump(value: Any) -> None:
    debug_path = os.getenv("WEAKNESS_SYNTH_DEBUG_PATH")
    if not debug_path:
        return
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    with open(debug_path, "w") as handle:
        handle.write(text)
