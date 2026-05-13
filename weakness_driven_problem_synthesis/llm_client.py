"""Async provider abstraction for JSON LLM completions."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_PROVIDERS = {"anthropic", "openai"}

OPENAI_JSON_OBJECT_MODE = "json_object"
OPENAI_PLAIN_TEXT_ARRAY_MODE = "plain_text_array"


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
    message = str(error).lower()
    retry_markers = ("rate limit", "429", "500", "502", "503", "504", "timeout", "temporarily unavailable")
    return any(marker in message for marker in retry_markers)


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
