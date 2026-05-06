"""Async provider abstraction for JSON LLM completions."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any


DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-4o",
}


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
        response = await self.client.responses.create(
            model=model,
            input=_build_openai_input(prompt=prompt, system=system),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=max_tokens,
        )
        return response.output_text


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


def _build_openai_input(*, prompt: str, system: str | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if system:
        items.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system}],
            }
        )
    items.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }
    )
    return items


def build_provider_client(provider: str, model: str | None) -> ProviderClient:
    if provider not in DEFAULT_MODELS:
        raise RuntimeError(f"Unsupported provider: {provider}")

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(env_var):
        raise RuntimeError(f"Missing required environment variable: {env_var}")

    resolved_model = model or DEFAULT_MODELS[provider]
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for provider=openai") from exc

        return OpenAIProviderClient(
            provider=provider,
            model=resolved_model,
            client=AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
        )

    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for provider=anthropic") from exc

    return AnthropicProviderClient(
        provider=provider,
        model=resolved_model,
        client=AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
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
    resolved_model = model or getattr(client, "model", DEFAULT_MODELS[provider])
    repair_prompt = prompt

    for attempt in range(3):
        raw_output = await _complete_with_backoff(
            client=client,
            prompt=repair_prompt,
            schema=schema,
            system=system,
            max_tokens=max_tokens,
            model=resolved_model,
        )
        if isinstance(raw_output, (dict, list)):
            return raw_output
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            if attempt == 2:
                break
            repair_prompt = (
                f"{prompt}\n\nPrevious output was not valid JSON. "
                f"Repair it to valid JSON only:\n{raw_output}"
            )

    raise ValueError("Model did not return valid JSON after 3 attempts")


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
