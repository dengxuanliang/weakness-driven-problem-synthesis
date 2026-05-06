"""Async provider abstraction for JSON LLM completions."""

from __future__ import annotations

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


def build_provider_client(provider: str, model: str | None) -> ProviderClient:
    if provider not in DEFAULT_MODELS:
        raise RuntimeError(f"Unsupported provider: {provider}")

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(env_var):
        raise RuntimeError(f"Missing required environment variable: {env_var}")

    return ProviderClient(provider=provider, model=model or DEFAULT_MODELS[provider])


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

    for _ in range(3):
        raw_output = await client.complete_json(
            prompt=repair_prompt,
            schema=schema,
            system=system,
            max_tokens=max_tokens,
            model=resolved_model,
        )
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            repair_prompt = (
                f"{prompt}\n\nPrevious output was not valid JSON. "
                f"Repair it to valid JSON only:\n{raw_output}"
            )

    raise ValueError("Model did not return valid JSON after 3 attempts")
