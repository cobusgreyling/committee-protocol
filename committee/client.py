"""Thin async wrapper around the Anthropic SDK."""
from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import AsyncAnthropic

DEFAULT_PROPOSER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CRITIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPARATOR_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LLMConfig:
    proposer_model: str = DEFAULT_PROPOSER_MODEL
    critic_model: str = DEFAULT_CRITIC_MODEL
    comparator_model: str = DEFAULT_COMPARATOR_MODEL
    max_tokens: int = 1024
    api_key: str | None = None


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=key)

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 1.0,
    ) -> str:
        msg = await self._client.messages.create(
            model=model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()
