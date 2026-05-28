"""Thin async wrapper around the Anthropic SDK.

Adds: bounded concurrency, prompt caching on the system block, structured
tool-use for critic and comparator, configurable retries, per-call usage
accounting, optional JSONL run logging.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from .logging import RunLogger

DEFAULT_PROPOSER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_CRITIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPARATOR_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class Usage:
    """Token usage aggregated across one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.calls += other.calls


@dataclass
class LLMConfig:
    proposer_model: str = DEFAULT_PROPOSER_MODEL
    critic_model: str = DEFAULT_CRITIC_MODEL
    comparator_model: str = DEFAULT_COMPARATOR_MODEL
    max_tokens: int = 1024
    max_concurrency: int = 8
    max_retries: int = 4
    api_key: str | None = None


@dataclass
class TextResult:
    text: str
    usage: Usage


@dataclass
class ToolResult:
    input: dict[str, Any]
    usage: Usage


class LLMClient:
    def __init__(
        self,
        config: LLMConfig | None = None,
        logger: RunLogger | None = None,
    ):
        self.config = config or LLMConfig()
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(
            api_key=key, max_retries=self.config.max_retries
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self.logger = logger

    @staticmethod
    def _extract_usage(msg: Any) -> Usage:
        u = msg.usage
        return Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            calls=1,
        )

    @staticmethod
    def _system_blocks(system: str) -> list[dict[str, Any]]:
        # The system prompt is identical across the k*m + k*r*2 calls per step.
        # Marking it ephemeral lets the API serve it from cache on every call
        # after the first, which is where most of the cost savings come from.
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _log(
        self,
        tag: str | None,
        model: str,
        kind: str,
        system: str,
        user: str,
        output: Any,
        usage: Usage,
    ) -> None:
        if not self.logger:
            return
        self.logger.write(
            {
                "tag": tag,
                "model": model,
                "kind": kind,
                "system_prefix": system[:200],
                "user": user,
                "output": output,
                "usage": dataclasses.asdict(usage),
            }
        )

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 1.0,
        tag: str | None = None,
    ) -> TextResult:
        async with self._semaphore:
            msg = await self._client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                system=self._system_blocks(system),
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
            )
        text = "".join(
            block.text
            for block in msg.content
            if getattr(block, "type", None) == "text"
        ).strip()
        usage = self._extract_usage(msg)
        self._log(tag, model, "text", system, user, text, usage)
        return TextResult(text=text, usage=usage)

    async def complete_with_tool(
        self,
        model: str,
        system: str,
        user: str,
        tool: dict[str, Any],
        temperature: float = 0.7,
        tag: str | None = None,
    ) -> ToolResult:
        async with self._semaphore:
            msg = await self._client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                system=self._system_blocks(system),
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                inp = dict(block.input)
                usage = self._extract_usage(msg)
                self._log(tag, model, "tool", system, user, inp, usage)
                return ToolResult(input=inp, usage=usage)
        raise RuntimeError(
            f"model did not return a tool_use block for {tool['name']!r}"
        )
