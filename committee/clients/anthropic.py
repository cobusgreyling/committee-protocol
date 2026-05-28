"""Anthropic Messages API adapter.

Marks the system block as `cache_control: ephemeral` so the API serves it from
cache across the k*m + k*r*2 calls per step — most of the cost savings come
from this.
"""
from __future__ import annotations

import os
from typing import Any

from anthropic import AsyncAnthropic

from ..logging import RunLogger
from .base import LLMClient, LLMConfig, TextResult, Tool, ToolResult, Usage


class AnthropicClient(LLMClient):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        config: LLMConfig | None = None,
        logger: RunLogger | None = None,
    ):
        super().__init__(config=config, logger=logger)
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(
            api_key=key, max_retries=self.config.max_retries
        )

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
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

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
        tool: Tool,
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
                tools=[self._tool_schema(tool)],
                tool_choice={"type": "tool", "name": tool.name},
            )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool.name:
                inp = dict(block.input)
                usage = self._extract_usage(msg)
                self._log(tag, model, "tool", system, user, inp, usage)
                return ToolResult(input=inp, usage=usage)
        raise RuntimeError(
            f"model did not return a tool_use block for {tool.name!r}"
        )
