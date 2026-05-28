"""OpenAI Chat Completions adapter.

OpenAI caches prompt prefixes automatically (no explicit cache_control marker),
so the same per-step system block still gets cached after the first call. The
`cached_tokens` field from the API is surfaced as `Usage.cache_read_input_tokens`.

`openai` is an optional dependency: install with `pip install committee-protocol[openai]`.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..logging import RunLogger
from .base import LLMClient, LLMConfig, TextResult, Tool, ToolResult, Usage


class OpenAIClient(LLMClient):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        config: LLMConfig | None = None,
        logger: RunLogger | None = None,
    ):
        super().__init__(config=config, logger=logger)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIClient requires the openai package. "
                "Install with: pip install 'committee-protocol[openai]'"
            ) from e
        key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(
            api_key=key, max_retries=self.config.max_retries
        )

    @staticmethod
    def _extract_usage(response: Any) -> Usage:
        u = response.usage
        cache_read = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        # OpenAI reports prompt_tokens INCLUSIVE of cached tokens; normalize to
        # the Anthropic shape (input_tokens = uncached input only) so cost
        # math and cross-provider aggregation work uniformly.
        total_prompt = getattr(u, "prompt_tokens", 0) or 0
        return Usage(
            input_tokens=max(0, total_prompt - cache_read),
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read,
            calls=1,
        )

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
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
            response = await self._client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
        text = (response.choices[0].message.content or "").strip()
        usage = self._extract_usage(response)
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
            response = await self._client.chat.completions.create(
                model=model,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                tools=[self._tool_schema(tool)],
                tool_choice={"type": "function", "function": {"name": tool.name}},
            )
        message = response.choices[0].message
        for call in message.tool_calls or []:
            if call.function.name == tool.name:
                inp = json.loads(call.function.arguments)
                usage = self._extract_usage(response)
                self._log(tag, model, "tool", system, user, inp, usage)
                return ToolResult(input=inp, usage=usage)
        raise RuntimeError(
            f"model did not return a tool_call for {tool.name!r}"
        )
