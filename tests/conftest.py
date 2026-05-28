"""Shared test fixtures: a mock LLMClient that returns scripted responses."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from committee import LLMClient, LLMConfig, TextResult, Tool, ToolResult, Usage


@dataclass
class FakeUsage:
    @staticmethod
    def one() -> Usage:
        return Usage(input_tokens=1, output_tokens=1, calls=1)


class MockClient(LLMClient):
    """LLMClient that bypasses the network and returns scripted responses.

    `text_responder(tag, system, user) -> str` for complete()
    `tool_responder(tag, tool_name, system, user) -> dict` for complete_with_tool()
    """

    def __init__(
        self,
        text_responder: Callable[[str | None, str, str], str] | None = None,
        tool_responder: Callable[[str | None, str, str, str], dict[str, Any]] | None = None,
    ):
        super().__init__(config=LLMConfig())
        self.text_responder = text_responder
        self.tool_responder = tool_responder
        self.text_calls: list[tuple[str | None, str, str, float]] = []
        self.tool_calls: list[tuple[str | None, str, str, str]] = []

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 1.0,
        tag: str | None = None,
    ) -> TextResult:
        self.text_calls.append((tag, system, user, temperature))
        if self.text_responder is None:
            raise RuntimeError("MockClient.text_responder not set")
        return TextResult(text=self.text_responder(tag, system, user), usage=FakeUsage.one())

    async def complete_with_tool(
        self,
        model: str,
        system: str,
        user: str,
        tool: Tool,
        temperature: float = 0.7,
        tag: str | None = None,
    ) -> ToolResult:
        self.tool_calls.append((tag, tool.name, system, user))
        if self.tool_responder is None:
            raise RuntimeError("MockClient.tool_responder not set")
        return ToolResult(
            input=self.tool_responder(tag, tool.name, system, user),
            usage=FakeUsage.one(),
        )
