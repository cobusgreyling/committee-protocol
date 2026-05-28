"""Provider-neutral async client used by Proposer / Critic / Comparator.

The Πk,m,r protocol does not depend on Anthropic. Every modern chat API can
serve a single system block + a single user message + an optional forced tool
call, which is everything the three roles need. Concrete adapters live in
sibling modules (`anthropic.py`, `openai.py`).
"""
from __future__ import annotations

import asyncio
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..logging import RunLogger


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
class TextResult:
    text: str
    usage: Usage


@dataclass
class ToolResult:
    input: dict[str, Any]
    usage: Usage


@dataclass
class Tool:
    """Provider-neutral tool definition. Adapters translate to their wire format."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMConfig:
    """Runtime config shared across adapters.

    `*_model` fields may be left as None — each adapter fills in its own default
    (e.g. Haiku for Anthropic, gpt-4o-mini for OpenAI).
    """

    proposer_model: str | None = None
    critic_model: str | None = None
    comparator_model: str | None = None
    max_tokens: int = 1024
    max_concurrency: int = 8
    max_retries: int = 4
    api_key: str | None = None


class LLMClient(ABC):
    """Abstract base for any async chat-completion provider.

    Subclass and implement `complete` (free-form text) and `complete_with_tool`
    (forced structured tool call). The shared `__init__` handles config copy,
    default-model fill-in, concurrency limiting, and optional JSONL logging.
    """

    DEFAULT_MODEL: str = ""

    def __init__(
        self,
        config: LLMConfig | None = None,
        logger: RunLogger | None = None,
    ):
        self.config = dataclasses.replace(config) if config else LLMConfig()
        for field in ("proposer_model", "critic_model", "comparator_model"):
            if getattr(self.config, field) is None:
                setattr(self.config, field, self.DEFAULT_MODEL)
        self.logger = logger
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

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

    @abstractmethod
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 1.0,
        tag: str | None = None,
    ) -> TextResult: ...

    @abstractmethod
    async def complete_with_tool(
        self,
        model: str,
        system: str,
        user: str,
        tool: Tool,
        temperature: float = 0.7,
        tag: str | None = None,
    ) -> ToolResult: ...
