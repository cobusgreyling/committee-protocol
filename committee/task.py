"""Task abstraction. Subclass `Task` to plug a new domain into the committee."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class TaskContext:
    """Prompts the three roles use to reason about the current state."""

    system_proposer: str
    system_critic: str
    system_comparator: str
    describe_state: str


class Task(ABC, Generic[S, A]):
    @abstractmethod
    def initial_state(self) -> S: ...

    @abstractmethod
    def is_terminal(self, state: S) -> bool: ...

    @abstractmethod
    def apply(self, state: S, action: A) -> S: ...

    @abstractmethod
    def render(self, state: S) -> TaskContext: ...

    @abstractmethod
    def parse_action(self, llm_text: str) -> A:
        """Extract a structured action from raw LLM output. Raise ValueError if unparseable."""

    def verify(self, state: S, action: A) -> bool | None:
        """Optional programmatic ground-truth verifier.

        Return True to short-circuit the critic with ACCEPT, False to short-circuit
        with REJECT, None to fall back to LLM-judged critique. This is the hook
        the README's "verifier-backed tasks" claim depends on — running tests,
        type-checking, executing code, checking arithmetic. The LLM critic alone
        is weak evidence; the verifier is what makes propose/critique/compare
        meaningfully better than majority vote.
        """
        return None
