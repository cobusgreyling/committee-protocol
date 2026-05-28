"""The Πk,m,r committee orchestrator."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from .client import LLMClient, LLMConfig
from .roles import Candidate, Comparator, Critic, Proposer
from .task import Task

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class CommitteeConfig:
    k: int = 8
    m: int = 5
    r: int = 5
    max_steps: int = 1
    proposer_temperature: float = 1.0


@dataclass
class StepResult(Generic[A]):
    chosen: Candidate[A] | None
    survivors: list[Candidate[A]]
    proposals: list[Candidate[A]]


class Committee(Generic[S, A]):
    def __init__(
        self,
        task: Task[S, A],
        config: CommitteeConfig | None = None,
        llm: LLMClient | None = None,
    ):
        self.task = task
        self.config = config or CommitteeConfig()
        self.llm = llm or LLMClient(LLMConfig())
        self.proposer = Proposer(self.llm, task)
        self.critic = Critic(self.llm, task)
        self.comparator = Comparator(self.llm, task)

    async def step(self, state: S) -> StepResult[A]:
        proposals = await self.proposer.propose(
            state, self.config.k, self.config.proposer_temperature
        )
        if not proposals:
            return StepResult(chosen=None, survivors=[], proposals=[])

        survivors = await self.critic.filter(state, proposals, self.config.m)
        if not survivors:
            return StepResult(chosen=None, survivors=[], proposals=proposals)
        if len(survivors) == 1:
            return StepResult(
                chosen=survivors[0], survivors=survivors, proposals=proposals
            )

        pairs = [
            (i, j)
            for i in range(len(survivors))
            for j in range(i + 1, len(survivors))
        ]
        outcomes = await asyncio.gather(
            *[
                self.comparator.pair(state, survivors[i], survivors[j], self.config.r)
                for i, j in pairs
            ]
        )
        scores = [0] * len(survivors)
        for (i, j), outcome in zip(pairs, outcomes):
            if outcome > 0:
                scores[i] += 1
            elif outcome < 0:
                scores[j] += 1
        winner_idx = max(range(len(survivors)), key=lambda x: scores[x])
        return StepResult(
            chosen=survivors[winner_idx], survivors=survivors, proposals=proposals
        )

    async def solve(self, initial: S | None = None) -> S:
        state = initial if initial is not None else self.task.initial_state()
        for _ in range(self.config.max_steps):
            if self.task.is_terminal(state):
                break
            result = await self.step(state)
            if result.chosen is None:
                break
            state = self.task.apply(state, result.chosen.action)
        return state
