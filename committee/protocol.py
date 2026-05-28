"""The Πk,m,r committee orchestrator."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .client import LLMClient, LLMConfig, Usage
from .roles import Candidate, Comparator, Critic, Proposer
from .task import Task

S = TypeVar("S")
A = TypeVar("A")

logger = logging.getLogger(__name__)


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
    usage: Usage = field(default_factory=Usage)


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
        usage = Usage()
        proposals, prop_usage = await self.proposer.propose(
            state, self.config.k, self.config.proposer_temperature
        )
        usage.add(prop_usage)
        if not proposals:
            return StepResult(chosen=None, survivors=[], proposals=[], usage=usage)

        survivors, accept_counts, critic_usage = await self.critic.filter(
            state, proposals, self.config.m
        )
        usage.add(critic_usage)
        if not survivors:
            return StepResult(
                chosen=None, survivors=[], proposals=proposals, usage=usage
            )
        if len(survivors) == 1:
            return StepResult(
                chosen=survivors[0],
                survivors=survivors,
                proposals=proposals,
                usage=usage,
            )

        pairs = [
            (i, j)
            for i in range(len(survivors))
            for j in range(i + 1, len(survivors))
        ]
        outcomes = await asyncio.gather(
            *[
                self.comparator.pair(
                    state, survivors[i], survivors[j], self.config.r
                )
                for i, j in pairs
            ]
        )
        scores = [0] * len(survivors)
        for (i, j), (outcome, comp_usage) in zip(pairs, outcomes, strict=True):
            usage.add(comp_usage)
            if outcome > 0:
                scores[i] += 1
            elif outcome < 0:
                scores[j] += 1

        chosen = self._pick_winner(survivors, scores, accept_counts)
        return StepResult(
            chosen=chosen, survivors=survivors, proposals=proposals, usage=usage
        )

    @staticmethod
    def _pick_winner(
        survivors: list[Candidate[A]],
        scores: list[int],
        accept_counts: list[int],
    ) -> Candidate[A] | None:
        """Copeland with critic-margin tiebreak.

        If multiple survivors top the pairwise score, prefer the one with the
        most critic ACCEPTs. If still tied, return None — callers decide what
        to do with an unbroken tie rather than silently picking by list index.
        """
        max_score = max(scores)
        top = [i for i, s in enumerate(scores) if s == max_score]
        if len(top) == 1:
            return survivors[top[0]]
        max_accepts = max(accept_counts[i] for i in top)
        finalists = [i for i in top if accept_counts[i] == max_accepts]
        if len(finalists) == 1:
            return survivors[finalists[0]]
        logger.warning(
            "Copeland tie unbroken: %d survivors tied on score=%d and accepts=%d",
            len(finalists),
            max_score,
            max_accepts,
        )
        return None

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
