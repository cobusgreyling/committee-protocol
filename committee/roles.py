"""Proposer, Critic, Comparator."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Generic, TypeVar

from .client import LLMClient
from .task import Task

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Candidate(Generic[A]):
    action: A
    raw: str


def _last_token(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    last = re.sub(r"[^A-Za-z]+$", "", lines[-1])
    last = re.sub(r"^[^A-Za-z]+", "", last)
    tokens = last.split()
    return tokens[-1].upper() if tokens else ""


class Proposer(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def propose(
        self, state: S, k: int, temperature: float = 1.0
    ) -> list[Candidate[A]]:
        ctx = self.task.render(state)
        raws = await asyncio.gather(
            *[
                self.client.complete(
                    model=self.client.config.proposer_model,
                    system=ctx.system_proposer,
                    user=ctx.describe_state,
                    temperature=temperature,
                )
                for _ in range(k)
            ]
        )
        out: list[Candidate[A]] = []
        for raw in raws:
            try:
                out.append(Candidate(action=self.task.parse_action(raw), raw=raw))
            except ValueError:
                continue
        return out


CRITIC_INSTR = (
    "You are a strict reviewer. Decide if the candidate move is clearly wrong, "
    "malformed, or violates the task constraints. "
    "On the LAST line, output exactly one token: ACCEPT or REJECT."
)


class Critic(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def _vote_once(self, state: S, cand: Candidate[A]) -> bool:
        ctx = self.task.render(state)
        prompt = (
            f"{ctx.describe_state}\n\n"
            f"Candidate move:\n{cand.raw}\n\n"
            f"Is this move clearly wrong, malformed, or constraint-violating?"
        )
        out = await self.client.complete(
            model=self.client.config.critic_model,
            system=ctx.system_critic + "\n\n" + CRITIC_INSTR,
            user=prompt,
            temperature=0.7,
        )
        token = _last_token(out)
        return token != "REJECT"

    async def survives(self, state: S, cand: Candidate[A], m: int) -> bool:
        """True only if all m critic votes ACCEPT. Any REJECT drops it."""
        verdicts = await asyncio.gather(
            *[self._vote_once(state, cand) for _ in range(m)]
        )
        return all(verdicts)

    async def filter(
        self, state: S, cands: list[Candidate[A]], m: int
    ) -> list[Candidate[A]]:
        keeps = await asyncio.gather(*[self.survives(state, c, m) for c in cands])
        return [c for c, keep in zip(cands, keeps) if keep]


COMPARATOR_INSTR = (
    "Compare two candidate moves and pick the better one for the task. "
    "On the LAST line, output exactly one token: A or B."
)


class Comparator(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def _one_vote(self, state: S, a: Candidate[A], b: Candidate[A]) -> str:
        ctx = self.task.render(state)
        prompt = (
            f"{ctx.describe_state}\n\n"
            f"Candidate A:\n{a.raw}\n\n"
            f"Candidate B:\n{b.raw}\n\n"
            f"Which is better?"
        )
        out = await self.client.complete(
            model=self.client.config.comparator_model,
            system=ctx.system_comparator + "\n\n" + COMPARATOR_INSTR,
            user=prompt,
            temperature=0.7,
        )
        token = _last_token(out)
        return token if token in ("A", "B") else "TIE"

    async def pair(
        self, state: S, a: Candidate[A], b: Candidate[A], r: int
    ) -> int:
        """Run r rounds; each round is one AB vote and one BA vote.

        A round counts only when both orderings agree. Returns +1 if A wins overall,
        -1 if B wins, 0 if tied.
        """
        ab_results, ba_results = await asyncio.gather(
            asyncio.gather(*[self._one_vote(state, a, b) for _ in range(r)]),
            asyncio.gather(*[self._one_vote(state, b, a) for _ in range(r)]),
        )
        a_score = 0
        b_score = 0
        for ab, ba in zip(ab_results, ba_results):
            ba_rewritten = (
                "A" if ba == "B" else ("B" if ba == "A" else "TIE")
            )
            if ab == ba_rewritten and ab in ("A", "B"):
                if ab == "A":
                    a_score += 1
                else:
                    b_score += 1
        if a_score > b_score:
            return 1
        if b_score > a_score:
            return -1
        return 0
