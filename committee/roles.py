"""Proposer, Critic, Comparator."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

from .clients import LLMClient, Tool, Usage
from .task import Task

S = TypeVar("S")
A = TypeVar("A")


@dataclass
class Candidate(Generic[A]):
    action: A
    raw: str


CRITIC_TOOL = Tool(
    name="submit_critique",
    description="Submit your verdict on the candidate move.",
    parameters={
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["ACCEPT", "REJECT"],
                "description": (
                    "ACCEPT if the move is plausibly correct. REJECT if it is "
                    "clearly wrong, malformed, or violates the task constraints."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One-sentence justification.",
            },
        },
        "required": ["verdict", "reason"],
    },
)


COMPARATOR_TOOL = Tool(
    name="submit_comparison",
    description="Submit which candidate is better.",
    parameters={
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": ["A", "B", "TIE"],
                "description": (
                    "A if Candidate A is better. B if Candidate B is better. "
                    "TIE only if they are genuinely equivalent."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One-sentence justification.",
            },
        },
        "required": ["choice", "reason"],
    },
)


CRITIC_INSTR = (
    "You are a strict reviewer. Decide whether the candidate move is clearly "
    "wrong, malformed, or violates the task constraints. Submit your verdict "
    "via the submit_critique tool."
)


COMPARATOR_INSTR = (
    "Compare two candidate moves and pick the better one. Submit your choice "
    "via the submit_comparison tool. Use TIE only if they are genuinely "
    "equivalent."
)


class Proposer(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def propose(
        self, state: S, k: int, temperature: float = 1.0
    ) -> tuple[list[Candidate[A]], Usage]:
        ctx = self.task.render(state)
        results = await asyncio.gather(
            *[
                self.client.complete(
                    model=self.client.config.proposer_model,
                    system=ctx.system_proposer,
                    user=ctx.describe_state,
                    temperature=temperature,
                    tag="proposer",
                )
                for _ in range(k)
            ]
        )
        usage = Usage()
        candidates: list[Candidate[A]] = []
        for r in results:
            usage.add(r.usage)
            try:
                candidates.append(
                    Candidate(action=self.task.parse_action(r.text), raw=r.text)
                )
            except ValueError:
                continue
        return candidates, usage


class Critic(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def _vote_once(
        self, state: S, cand: Candidate[A]
    ) -> tuple[bool, Usage]:
        ctx = self.task.render(state)
        prompt = (
            f"{ctx.describe_state}\n\n"
            f"Candidate move:\n{cand.raw}"
        )
        result = await self.client.complete_with_tool(
            model=self.client.config.critic_model,
            system=ctx.system_critic + "\n\n" + CRITIC_INSTR,
            user=prompt,
            tool=CRITIC_TOOL,
            temperature=0.7,
            tag="critic",
        )
        verdict = result.input.get("verdict", "REJECT")
        return verdict == "ACCEPT", result.usage

    async def _evaluate(
        self, state: S, cand: Candidate[A], m: int
    ) -> tuple[bool, int, Usage]:
        """Returns (survives, ACCEPT count, usage).

        Calls Task.verify first; if it returns a bool, short-circuit without
        spending any tokens. Otherwise run m LLM votes and require unanimous
        ACCEPT to survive — any REJECT drops the candidate.
        """
        usage = Usage()
        verified = self.task.verify(state, cand.action)
        if verified is False:
            return False, 0, usage
        if verified is True:
            return True, m, usage
        votes = await asyncio.gather(
            *[self._vote_once(state, cand) for _ in range(m)]
        )
        accepts = 0
        for ok, u in votes:
            usage.add(u)
            if ok:
                accepts += 1
        return accepts == m, accepts, usage

    async def filter(
        self, state: S, cands: list[Candidate[A]], m: int
    ) -> tuple[list[Candidate[A]], list[int], Usage]:
        """Returns (survivors, per-survivor ACCEPT counts, aggregate usage)."""
        outcomes = await asyncio.gather(
            *[self._evaluate(state, c, m) for c in cands]
        )
        usage = Usage()
        survivors: list[Candidate[A]] = []
        accepts: list[int] = []
        for cand, (kept, count, u) in zip(cands, outcomes, strict=True):
            usage.add(u)
            if kept:
                survivors.append(cand)
                accepts.append(count)
        return survivors, accepts, usage


class Comparator(Generic[S, A]):
    def __init__(self, client: LLMClient, task: Task[S, A]):
        self.client = client
        self.task = task

    async def _one_vote(
        self, state: S, a: Candidate[A], b: Candidate[A]
    ) -> tuple[str, Usage]:
        ctx = self.task.render(state)
        prompt = (
            f"{ctx.describe_state}\n\n"
            f"Candidate A:\n{a.raw}\n\n"
            f"Candidate B:\n{b.raw}"
        )
        result = await self.client.complete_with_tool(
            model=self.client.config.comparator_model,
            system=ctx.system_comparator + "\n\n" + COMPARATOR_INSTR,
            user=prompt,
            tool=COMPARATOR_TOOL,
            temperature=0.7,
            tag="comparator",
        )
        choice = result.input.get("choice", "TIE")
        if choice not in ("A", "B", "TIE"):
            choice = "TIE"
        return choice, result.usage

    async def pair(
        self, state: S, a: Candidate[A], b: Candidate[A], r: int
    ) -> tuple[int, Usage]:
        """Run r rounds; each round is one AB vote and one BA vote.

        A round counts only when both orderings agree. Returns +1 if A wins
        overall, -1 if B wins, 0 if tied.
        """
        ab_results, ba_results = await asyncio.gather(
            asyncio.gather(*[self._one_vote(state, a, b) for _ in range(r)]),
            asyncio.gather(*[self._one_vote(state, b, a) for _ in range(r)]),
        )
        usage = Usage()
        a_score = 0
        b_score = 0
        for (ab, u_ab), (ba, u_ba) in zip(ab_results, ba_results, strict=True):
            usage.add(u_ab)
            usage.add(u_ba)
            ba_rewritten = "A" if ba == "B" else ("B" if ba == "A" else "TIE")
            if ab == ba_rewritten and ab in ("A", "B"):
                if ab == "A":
                    a_score += 1
                else:
                    b_score += 1
        if a_score > b_score:
            return 1, usage
        if b_score > a_score:
            return -1, usage
        return 0, usage
