"""Unit tests for the Πk,m,r protocol pieces.

No network. Uses a MockClient that returns scripted responses.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from committee import (
    Candidate,
    Committee,
    CommitteeConfig,
    Comparator,
    Critic,
    Proposer,
    Task,
    TaskContext,
)

from .conftest import MockClient

# ---- minimal task used across tests --------------------------------------


@dataclass
class IntState:
    seed: int = 0
    answer: int | None = None


class IntTask(Task[IntState, int]):
    """Picks an integer. parse_action expects 'ANSWER: <int>'."""

    def __init__(self, verifier: callable | None = None):
        self._verifier = verifier

    def initial_state(self) -> IntState:
        return IntState()

    def is_terminal(self, state: IntState) -> bool:
        return state.answer is not None

    def apply(self, state: IntState, action: int) -> IntState:
        return IntState(seed=state.seed, answer=action)

    def render(self, state: IntState) -> TaskContext:
        return TaskContext(
            system_proposer="proposer", system_critic="critic",
            system_comparator="comparator", describe_state=f"state={state.seed}",
        )

    def parse_action(self, llm_text: str) -> int:
        import re
        m = re.search(r"ANSWER:\s*(-?\d+)", llm_text)
        if not m:
            raise ValueError("no answer")
        return int(m.group(1))

    def verify(self, state: IntState, action: int) -> bool | None:
        if self._verifier is None:
            return None
        return self._verifier(action)


# ---- Copeland tiebreak ----------------------------------------------------


def test_copeland_clear_winner():
    survivors = [Candidate(1, "raw"), Candidate(2, "raw"), Candidate(3, "raw")]
    chosen = Committee._pick_winner(survivors, scores=[2, 1, 0], accept_counts=[3, 3, 3])
    assert chosen is survivors[0]


def test_copeland_tie_broken_by_critic_margin():
    survivors = [Candidate(1, "raw"), Candidate(2, "raw"), Candidate(3, "raw")]
    chosen = Committee._pick_winner(survivors, scores=[1, 1, 0], accept_counts=[5, 3, 4])
    assert chosen is survivors[0]


def test_copeland_unbroken_tie_returns_none():
    survivors = [Candidate(1, "raw"), Candidate(2, "raw")]
    chosen = Committee._pick_winner(survivors, scores=[1, 1], accept_counts=[3, 3])
    assert chosen is None


# ---- Critic short-circuits on Task.verify --------------------------------


@pytest.mark.asyncio
async def test_verify_true_skips_llm():
    # If verify returns True, we must not call the LLM critic at all.
    client = MockClient(tool_responder=lambda *a: pytest.fail("LLM should not be called"))
    task = IntTask(verifier=lambda a: True)
    critic = Critic(client, task)
    kept, accepts, usage = await critic._evaluate(task.initial_state(), Candidate(1, "raw"), m=3)
    assert kept is True
    assert accepts == 3
    assert usage.calls == 0
    assert client.tool_calls == []


@pytest.mark.asyncio
async def test_verify_false_skips_llm():
    client = MockClient(tool_responder=lambda *a: pytest.fail("LLM should not be called"))
    task = IntTask(verifier=lambda a: False)
    critic = Critic(client, task)
    kept, accepts, usage = await critic._evaluate(task.initial_state(), Candidate(1, "raw"), m=3)
    assert kept is False
    assert accepts == 0
    assert usage.calls == 0


@pytest.mark.asyncio
async def test_verify_none_falls_back_to_llm():
    client = MockClient(tool_responder=lambda *a: {"verdict": "ACCEPT", "reason": "ok"})
    task = IntTask(verifier=None)  # no verifier
    critic = Critic(client, task)
    kept, accepts, usage = await critic._evaluate(task.initial_state(), Candidate(1, "raw"), m=3)
    assert kept is True
    assert accepts == 3
    assert usage.calls == 3


# ---- Critic: any REJECT drops the candidate ------------------------------


@pytest.mark.asyncio
async def test_any_reject_drops_candidate():
    call_count = {"n": 0}

    def responder(tag, tool_name, system, user):
        call_count["n"] += 1
        # accept, accept, reject -> total reject (any REJECT drops)
        return {"verdict": "REJECT" if call_count["n"] == 3 else "ACCEPT", "reason": "x"}

    client = MockClient(tool_responder=responder)
    task = IntTask()
    critic = Critic(client, task)
    kept, accepts, _ = await critic._evaluate(task.initial_state(), Candidate(1, "raw"), m=3)
    assert kept is False
    assert accepts == 2


# ---- Comparator: both-orders-disagree counts as tie ----------------------


@pytest.mark.asyncio
async def test_disagreement_counts_as_tie():
    # AB always picks A; BA also picks A (i.e. picks B in the A-frame).
    # So in the A-frame: AB says A, BA-rewritten says B -> disagreement -> tie.
    def responder(tag, tool_name, system, user):
        return {"choice": "A", "reason": "x"}

    client = MockClient(tool_responder=responder)
    task = IntTask()
    comp = Comparator(client, task)
    outcome, _ = await comp.pair(task.initial_state(), Candidate(1, "ra"), Candidate(2, "rb"), r=3)
    assert outcome == 0


@pytest.mark.asyncio
async def test_consistent_preference_wins():
    # AB picks A, BA picks B (in BA-frame B is the "first" candidate, so picking
    # B-in-BA means picking the original A). Both agree A wins -> a_score=3.
    def responder(tag, tool_name, system, user):
        # detect order by which raw shows up first; this is brittle, but works
        # because Candidate A's raw is "ra" and we put it first in AB.
        if "Candidate A:\nra" in user:
            return {"choice": "A", "reason": "x"}
        return {"choice": "B", "reason": "x"}

    client = MockClient(tool_responder=responder)
    task = IntTask()
    comp = Comparator(client, task)
    outcome, _ = await comp.pair(task.initial_state(), Candidate(1, "ra"), Candidate(2, "rb"), r=3)
    assert outcome == 1


# ---- Proposer: personas + temperature jitter -----------------------------


@pytest.mark.asyncio
async def test_personas_round_robin_into_system_prompt():
    client = MockClient(text_responder=lambda *a: "ANSWER: 1")
    task = IntTask()
    proposer = Proposer(client, task)
    personas = ["PERSONA-A", "PERSONA-B"]
    await proposer.propose(
        task.initial_state(), k=4, personas=personas
    )
    systems = [s for _tag, s, _u, _t in client.text_calls]
    assert "PERSONA-A" in systems[0]
    assert "PERSONA-B" in systems[1]
    assert "PERSONA-A" in systems[2]
    assert "PERSONA-B" in systems[3]


@pytest.mark.asyncio
async def test_temperature_jitter_within_bounds():
    import random as _random
    _random.seed(42)
    client = MockClient(text_responder=lambda *a: "ANSWER: 1")
    task = IntTask()
    proposer = Proposer(client, task)
    await proposer.propose(
        task.initial_state(), k=8, temperature=0.7, temperature_jitter=0.3
    )
    temps = [t for _tag, _s, _u, t in client.text_calls]
    assert all(0.4 <= t <= 1.0 for t in temps)
    assert len(set(temps)) > 1  # actually different


@pytest.mark.asyncio
async def test_no_jitter_keeps_temperature_constant():
    client = MockClient(text_responder=lambda *a: "ANSWER: 1")
    task = IntTask()
    proposer = Proposer(client, task)
    await proposer.propose(task.initial_state(), k=4, temperature=0.5)
    temps = [t for _tag, _s, _u, t in client.text_calls]
    assert temps == [0.5, 0.5, 0.5, 0.5]


# ---- Proposer: parse failures are silently dropped -----------------------


@pytest.mark.asyncio
async def test_parse_failures_dropped():
    outputs = iter([
        "Reasoning... ANSWER: 42",
        "this output has no answer line",
        "ANSWER: 7",
        "nope",
    ])

    def responder(tag, system, user):
        return next(outputs)

    client = MockClient(text_responder=responder)
    task = IntTask()
    proposer = Proposer(client, task)
    candidates, usage = await proposer.propose(task.initial_state(), k=4, temperature=1.0)
    assert [c.action for c in candidates] == [42, 7]
    assert usage.calls == 4  # all 4 were called even though 2 were dropped


# ---- Committee step: empty proposals returns chosen=None -----------------


@pytest.mark.asyncio
async def test_all_proposals_unparseable_returns_none():
    client = MockClient(text_responder=lambda *a: "no answer here")
    task = IntTask()
    committee = Committee(
        task=task, config=CommitteeConfig(k=3, m=1, r=1, max_steps=1), llm=client
    )
    result = await committee.step(task.initial_state())
    assert result.chosen is None
    assert result.proposals == []


@pytest.mark.asyncio
async def test_all_rejected_returns_none():
    client = MockClient(
        text_responder=lambda *a: "ANSWER: 1",
        tool_responder=lambda *a: {"verdict": "REJECT", "reason": "x"},
    )
    task = IntTask()
    committee = Committee(
        task=task, config=CommitteeConfig(k=3, m=2, r=1, max_steps=1), llm=client
    )
    result = await committee.step(task.initial_state())
    assert result.chosen is None
    assert len(result.proposals) == 3
    assert result.survivors == []
