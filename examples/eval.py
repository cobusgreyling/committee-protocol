"""Compare three strategies on the demo math problems:

  1. single   — one Haiku sample per problem
  2. majority — k Haiku samples, pick the most common answer
  3. committee — full Πk,m,r protocol (this repo)

For each strategy, print accuracy and total token usage. This is the
falsifiable form of the README's claim — without baselines, you can't tell
whether the protocol is doing work or whether one Haiku sample would have
solved it anyway.

Run:

    export ANTHROPIC_API_KEY=...
    python examples/eval.py
    python examples/eval.py --k 8 --m 5 --r 5   # heavier committee
"""
from __future__ import annotations

import argparse
import asyncio
import re
from collections import Counter
from dataclasses import dataclass

# Reuse the demo problems so this script is self-contained.
from math_word_problems import PROBLEMS, MathTask

from committee import Committee, CommitteeConfig, LLMClient, Usage


@dataclass
class Outcome:
    name: str
    correct: int
    total: int
    usage: Usage

    def line(self) -> str:
        acc = self.correct / self.total if self.total else 0.0
        u = self.usage
        return (
            f"{self.name:<10}  acc={self.correct}/{self.total} ({acc:.0%})  "
            f"calls={u.calls:<4} in={u.input_tokens:<6} out={u.output_tokens:<5} "
            f"cache_r={u.cache_read_input_tokens}"
        )


def _parse_answer(text: str) -> float | None:
    m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


async def _single(
    llm: LLMClient, task: MathTask
) -> tuple[float | None, Usage]:
    ctx = task.render(task.initial_state())
    r = await llm.complete(
        model=llm.config.proposer_model,
        system=ctx.system_proposer,
        user=ctx.describe_state,
        temperature=1.0,
        tag="single",
    )
    return _parse_answer(r.text), r.usage


async def _majority(
    llm: LLMClient, task: MathTask, k: int
) -> tuple[float | None, Usage]:
    ctx = task.render(task.initial_state())
    results = await asyncio.gather(
        *[
            llm.complete(
                model=llm.config.proposer_model,
                system=ctx.system_proposer,
                user=ctx.describe_state,
                temperature=1.0,
                tag="majority",
            )
            for _ in range(k)
        ]
    )
    usage = Usage()
    answers: list[float] = []
    for r in results:
        usage.add(r.usage)
        a = _parse_answer(r.text)
        if a is not None:
            answers.append(a)
    if not answers:
        return None, usage
    most_common, _ = Counter(answers).most_common(1)[0]
    return most_common, usage


async def _committee(
    llm: LLMClient, task: MathTask, k: int, m: int, r: int
) -> tuple[float | None, Usage]:
    committee = Committee(
        task=task,
        config=CommitteeConfig(k=k, m=m, r=r, max_steps=1),
        llm=llm,
    )
    result = await committee.step(task.initial_state())
    chosen = result.chosen.action if result.chosen else None
    return chosen, result.usage


def _matches(got: float | None, gold: float) -> bool:
    return got is not None and abs(got - gold) < 1e-6


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--r", type=int, default=2)
    args = ap.parse_args()

    llm = LLMClient()
    single = Outcome("single", 0, len(PROBLEMS), Usage())
    majority = Outcome(f"majority{args.k}", 0, len(PROBLEMS), Usage())
    committee = Outcome("committee", 0, len(PROBLEMS), Usage())

    for question, gold in PROBLEMS:
        task = MathTask(question)

        s_ans, s_u = await _single(llm, task)
        single.usage.add(s_u)
        if _matches(s_ans, gold):
            single.correct += 1

        m_ans, m_u = await _majority(llm, task, args.k)
        majority.usage.add(m_u)
        if _matches(m_ans, gold):
            majority.correct += 1

        c_ans, c_u = await _committee(llm, task, args.k, args.m, args.r)
        committee.usage.add(c_u)
        if _matches(c_ans, gold):
            committee.correct += 1

    print(f"Πk,m,r = ({args.k},{args.m},{args.r})  problems={len(PROBLEMS)}\n")
    print(single.line())
    print(majority.line())
    print(committee.line())


if __name__ == "__main__":
    asyncio.run(main())
