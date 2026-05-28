"""Toy demo: grade-school math word problems.

Each task is single-step. The committee proposes k candidate solutions, the
critic drops malformed or arithmetically wrong ones, and the comparator ranks
survivors in both orderings (Copeland aggregation, critic-margin tiebreak) to
pick the winner.

Run:

    export ANTHROPIC_API_KEY=...
    python examples/math_word_problems.py
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from committee import AnthropicClient, Committee, CommitteeConfig, Task, TaskContext, Usage


@dataclass
class MathState:
    question: str
    answer: float | None = None


class MathTask(Task[MathState, float]):
    def __init__(self, question: str):
        self.question = question

    def initial_state(self) -> MathState:
        return MathState(question=self.question)

    def is_terminal(self, state: MathState) -> bool:
        return state.answer is not None

    def apply(self, state: MathState, action: float) -> MathState:
        return MathState(question=state.question, answer=action)

    def render(self, state: MathState) -> TaskContext:
        return TaskContext(
            system_proposer=(
                "You solve grade-school math word problems. Reason step by step. "
                "On the LAST line, output exactly: ANSWER: <number>"
            ),
            system_critic=(
                "You are a math reviewer. REJECT a solution if its arithmetic is "
                "wrong, its final ANSWER line is missing or malformed, or it "
                "misreads the question. Otherwise ACCEPT."
            ),
            system_comparator=(
                "You compare two math solutions. The better one is correct, "
                "well-reasoned, and uses the right interpretation of the question."
            ),
            describe_state=f"Problem: {state.question}",
        )

    def parse_action(self, llm_text: str) -> float:
        match = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", llm_text)
        if not match:
            raise ValueError("no ANSWER line")
        return float(match.group(1))


PROBLEMS = [
    ("Janet sells 16 eggs at $2 each, but she eats 3 herself. How much does she make?", 26.0),
    ("A shirt costs $40 with 25% off. What is the final price?", 30.0),
    ("A library has 240 books and 1/4 are nonfiction. How many nonfiction books are there?", 60.0),
    ("If 5 friends split a $85 bill equally, how much does each pay?", 17.0),
]


async def main():
    config = CommitteeConfig(k=4, m=3, r=2, max_steps=1)
    llm = AnthropicClient()
    totals = Usage()
    passes = 0
    for question, gold in PROBLEMS:
        task = MathTask(question)
        committee = Committee(task=task, config=config, llm=llm)
        # use step() so we can print usage; solve() throws the StepResult away
        state = task.initial_state()
        result = await committee.step(state)
        totals.add(result.usage)
        got = result.chosen.action if result.chosen else None
        ok = got is not None and abs(got - gold) < 1e-6
        passes += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] gold={gold}  got={got}  calls={result.usage.calls}")
        print(f"       Q: {question}")
    print()
    print(f"Score: {passes}/{len(PROBLEMS)}")
    print(
        f"Tokens: in={totals.input_tokens} out={totals.output_tokens} "
        f"cache_read={totals.cache_read_input_tokens} "
        f"cache_write={totals.cache_creation_input_tokens} "
        f"calls={totals.calls}"
    )


if __name__ == "__main__":
    asyncio.run(main())
