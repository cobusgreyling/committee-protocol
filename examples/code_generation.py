"""Verifier-backed code generation — the example the README's verify() pitch needs.

Each task is a Python function spec + hidden unit tests. The proposer emits a
function inside a ```python``` fence. `Task.verify` exec's the code, runs the
hidden tests, and returns True / False — short-circuiting the LLM critic
entirely. This is where the protocol earns its keep over majority vote: when
the verifier is hard ground truth, no amount of LLM consensus on a wrong
answer can override it. Math (the other example) doesn't show this — the LLM
critic catches arithmetic mistakes well enough that verify() is mostly
redundant there.

Run:

    export ANTHROPIC_API_KEY=...
    python examples/code_generation.py

SAFETY: this exec's untrusted LLM output. Fine for a demo against a known
proposer; do not run on adversarial input without sandboxing (subprocess,
container, restricted exec, etc.). A real verifier-backed task should isolate
the execution.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from committee import (
    AnthropicClient,
    Committee,
    CommitteeConfig,
    Task,
    TaskContext,
    Usage,
)
from committee.pricing import HAIKU_4_5

CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


@dataclass
class CodeState:
    signature: str
    spec: str
    tests: list[tuple[tuple[Any, ...], Any]]
    solution: str | None = None


class CodeTask(Task[CodeState, str]):
    def __init__(
        self,
        signature: str,
        spec: str,
        tests: list[tuple[tuple[Any, ...], Any]],
    ):
        self.signature = signature
        self.spec = spec
        self.tests = tests

    def initial_state(self) -> CodeState:
        return CodeState(signature=self.signature, spec=self.spec, tests=self.tests)

    def is_terminal(self, state: CodeState) -> bool:
        return state.solution is not None

    def apply(self, state: CodeState, action: str) -> CodeState:
        return CodeState(
            signature=state.signature,
            spec=state.spec,
            tests=state.tests,
            solution=action,
        )

    def render(self, state: CodeState) -> TaskContext:
        return TaskContext(
            system_proposer=(
                "You write small Python functions. Emit one function inside a "
                "```python ... ``` fenced code block. Match the requested "
                "signature exactly. No imports unless asked; no test code; "
                "no commentary outside the fence."
            ),
            system_critic=(
                "You review a Python function against its spec. REJECT only if "
                "it is syntactically broken, ignores the spec entirely, or is "
                "obviously not a function definition. Otherwise ACCEPT — the "
                "verifier will decide correctness."
            ),
            system_comparator=(
                "You compare two Python functions implementing the same spec. "
                "Prefer the one more likely to be correct on edge cases."
            ),
            describe_state=f"Signature: {state.signature}\nSpec: {state.spec}",
        )

    def parse_action(self, llm_text: str) -> str:
        match = CODE_FENCE.search(llm_text)
        if not match:
            raise ValueError("no python code block")
        return match.group(1).strip()

    def verify(self, state: CodeState, action: str) -> bool | None:
        """Hard ground truth: exec the code, run the hidden tests."""
        name = state.signature.split("(")[0].split()[-1]
        namespace: dict[str, Any] = {}
        try:
            exec(action, namespace)
        except Exception:
            return False
        fn = namespace.get(name)
        if not callable(fn):
            return False
        for args, expected in state.tests:
            try:
                if fn(*args) != expected:
                    return False
            except Exception:
                return False
        return True


PROBLEMS: list[tuple[str, str, list[tuple[tuple[Any, ...], Any]]]] = [
    (
        "def fizzbuzz(n: int) -> str",
        "Return 'Fizz' if n is divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if by both, otherwise str(n). Treat 0 as divisible by both.",
        [((3,), "Fizz"), ((5,), "Buzz"), ((15,), "FizzBuzz"), ((7,), "7"), ((0,), "FizzBuzz")],
    ),
    (
        "def is_palindrome(s: str) -> bool",
        "True if s reads the same forward and backward, case-insensitive, ignoring non-alphanumeric. Empty / single char return True.",
        [
            (("racecar",), True),
            (("A man a plan a canal Panama",), True),
            (("hello",), False),
            (("",), True),
            (("a.",), True),
        ],
    ),
    (
        "def gcd(a: int, b: int) -> int",
        "Greatest common divisor of two non-negative integers. gcd(0, x) == x.",
        [((12, 18), 6), ((100, 75), 25), ((0, 5), 5), ((7, 7), 7), ((1, 1), 1)],
    ),
    (
        "def remove_duplicates(items: list) -> list",
        "Return a list with duplicates removed, preserving order of first occurrence.",
        [
            (([1, 2, 1, 3, 2],), [1, 2, 3]),
            (([],), []),
            (([1, 1, 1],), [1]),
            ((["a", "b", "a"],), ["a", "b"]),
        ],
    ),
    (
        "def is_prime(n: int) -> bool",
        "True iff n is a prime (n >= 2, no divisors other than 1 and n).",
        [
            ((2,), True),
            ((3,), True),
            ((4,), False),
            ((1,), False),
            ((0,), False),
            ((97,), True),
            ((100,), False),
        ],
    ),
    (
        "def reverse_words(s: str) -> str",
        "Reverse the order of whitespace-separated words. Collapse multiple spaces. No leading/trailing whitespace.",
        [
            (("the quick brown fox",), "fox brown quick the"),
            (("hello",), "hello"),
            (("",), ""),
            (("a b c",), "c b a"),
        ],
    ),
]


PERSONAS = [
    "Approach: write the most direct implementation. One expression if possible.",
    "Approach: handle edge cases first (empty input, boundary values), then the general case.",
    "Approach: think about what could go wrong before writing the code. Defensive but minimal.",
    "Approach: write it the way a beginner would — explicit, verbose, clarity over cleverness.",
]


async def main() -> None:
    config = CommitteeConfig(
        k=4, m=3, r=2, max_steps=1,
        proposer_personas=PERSONAS,
    )
    llm = AnthropicClient()
    totals = Usage()
    passes = 0
    for signature, spec, tests in PROBLEMS:
        task = CodeTask(signature, spec, tests)
        committee = Committee(task=task, config=config, llm=llm)
        result = await committee.step(task.initial_state())
        totals.add(result.usage)
        chosen = result.chosen
        ok = (
            chosen is not None
            and task.verify(task.initial_state(), chosen.action) is True
        )
        passes += ok
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{mark}] {signature}  "
            f"proposals={len(result.proposals)} survivors={len(result.survivors)}"
        )
    print()
    print(f"Score: {passes}/{len(PROBLEMS)}")
    print(
        f"Tokens: in={totals.input_tokens} out={totals.output_tokens} "
        f"cache_read={totals.cache_read_input_tokens} calls={totals.calls}"
    )
    print(f"Cost (Haiku 4.5): ${totals.estimated_cost(HAIKU_4_5):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
