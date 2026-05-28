"""GSM8K benchmark harness.

Compares three strategies on a fixed subset of the GSM8K test set:

  1. single   — one Haiku sample per problem
  2. majority — k Haiku samples, pick the most common answer
  3. committee — full Πk,m,r protocol with personas

The dataset (test.jsonl, ~1319 problems) is downloaded from openai/grade-school-math
on first run and cached locally. Pass --n to restrict to a smaller subset for
faster iteration.

Run:

    export ANTHROPIC_API_KEY=...
    python examples/gsm8k.py --n 200
    python examples/gsm8k.py --n 200 --k 8 --m 5 --r 5   # heavier committee

Output goes to stdout and a JSON file (results/gsm8k_<timestamp>.json) so you
can drop it into RESULTS.md without retyping.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from math_word_problems import MathTask

from committee import AnthropicClient, Committee, CommitteeConfig, Usage
from committee.pricing import HAIKU_4_5

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
CACHE_PATH = Path(__file__).parent.parent / ".cache" / "gsm8k_test.jsonl"
ANSWER_LINE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
ANSWER_FIELD = re.compile(r"ANSWER:\s*(-?\d+(?:\.\d+)?)")

PERSONAS = [
    "Approach: set up the problem as a single arithmetic expression and evaluate it.",
    "Approach: work step by step, naming intermediate quantities. Double-check the final line.",
    "Approach: identify what is being asked first. Re-read the question before computing.",
    "Approach: estimate the answer's order of magnitude first; reject implausible final values.",
]


def _ensure_dataset() -> list[dict]:
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GSM8K test set to {CACHE_PATH} ...")
        urllib.request.urlretrieve(GSM8K_URL, CACHE_PATH)
    rows = []
    with CACHE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _gold_from_answer(answer_field: str) -> float:
    m = ANSWER_LINE.search(answer_field)
    if not m:
        raise ValueError(f"no #### line in: {answer_field[-80:]}")
    return float(m.group(1))


def _parse_answer(text: str) -> float | None:
    m = ANSWER_FIELD.search(text)
    return float(m.group(1)) if m else None


def _matches(got: float | None, gold: float) -> bool:
    return got is not None and abs(got - gold) < 1e-6


@dataclass
class Outcome:
    name: str
    correct: int = 0
    total: int = 0
    usage: Usage | None = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = Usage()


async def _single(llm: AnthropicClient, task: MathTask) -> tuple[float | None, Usage]:
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
    llm: AnthropicClient, task: MathTask, k: int
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
    llm: AnthropicClient, task: MathTask, k: int, m: int, r: int
) -> tuple[float | None, Usage]:
    committee = Committee(
        task=task,
        config=CommitteeConfig(
            k=k, m=m, r=r, max_steps=1,
            proposer_personas=PERSONAS,
        ),
        llm=llm,
    )
    result = await committee.step(task.initial_state())
    chosen = result.chosen.action if result.chosen else None
    return chosen, result.usage


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="problems to evaluate")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument(
        "--strategies",
        default="single,majority,committee",
        help="comma-separated subset of {single,majority,committee}",
    )
    ap.add_argument("--seed", type=int, default=0, help="subset selection seed (0 = first n)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    rows = _ensure_dataset()
    if args.seed:
        import random as _r
        _r.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    print(f"Loaded {len(rows)} GSM8K problems (of {args.n} requested).")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    llm = AnthropicClient()
    outcomes = {name: Outcome(name=name, total=len(rows)) for name in strategies}
    if "majority" in outcomes:
        outcomes["majority"].name = f"majority{args.k}"

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        question = row["question"]
        try:
            gold = _gold_from_answer(row["answer"])
        except ValueError:
            continue
        task = MathTask(question)

        if "single" in strategies:
            ans, u = await _single(llm, task)
            outcomes["single"].usage.add(u)
            if _matches(ans, gold):
                outcomes["single"].correct += 1
        if "majority" in strategies:
            ans, u = await _majority(llm, task, args.k)
            outcomes["majority"].usage.add(u)
            if _matches(ans, gold):
                outcomes["majority"].correct += 1
        if "committee" in strategies:
            ans, u = await _committee(llm, task, args.k, args.m, args.r)
            outcomes["committee"].usage.add(u)
            if _matches(ans, gold):
                outcomes["committee"].correct += 1

        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(f"  {i}/{len(rows)} done  ({elapsed:.0f}s)")

    print()
    print(f"Πk,m,r = ({args.k},{args.m},{args.r})  problems={len(rows)}\n")
    header = f"{'strategy':<12} {'acc':<10} {'calls':<8} {'in':<10} {'out':<8} {'cache_r':<10} {'cost (Haiku 4.5)':<10}"
    print(header)
    print("-" * len(header))
    payload = {
        "k": args.k, "m": args.m, "r": args.r,
        "n": len(rows), "model": "claude-haiku-4-5-20251001",
        "strategies": {},
    }
    for o in outcomes.values():
        acc = o.correct / o.total if o.total else 0.0
        u = o.usage
        cost = u.estimated_cost(HAIKU_4_5)
        print(
            f"{o.name:<12} {o.correct}/{o.total} ({acc:.1%})  "
            f"{u.calls:<8} {u.input_tokens:<10} {u.output_tokens:<8} "
            f"{u.cache_read_input_tokens:<10} ${cost:.4f}"
        )
        payload["strategies"][o.name] = {
            "correct": o.correct,
            "total": o.total,
            "accuracy": acc,
            "usage": asdict(u),
            "estimated_cost_usd": cost,
        }

    out_path = args.out or (
        Path(__file__).parent.parent / "results" / f"gsm8k_{int(time.time())}.json"
    )
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
