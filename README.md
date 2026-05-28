# committee-protocol

A reference implementation of **Πk,m,r** — the propose / critique / compare committee protocol that lets a weak reasoning model match frontier-class performance on verifier-backed tasks.

Three roles. Each does one thing.

- **Proposer** — generates `k` candidate moves per step.
- **Critic** — votes `m` times per candidate. Any REJECT drops it.
- **Comparator** — pairwise votes between survivors, in **both orderings**. Disagreements count as ties. Copeland aggregation picks the winner.

The both-orders-or-tie rule is what makes the LLM judge usable as weak evidence rather than ground truth.

```
            state s_t
                │
                ▼
   ┌──────────────────────────────┐
   │  PROPOSER × k                │   ─→ k candidates       coverage
   │  personas, jittered temp     │
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │  Task.verify(candidate)      │
   │   True  → ACCEPT, skip LLM   │
   │   False → REJECT, skip LLM   │
   │   None  → ↓                  │
   │  CRITIC × m                  │   ─→ survivors          identifiability
   │  any REJECT drops            │
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │  COMPARATOR × r per pair     │
   │  both orders; agree → vote;  │
   │  disagree → tie              │   ─→ Copeland scores    identifiability
   │  critic-margin tiebreak      │
   └──────────────────────────────┘
                │
                ▼
            winner s_{t+1}
```

For the longer-form version — why unanimous ACCEPT, why both-orders-or-tie, why Copeland, and where Πk,m,r sits relative to best-of-n / self-consistency / ToT / debate — see [THEORY.md](THEORY.md).

## Install

```bash
git clone https://github.com/cobusgreyling/committee-protocol
cd committee-protocol
pip install -e .
export ANTHROPIC_API_KEY=...
```

## Run the demo

```bash
python examples/math_word_problems.py
```

Four grade-school math problems, solved by a Haiku-class committee. Each problem runs `k=4` proposers, `m=3` critic votes per candidate, and `r=2` comparator rounds per pair (each round = AB vote + BA vote).

## Use it on your own task

Subclass `Task`. Five methods are required; a sixth — `verify` — is optional but is what makes the protocol meaningfully better than majority vote:

```python
from committee import Task, TaskContext

class MyTask(Task):
    def initial_state(self): ...
    def is_terminal(self, state): ...
    def apply(self, state, action): ...
    def render(self, state) -> TaskContext: ...   # prompts for the three roles
    def parse_action(self, llm_text): ...         # extract structured action

    # Optional. Return True / False to short-circuit the LLM critic with hard
    # ground truth (run tests, type-check, check arithmetic, execute the code).
    # Return None to fall back to LLM-judged critique. This is the hook that
    # makes propose/critique/compare actually verifier-backed rather than
    # LLM-judged-only.
    def verify(self, state, action) -> bool | None: ...
```

Then run a committee over it:

```python
import asyncio
from committee import Committee, CommitteeConfig

async def main():
    task = MyTask(...)
    committee = Committee(task=task, config=CommitteeConfig(k=8, m=5, r=5))
    result = await committee.solve()
    print(result)

asyncio.run(main())
```

Token usage is surfaced on `StepResult.usage` (input, output, cache reads, cache writes, call count) so you can put numbers on what the protocol costs versus what you get for it. Prompt caching is on by default — the system block is cached across the k·m + k·r·2 calls per step.

## Compare against baselines

The protocol is only interesting if it beats simpler strategies. `examples/eval.py` runs three on the demo problems and prints a table:

```bash
python examples/eval.py
```

- `single` — one Haiku sample per problem
- `majority` — k Haiku samples, pick the most common answer
- `committee` — full Πk,m,r protocol

For a real benchmark, `examples/gsm8k.py` runs the same three strategies over a 200-problem subset of GSM8K (or the full 1319), with personas wired up and per-strategy cost tracking. See [RESULTS.md](RESULTS.md) for methodology and the table to fill in.

```bash
python examples/gsm8k.py --n 200
```

## Knobs

`CommitteeConfig`:

| Field | Meaning | Scales |
|-------|---------|--------|
| `k` | proposals per step | **coverage** |
| `m` | critic votes per candidate | **identifiability** (rejection) |
| `r` | comparator rounds per pair | **identifiability** (ranking) |
| `max_steps` | outer loop length | **progress** (multi-step tasks) |

A fourth quantity — **diversity** — is set by your prompt design and decomposition, not by these knobs. If the proposer has shared blind spots, no value of `k`, `m`, or `r` will close the gap.

## What this is not

A SWE-bench harness. The protocol is small; the verifier and the task decomposition are the hard parts. Bring your own.

## License

MIT
