# committee-protocol

A reference implementation of **Πk,m,r** — the propose / critique / compare committee protocol that lets a weak reasoning model match frontier-class performance on verifier-backed tasks.

Three roles. Each does one thing.

- **Proposer** — generates `k` candidate moves per step.
- **Critic** — votes `m` times per candidate. Any REJECT drops it.
- **Comparator** — pairwise votes between survivors, in **both orderings**. Disagreements count as ties. Copeland aggregation picks the winner.

The both-orders-or-tie rule is what makes the LLM judge usable as weak evidence rather than ground truth.

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

Subclass `Task` with five methods:

```python
from committee import Task, TaskContext

class MyTask(Task):
    def initial_state(self): ...
    def is_terminal(self, state): ...
    def apply(self, state, action): ...
    def render(self, state) -> TaskContext: ...   # prompts for the three roles
    def parse_action(self, llm_text): ...         # extract structured action
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
