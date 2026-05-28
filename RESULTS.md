# Results

The README's central claim — *a weak reasoning model can match a frontier
model on verifier-backed tasks* — is only useful if it's falsifiable. This
file is where the falsification lives. The benchmark harness is in
`examples/gsm8k.py`; the table below is the place to drop the numbers it
prints.

## Methodology

**Dataset.** GSM8K test split (1319 grade-school math word problems). The
harness defaults to the first 200 problems; pass `--n 1319` for the full set
or `--seed 1` to shuffle.

**Strategies.**

- `single` — one Haiku sample, temperature 1.0.
- `majority` — *k* Haiku samples at temperature 1.0, pick the most common
  parsed answer (Counter.most_common(1)).
- `committee` — full Πk,m,r with four diverse personas
  (set up the problem / step-by-step / restate the question /
  estimate first), critic at temperature 0.7, comparator at temperature 0.7,
  both-orders-or-tie comparator aggregation, Copeland with critic-margin
  tiebreak.

**Model.** `claude-haiku-4-5-20251001` for all three roles. The
provider-neutral `LLMClient` means you can rerun the same harness against
OpenAI (or any future adapter) and get a comparable column.

**Scoring.** Exact match on the parsed `ANSWER: <number>` line against the
GSM8K `#### <number>` gold value. `abs(got - gold) < 1e-6`.

**Cost.** Token usage is recorded per strategy and converted to USD using
`committee.pricing.HAIKU_4_5`. Verify the current Anthropic prices before
quoting any dollar figure as authoritative — pricing moves.

## How to reproduce

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
python examples/gsm8k.py --n 200                  # default Πk,m,r = (4, 3, 2)
python examples/gsm8k.py --n 200 --k 8 --m 5 --r 5  # heavier committee
```

The first run downloads `test.jsonl` to `.cache/gsm8k_test.jsonl`. Each run
writes a machine-readable summary to `results/gsm8k_<timestamp>.json` so the
table below can be regenerated from raw data.

## Headline numbers

Numbers below are **not pre-filled**. Run the harness, paste the output. PRs
adding rows with reproducible JSON payloads in `results/` are the point of
this file existing.

### Πk,m,r = (4, 3, 2), N = 200

| strategy   | accuracy | calls | input tok | output tok | cache reads | cost (Haiku 4.5) |
| ---------- | -------- | ----- | --------- | ---------- | ----------- | ---------------- |
| single     |          |       |           |            |             |                  |
| majority4  |          |       |           |            |             |                  |
| committee  |          |       |           |            |             |                  |

### Πk,m,r = (8, 5, 5), N = 200

| strategy   | accuracy | calls | input tok | output tok | cache reads | cost (Haiku 4.5) |
| ---------- | -------- | ----- | --------- | ---------- | ----------- | ---------------- |
| single     |          |       |           |            |             |                  |
| majority8  |          |       |           |            |             |                  |
| committee  |          |       |           |            |             |                  |

## Reading the table

The protocol is only interesting if the **cost-adjusted accuracy delta**
beats both baselines. Three patterns to watch for:

1. `committee` accuracy > `majority`*k* accuracy at the same *k*. If they
   match, the critic/comparator phases bought nothing on this task and
   majority vote is sufficient.
2. `committee` cost per correct answer ≤ a Sonnet-single baseline. The
   pitch is "spend more inference on Haiku to skip Sonnet"; if Sonnet wins
   on $/correct, the pitch fails for this task.
3. Cache reads should dominate input tokens at *k* > 4. If they don't, the
   per-step system block isn't shaped right — the cache is doing nothing.

GSM8K is a particularly hard test for the protocol because the LLM critic
is decent at catching arithmetic errors *without* a verifier — so the
ceiling on improvement is lower than on code tasks (see
`examples/code_generation.py`). Don't read a flat GSM8K result as the
protocol failing; read it as GSM8K being the wrong benchmark to showcase
it on. Verifier-backed tasks are where the asymmetry shows up.
