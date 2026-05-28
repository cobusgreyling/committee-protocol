# Theory

## What Πk,m,r means

Πk,m,r is a family of orchestration policies for sampling-based reasoning, parameterised by three integers:

- **k** — how many proposals are drawn per step
- **m** — how many critic votes each proposal receives
- **r** — how many comparator rounds each pair of survivors receives

A fourth quantity, **diversity**, is set by prompt design — by how unlike each
other the k proposers are. There is no integer for it because no integer makes
proposers disagree if they share blind spots.

Π is a shorthand because the policy decomposes as a product of three independent
roles, each doing one thing.

## Four quantities

A reasoning system that combines weak samples gets traction from four levers.
Read them as the things you can be short on:

**Coverage** — *did the right answer get proposed at least once?*
Set by k and by sampling temperature. If k=1 you have no coverage; if k=∞ you
have it for free. Coverage failures look like all k proposals being wrong in
the same way.

**Identifiability** — *given that the right answer was proposed, can the
system tell?*
Set by m (critic) and r (comparator). The critic identifies wrong answers
against an absolute standard ("does this typecheck?"); the comparator
identifies a relative ranking ("is A better than B?"). Identifiability
failures look like the right answer surviving the critic but losing the
pairwise vote.

**Progress** — *does each step actually advance the state?*
Set by `max_steps` and the task's `apply`. Multi-step tasks have progress
failures (loops, no-ops) that single-step tasks cannot.

**Diversity** — *do the k proposals span the answer space, not just the most
likely region of it?*
Set by prompt design, not by k. The `proposer_personas` knob is the explicit
hook: round-robin distinct personas across the k calls so they explore
different angles.

The trap is treating coverage as the only quantity that matters and scaling k.
That works on tasks where the right answer is highly likely to appear in a
random sample (math word problems often). On harder tasks the proposers
share priors and converge on a plausible-but-wrong answer; k=64 gets you 64
votes for the same wrong thing.

## Why the critic is unanimous-ACCEPT

The critic runs m votes per candidate. Any single REJECT drops the candidate.

This is asymmetric on purpose. The LLM critic has weak evidence: a single ACCEPT
is uninformative (LLMs love saying yes), but a single REJECT for a clear
reason is a strong signal. Requiring unanimity ACCEPTs is the cheapest way to
convert weak per-vote evidence into a useful filter.

When `Task.verify` returns True or False, the LLM critic is skipped entirely.
That is the hook where the protocol becomes meaningfully different from
majority vote. The LLM critic is a fallback; the verifier is ground truth.
Tasks with real verifiers (run the tests, typecheck, execute the code) are
where Πk,m,r earns its keep.

## Why both-orders-or-tie for the comparator

The comparator gets r rounds per pair. Each round is two votes — A-versus-B
*and* B-versus-A. If the two orderings disagree, the round is a tie. Only
agreed rounds count.

The reason is order bias. LLMs systematically prefer the first option they see
in a comparison, the second option they see, or whichever option has more
tokens — whichever bias the model happens to have. The both-orders-or-tie
rule cancels the bias: if the comparator is responding to *content* rather
than *position*, the two orderings agree. If it is responding to position,
they disagree and the round contributes nothing.

This is what turns the LLM judge into usable weak evidence rather than
treating it as ground truth. Agreed rounds are signal; disagreed rounds are
noise that the protocol declines to act on.

## Why Copeland for pair aggregation

Once each pair has produced a +1 / -1 / 0 outcome, the protocol uses Copeland:
each survivor scores +1 per pair won. The highest score wins. Ties are
broken by critic-acceptance margin (more ACCEPTs wins). Unbroken ties return
`chosen=None` rather than picking by list index.

Copeland is cheap, monotone, and Condorcet-consistent: if some candidate beats
every other candidate pairwise, it wins. The alternative — running a global
ranking model — would cost O(survivors!) prompts to be principled, and
would re-introduce the order bias the pairwise vote was set up to cancel.

The unbroken-tie case matters more than it looks. Silently picking by list
index would let the proposer order leak into the final answer; logging the tie
and returning None forces the caller to decide (rerun, escalate, abstain).
The protocol is honest about not being able to identify a winner.

## Where this comes from

Πk,m,r is a reframing of test-time compute decompositions that show up across
several recent lines of work:

- **Best-of-n with a verifier** is the k=n, m=1 (deterministic), r=0 corner.
  Coverage with no identifiability.
- **Majority vote** is k=n, m=0, r=0. Coverage with no identifiability at all.
- **Self-consistency** is majority vote with a sampling-temperature lever.
- **Tree-of-Thoughts** introduces multi-step planning (max_steps > 1) but
  collapses critic and comparator into a single scoring head.
- **Debate / multi-agent critique** adds the comparator phase but typically
  uses one ordering — keeping the order bias.

The Πk,m,r notation makes the corners explicit, which makes the cost / quality
tradeoff legible: doubling k buys coverage at linear cost; doubling m or r
buys identifiability at linear cost per pair (so r is quadratic in survivor
count); diversity costs nothing per token but requires designing personas
that are actually different.

## What to tune first

A practical sequencing:

1. **Get a verifier.** Without `Task.verify`, the LLM critic is the only thing
   distinguishing the protocol from sophisticated majority vote. The gap
   between "verifier-backed" and "LLM-judged-only" is bigger than the gap
   between any two knob settings.
2. **Set k = 4 to 8 with personas.** Diversity first; coverage second. k=16
   with one persona is worse than k=4 with four good personas.
3. **Set m = 3 to 5.** Unanimous-ACCEPT means each additional m drops the
   false-accept rate exponentially. More than 5 is rarely worth it.
4. **Set r = 2 to 5.** Both-orders-per-round means each r costs two calls per
   pair. Stop adding r when most rounds are agreeing (look at the rejected
   tie rate).
5. **Only then scale up.** If accuracy is still short, the proposer's prior
   has a blind spot the critic can't see around. Adding more proposers won't
   fix it; adding a different proposer (different model, different personas)
   might.
