# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- `OpenAIClient` adapter alongside `AnthropicClient`. The protocol itself was
  always provider-agnostic; the import wasn't. Install with
  `pip install committee-protocol[openai]`.
- `LLMClient` is now an abstract base class. Subclass it to plug in any
  async chat-completion provider with one system block + one user message +
  optional forced tool call (i.e. every modern chat API).
- `Tool` dataclass — provider-neutral tool definition; adapters translate to
  the Anthropic `input_schema` / OpenAI `function` wire format.
- `Task.verify(state, action) -> bool | None` — optional programmatic verifier.
  Returning True or False short-circuits the LLM critic; returning None falls
  back to LLM-judged critique. This is the hook real verifier-backed tasks
  plug into (run tests, type-check, check arithmetic).
- `Usage` dataclass returned from every `StepResult` and from each role.
  Tracks `input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
  `cache_read_input_tokens`, `calls`.
- Prompt caching on the system block (`cache_control: ephemeral`). With the
  default knobs that's hundreds of cache reads per step.
- Bounded concurrency in `LLMClient` via `asyncio.Semaphore`
  (`LLMConfig.max_concurrency`, default 8).
- `LLMConfig.max_retries` (default 4) wired through to the SDK's built-in
  exponential backoff.
- Optional JSONL run logging via `RunLogger`. Off by default; pass
  `AnthropicClient(logger=RunLogger("run.jsonl"))` to capture one line per call.
- `examples/eval.py` — single-shot vs majority-of-k vs full committee on the
  demo problems, with per-strategy accuracy and token usage.
- Test suite (`tests/`) and CI (`.github/workflows/ci.yml`) on Python 3.10,
  3.11, 3.12 with ruff + pytest.

### Changed
- `committee/client.py` → `committee/clients/{base,anthropic,openai}.py`. The
  default client (`Committee(...)` with no `llm=` argument) is still Anthropic.
  Examples that built one explicitly now use `AnthropicClient()`.
- Critic and Comparator now use tool-use with a JSON schema instead of
  last-line text parsing. Eliminates the silent ACCEPT-on-malformed-output
  failure mode in the previous `_last_token` heuristic.
- Copeland aggregation uses critic-acceptance margin as an explicit
  tiebreaker. Unbroken ties return `chosen=None` and log a warning rather
  than picking by list index.

### Removed
- The `_last_token` text parser is gone, replaced by tool-use.
