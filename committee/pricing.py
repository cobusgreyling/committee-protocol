"""Provider pricing for `Usage.estimated_cost(pricing)`.

The catalog below is convenience. Prices change — always verify against the
provider's current pricing page before relying on these numbers for anything
that matters. Build your own `Pricing(...)` to override.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """USD per **million** tokens for each usage channel."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


# Anthropic. Rates as of 2026-05; check anthropic.com/pricing for the latest.
HAIKU_4_5 = Pricing(
    input_per_mtok=1.00,
    output_per_mtok=5.00,
    cache_read_per_mtok=0.10,
    cache_write_per_mtok=1.25,
)
SONNET_4_5 = Pricing(
    input_per_mtok=3.00,
    output_per_mtok=15.00,
    cache_read_per_mtok=0.30,
    cache_write_per_mtok=3.75,
)
OPUS_4_5 = Pricing(
    input_per_mtok=15.00,
    output_per_mtok=75.00,
    cache_read_per_mtok=1.50,
    cache_write_per_mtok=18.75,
)

# OpenAI. Rates as of 2026-05; check openai.com/pricing for the latest.
GPT_4O_MINI = Pricing(
    input_per_mtok=0.15,
    output_per_mtok=0.60,
    cache_read_per_mtok=0.075,
)
GPT_4O = Pricing(
    input_per_mtok=2.50,
    output_per_mtok=10.00,
    cache_read_per_mtok=1.25,
)
