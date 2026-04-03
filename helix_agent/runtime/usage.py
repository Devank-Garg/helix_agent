"""helix_agent.runtime.usage — Token accounting and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """Per-million-token pricing for a Claude model tier."""

    input_cost_per_million: float
    output_cost_per_million: float
    cache_creation_cost_per_million: float
    cache_read_cost_per_million: float

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def haiku(cls) -> "ModelPricing":
        """Pricing for Claude Haiku."""
        return cls(
            input_cost_per_million=1.0,
            output_cost_per_million=5.0,
            cache_creation_cost_per_million=1.25,
            cache_read_cost_per_million=0.1,
        )

    @classmethod
    def sonnet(cls) -> "ModelPricing":
        """Pricing for Claude Sonnet."""
        return cls(
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
            cache_creation_cost_per_million=18.75,
            cache_read_cost_per_million=1.5,
        )

    @classmethod
    def opus(cls) -> "ModelPricing":
        """Pricing for Claude Opus."""
        return cls(
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
            cache_creation_cost_per_million=18.75,
            cache_read_cost_per_million=1.5,
        )


def pricing_for_model(model: str) -> Optional[ModelPricing]:
    """
    Return the pricing preset for *model* based on a case-insensitive
    substring match.  Returns ``None`` if the model is unrecognised.
    """
    m = model.lower()
    if "haiku" in m:
        return ModelPricing.haiku()
    if "opus" in m:
        return ModelPricing.opus()
    if "sonnet" in m:
        return ModelPricing.sonnet()
    return None


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


@dataclass
class UsageCostEstimate:
    """Estimated USD cost broken down by token category."""

    input_cost_usd: float
    output_cost_usd: float
    cache_creation_cost_usd: float
    cache_read_cost_usd: float

    def total_cost_usd(self) -> float:
        """Return the sum of all cost components."""
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_creation_cost_usd
            + self.cache_read_cost_usd
        )


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token counts for one API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def total_tokens(self) -> int:
        """Return the sum of all token counts."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def estimate_cost_usd(self) -> UsageCostEstimate:
        """Estimate cost using the default Sonnet pricing tier."""
        return self.estimate_cost_usd_with_pricing(ModelPricing.sonnet())

    def estimate_cost_usd_with_pricing(self, pricing: ModelPricing) -> UsageCostEstimate:
        """Estimate cost using the given *pricing* presets."""
        M = 1_000_000
        return UsageCostEstimate(
            input_cost_usd=self.input_tokens * pricing.input_cost_per_million / M,
            output_cost_usd=self.output_tokens * pricing.output_cost_per_million / M,
            cache_creation_cost_usd=(
                self.cache_creation_input_tokens * pricing.cache_creation_cost_per_million / M
            ),
            cache_read_cost_usd=(
                self.cache_read_input_tokens * pricing.cache_read_cost_per_million / M
            ),
        )

    def summary_lines(self, label: str = "") -> list[str]:
        """Return a multi-line human-readable summary."""
        cost = self.estimate_cost_usd()
        prefix = f"{label} " if label else ""
        return [
            f"{prefix}Total tokens: {self.total_tokens():,}",
            f"  Input: {self.input_tokens:,}  Output: {self.output_tokens:,}",
            f"  Cache write: {self.cache_creation_input_tokens:,}"
            f"  Cache read: {self.cache_read_input_tokens:,}",
            f"  Estimated cost: {format_usd(cost.total_cost_usd())}",
        ]


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """
    Accumulates token usage across multiple API turns.

    Use :meth:`from_session` to initialise from a saved session, then call
    :meth:`record` after each API response.
    """

    def __init__(self) -> None:
        self._latest: TokenUsage = TokenUsage()
        self._cumulative: TokenUsage = TokenUsage()
        self._turns: int = 0

    @classmethod
    def from_session(cls, session: object) -> "UsageTracker":
        """
        Build a tracker pre-seeded with usage from all messages in
        *session* that carry a ``usage`` attribute.
        """
        tracker = cls()
        messages = getattr(session, "messages", [])
        for msg in messages:
            usage = getattr(msg, "usage", None)
            if usage is not None:
                tracker.record(usage)
        return tracker

    def record(self, usage: TokenUsage) -> None:
        """Record the usage from one API turn."""
        self._latest = usage
        self._cumulative.input_tokens += usage.input_tokens
        self._cumulative.output_tokens += usage.output_tokens
        self._cumulative.cache_creation_input_tokens += usage.cache_creation_input_tokens
        self._cumulative.cache_read_input_tokens += usage.cache_read_input_tokens
        self._turns += 1

    def current_turn_usage(self) -> TokenUsage:
        """Return the usage from the most recently recorded turn."""
        return self._latest

    def cumulative_usage(self) -> TokenUsage:
        """Return cumulative usage across all recorded turns."""
        return self._cumulative

    def turns(self) -> int:
        """Return the total number of recorded turns."""
        return self._turns


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_usd(amount: float) -> str:
    """Format *amount* as a USD string with 4 decimal places."""
    return f"${amount:.4f}"
