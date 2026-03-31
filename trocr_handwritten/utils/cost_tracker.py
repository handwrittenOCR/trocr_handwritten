import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict

from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    # Gemini prices in EUR/M tokens (from Google billing CSV 2026-03-25)
    "gemini-2.0-flash": {"input": 0.085, "output": 0.339},
    "gemini-2.0-flash-lite": {"input": 0.064, "output": 0.254},
    "gemini-2.5-pro": {"input": 1.060, "output": 8.482},
    "gemini-2.5-flash-lite": {"input": 0.085, "output": 0.339},
    "gemini-3-pro-preview": {"input": 1.696, "output": 10.178},
    "gemini-3-flash-preview": {"input": 0.424, "output": 2.545},
    "gemini-3.1-pro-preview": {"input": 1.696, "output": 10.178},
    "gemini-3.1-flash-lite-preview": {"input": 0.212, "output": 1.272},
    "mistral-large-latest": {"input": 0.50, "output": 1.50},
    "pixtral-large-latest": {"input": 2.00, "output": 6.00},
    "ministral-3b-2512": {"input": 0.10, "output": 0.10},
    "ministral-8b-2512": {"input": 0.15, "output": 0.15},
    "ministral-14b-2512": {"input": 0.20, "output": 0.20},
}


@dataclass
class CostTracker:
    """Tracks API usage costs across multiple calls."""

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_calls: int = 0
    _pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: PRICING)

    def add_usage(
        self, input_tokens: int, output_tokens: int, thinking_tokens: int = 0
    ) -> None:
        """
        Add token usage from an API call.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            thinking_tokens: Number of thinking/reasoning tokens (billed as output).
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.thinking_tokens += thinking_tokens
        self.total_calls += 1

        if thinking_tokens > 0:
            logger.warning(
                f"THINKING TOKENS DETECTED: {thinking_tokens} tokens in this call "
                f"(cumulative: {self.thinking_tokens}). "
                f"These are billed at output token rate!"
            )

    def get_cost(self) -> Dict[str, float]:
        """
        Calculate total cost breakdown in EUR.

        Returns:
            Dict with input_cost, output_cost, thinking_cost, and total.
        """
        pricing = self._pricing.get(self.model_name, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        # Thinking tokens are billed at output token rate
        thinking_cost = (self.thinking_tokens / 1_000_000) * pricing["output"]
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "thinking_cost": thinking_cost,
            "total": input_cost + output_cost + thinking_cost,
        }

    def get_total_cost(self) -> float:
        """Get total cost as a single float (backward compat)."""
        return self.get_cost()["total"]

    def summary(self) -> str:
        """
        Generate a detailed summary of usage and costs.

        Returns:
            Formatted string with usage statistics.
        """
        costs = self.get_cost()
        lines = [
            f"Model: {self.model_name}",
            f"Total calls: {self.total_calls}",
            f"Input tokens:    {self.input_tokens:>12,}  (EUR {costs['input_cost']:.4f})",
            f"Output tokens:   {self.output_tokens:>12,}  (EUR {costs['output_cost']:.4f})",
            f"Thinking tokens: {self.thinking_tokens:>12,}  (EUR {costs['thinking_cost']:.4f})",
            f"{'-' * 50}",
            f"TOTAL ESTIMATED COST: EUR {costs['total']:.4f}",
        ]
        if self.thinking_tokens > 0:
            lines.append(
                f"WARNING: {self.thinking_tokens:,} thinking tokens detected! "
                f"Verify your Google billing matches this estimate."
            )
        return "\n".join(lines)

    def log_summary(self, log_dir: str = "logs") -> None:
        """Log the usage summary and append to persistent cost log."""
        logger.info(f"\n{'='*50}\nCost Summary\n{'='*50}\n{self.summary()}")
        self._append_to_cost_log(log_dir)

    def _append_to_cost_log(self, log_dir: str = "logs") -> None:
        """Append cost entry to a persistent JSON Lines log file."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        cost_log = log_path / "api_costs.jsonl"

        costs = self.get_cost()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "total_calls": self.total_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_input_eur": round(costs["input_cost"], 6),
            "cost_output_eur": round(costs["output_cost"], 6),
            "cost_thinking_eur": round(costs["thinking_cost"], 6),
            "cost_total_eur": round(costs["total"], 6),
        }

        with open(cost_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Cost entry appended to {cost_log}")
