from dataclasses import dataclass, field
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
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
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
    total_calls: int = 0
    _pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: PRICING)

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """
        Add token usage from an API call.

        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1

    def get_cost(self) -> float:
        """
        Calculate total cost in USD.

        Returns:
            Total cost based on token usage and model pricing.
        """
        pricing = self._pricing.get(self.model_name, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def summary(self) -> str:
        """
        Generate a summary of usage and costs.

        Returns:
            Formatted string with usage statistics.
        """
        cost = self.get_cost()
        return (
            f"Model: {self.model_name}\n"
            f"Total calls: {self.total_calls}\n"
            f"Input tokens: {self.input_tokens:,}\n"
            f"Output tokens: {self.output_tokens:,}\n"
            f"Estimated cost: ${cost:.4f}"
        )

    def log_summary(self) -> None:
        """Log the usage summary."""
        logger.info(f"\n{'='*40}\nCost Summary\n{'='*40}\n{self.summary()}")
