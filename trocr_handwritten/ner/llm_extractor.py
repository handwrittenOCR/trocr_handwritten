"""Generic LLM-based NER extractor using function calling.

Accepts a caller-supplied tool schema and system prompt; returns raw dicts.
Domain-specific tool definitions and result parsing belong in the caller.
"""

import asyncio
import json
import logging
from typing import Any, Callable

from tqdm.asyncio import tqdm_asyncio

from trocr_handwritten.llm.factory import get_provider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Async LLM-based NER extractor using function calling.

    Args:
        settings: LLM provider and model configuration.
        prompt: System prompt instructing the model on extraction task.
        tool: OpenAI-format function-calling tool definition (single tool).
        max_concurrent: Max parallel API calls.
    """

    def __init__(
        self,
        settings: LLMSettings,
        prompt: str,
        tool: dict,
        max_concurrent: int = 10,
    ):
        self.provider = get_provider(settings)
        self.cost_tracker = CostTracker(model_name=settings.model_name)
        self.prompt = prompt
        self.tool = tool
        self.max_concurrent = max_concurrent
        self.failed: dict[str, str] = {}

    async def extract(self, record_id: str, text: str) -> dict[str, Any] | None:
        """Extract entities from a single text via function calling.

        Returns parsed JSON dict from the tool call, or None on failure.
        """
        raw_json, inp, out, think = await self.provider.call_text_async(
            text,
            self.prompt,
            tools=[self.tool],
            tool_choice="required",
        )
        self.cost_tracker.add_usage(inp, out, think)
        if not raw_json:
            return None
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON for %s: %s", record_id, raw_json[:200])
            return None

    async def extract_batch(
        self,
        records: list[Any],
        text_fn: Callable[[Any], str],
        id_fn: Callable[[Any], str],
    ) -> list[dict[str, Any] | None]:
        """Extract entities from a list of records with concurrency control.

        Args:
            records: List of input records (any type).
            text_fn: Function mapping a record to the input text string.
            id_fn: Function mapping a record to a unique record ID string.

        Returns:
            List of result dicts (same order as input); None for failed records.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _process(record: Any) -> dict[str, Any] | None:
            record_id = id_fn(record)
            async with semaphore:
                try:
                    return await self.extract(record_id, text_fn(record))
                except Exception as e:
                    logger.error("Failed to extract %s: %s", record_id, e)
                    self.failed[record_id] = str(e)
                    return None

        tasks = [_process(r) for r in records]
        return list(await tqdm_asyncio.gather(*tasks, desc="LLM NER extraction"))
