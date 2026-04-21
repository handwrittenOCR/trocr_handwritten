from pathlib import Path
from typing import Tuple

from openai import OpenAI, AsyncOpenAI

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using OpenAI-compatible SDK."""

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, settings: LLMSettings):
        """
        Initialize the Gemini provider.

        Args:
            settings: LLM configuration settings.
        """
        super().__init__(settings)
        self.total_thinking_tokens = 0
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI-compatible client for Gemini."""
        self.client = OpenAI(
            api_key=self.settings.google_api_key,
            base_url=self.GEMINI_BASE_URL,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.settings.google_api_key,
            base_url=self.GEMINI_BASE_URL,
        )

    def _build_messages(self, image_path: Path, prompt: str) -> list:
        """Build the messages payload for the API call."""
        base64_image = self._encode_image_base64(image_path)
        mime_type = self._get_mime_type(image_path)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]

    def _build_kwargs(self, model: str, messages: list) -> dict:
        """Build shared kwargs for API calls."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        return kwargs

    def _extract_thinking_tokens(self, response) -> int:
        """Extract thinking/reasoning tokens from response usage details."""
        if not response.usage:
            return 0
        total = response.usage.total_tokens or 0
        prompt = response.usage.prompt_tokens or 0
        completion = response.usage.completion_tokens or 0
        thinking = total - prompt - completion
        return thinking if thinking > 0 else 0

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make a synchronous API call."""
        response = self.client.chat.completions.create(
            **self._build_kwargs(model, messages)
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        self.total_thinking_tokens += self._extract_thinking_tokens(response)
        return text, input_tokens, output_tokens

    async def _call_api_async(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make an asynchronous API call."""
        response = await self.async_client.chat.completions.create(
            **self._build_kwargs(model, messages)
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        self.total_thinking_tokens += self._extract_thinking_tokens(response)
        return text, input_tokens, output_tokens
