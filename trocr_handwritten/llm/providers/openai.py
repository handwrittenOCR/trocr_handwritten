from pathlib import Path
from typing import Tuple

from openai import OpenAI, AsyncOpenAI

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings


class OpenAIProvider(LLMProvider):
    """OpenAI provider for LLM-based OCR."""

    def __init__(self, settings: LLMSettings):
        """
        Initialize the OpenAI provider.

        Args:
            settings: LLM configuration settings.
        """
        super().__init__(settings)
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

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

    def _is_new_model(self, model: str) -> bool:
        """Check if the model uses new API parameters (GPT-5+)."""
        model_lower = model.lower()
        return (
            model_lower.startswith("gpt-5")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        )

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make a synchronous API call."""
        params = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if self._is_new_model(model):
            params["max_completion_tokens"] = self.settings.max_tokens
        else:
            params["max_tokens"] = self.settings.max_tokens

        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens

    async def _call_api_async(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make an asynchronous API call."""
        params = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if self._is_new_model(model):
            params["max_completion_tokens"] = self.settings.max_tokens
        else:
            params["max_tokens"] = self.settings.max_tokens

        response = await self.async_client.chat.completions.create(**params)
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens
