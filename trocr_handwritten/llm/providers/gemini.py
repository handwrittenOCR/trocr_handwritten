from pathlib import Path
from typing import Tuple

from openai import OpenAI, AsyncOpenAI

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings


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

    def ocr_image(self, image_path: Path, prompt: str) -> Tuple[str, int, int]:
        """
        Perform OCR on an image using Gemini vision models.

        Args:
            image_path: Path to the image file.
            prompt: Prompt template for OCR extraction.

        Returns:
            Tuple of (transcribed text, input tokens, output tokens).
        """
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=self._build_messages(image_path, prompt),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    async def ocr_image_async(
        self, image_path: Path, prompt: str
    ) -> Tuple[str, int, int]:
        """
        Perform OCR on an image asynchronously.

        Args:
            image_path: Path to the image file.
            prompt: Prompt template for OCR extraction.

        Returns:
            Tuple of (transcribed text, input tokens, output tokens).
        """
        response = await self.async_client.chat.completions.create(
            model=self.settings.model_name,
            messages=self._build_messages(image_path, prompt),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens
