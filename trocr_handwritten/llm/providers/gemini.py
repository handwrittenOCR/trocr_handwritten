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

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make a synchronous API call."""
        extra_body = {}
        if "gemini-3" in model:
            extra_body["thinking"] = {"thinking_budget": 0}
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            extra_body=extra_body if extra_body else None,
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    async def _call_api_async(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make an asynchronous API call."""
        extra_body = {}
        if "gemini-3" in model:
            extra_body["thinking"] = {"thinking_budget": 0}
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            extra_body=extra_body if extra_body else None,
        )
        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens

    async def _call_text_api_async(
        self, model: str, messages: list, tools: list = None, tool_choice: str = "auto"
    ) -> Tuple[str, int, int]:
        """Make an async text API call with optional function calling."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if "gemini-3" in model:
            kwargs["extra_body"] = {"thinking": {"thinking_budget": 0}}
        response = await self.async_client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # If the model used a tool call, return the function arguments JSON
        if message.tool_calls:
            text = message.tool_calls[0].function.arguments
        else:
            text = message.content

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        return text, input_tokens, output_tokens
