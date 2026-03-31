import asyncio
from pathlib import Path
from typing import Tuple

from google import genai
from google.genai import types

from trocr_handwritten.llm.base import LLMProvider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the native google-genai SDK."""

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)
        self.actual_model_name: str = settings.model_name
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the native Gemini client."""
        self.native_client = genai.Client(api_key=self.settings.google_api_key)

    def _build_messages(self, image_path: Path, prompt: str) -> list:
        """Build native SDK content parts for an image + prompt."""
        image_bytes = image_path.read_bytes()
        mime_type = self._get_mime_type(image_path)
        return [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ]

    def _get_config(self, model: str) -> types.GenerateContentConfig:
        """Build generation config with appropriate thinking level."""
        thinking_level = None
        if "gemini-3" in model:
            thinking_level = "LOW"
        elif "gemini-2.5" in model:
            thinking_level = "OFF"

        thinking_config = None
        if thinking_level:
            thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

        return types.GenerateContentConfig(
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
            thinking_config=thinking_config,
        )

    @staticmethod
    def _extract_tokens(response) -> Tuple[int, int, int]:
        """Extract input, output, and thinking tokens from native SDK response."""
        um = response.usage_metadata
        if not um:
            return 0, 0, 0

        input_tokens = um.prompt_token_count or 0
        output_tokens = um.candidates_token_count or 0
        thinking_tokens = um.thoughts_token_count or 0

        if thinking_tokens > 0:
            logger.info(
                f"Tokens: input={input_tokens}, output={output_tokens}, "
                f"thinking={thinking_tokens}"
            )

        return input_tokens, output_tokens, thinking_tokens

    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int, int]:
        """Make a synchronous API call via native SDK."""
        config = self._get_config(model)
        response = self.native_client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )
        actual_model = getattr(response, "model_version", None)
        if actual_model and actual_model != model:
            logger.info(f"Requested model '{model}' redirected to '{actual_model}'")
            self.actual_model_name = actual_model
        text = response.text
        input_tokens, output_tokens, thinking_tokens = self._extract_tokens(response)
        return text, input_tokens, output_tokens, thinking_tokens

    async def _call_api_async(
        self, model: str, messages: list
    ) -> Tuple[str, int, int, int]:
        """Make an async API call (runs sync call in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_api, model, messages)

    async def _call_text_api_async(
        self, model: str, messages: list, tools: list = None, tool_choice: str = "auto"
    ) -> Tuple[str, int, int, int]:
        """Make an async text API call with optional function calling.

        Note: tools here use OpenAI format from NER pipeline. We convert to
        native SDK tool definitions.
        """
        config = self._get_config(model)

        if tools:
            # For function calling, fall back to OpenAI-compat endpoint
            # since tool schemas come in OpenAI format from the NER pipeline
            return await self._call_text_openai_compat(
                model, messages, tools, tool_choice
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.native_client.models.generate_content(
                model=model,
                contents=messages,
                config=config,
            ),
        )
        actual_model = getattr(response, "model_version", None)
        if actual_model and actual_model != model:
            logger.info(f"Requested model '{model}' redirected to '{actual_model}'")
            self.actual_model_name = actual_model
        text = response.text
        input_tokens, output_tokens, thinking_tokens = self._extract_tokens(response)
        return text, input_tokens, output_tokens, thinking_tokens

    async def _call_text_openai_compat(
        self, model: str, messages: list, tools: list, tool_choice: str
    ) -> Tuple[str, int, int, int]:
        """Fallback to OpenAI-compat endpoint for function calling (NER pipeline)."""
        from openai import AsyncOpenAI

        async_client = AsyncOpenAI(
            api_key=self.settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if "gemini-3" in model:
            kwargs["reasoning_effort"] = "low"
        elif "gemini-2.5" in model:
            kwargs["reasoning_effort"] = "none"

        response = await async_client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        if message.tool_calls:
            text = message.tool_calls[0].function.arguments
        else:
            text = message.content

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        # OpenAI compat doesn't report thinking tokens separately
        logger.warning(
            "Function calling via OpenAI-compat: thinking tokens not tracked separately"
        )
        return text, input_tokens, output_tokens, 0
