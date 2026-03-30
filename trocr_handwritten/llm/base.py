import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


FALLBACK_MODELS = {
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash": "gemini-2.0-flash",
    "gpt-5.2": "gpt-5",
    "gpt-5": "gpt-4o",
    "mistral-large-latest": "pixtral-large-latest",
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, settings: LLMSettings):
        """
        Initialize the LLM provider.

        Args:
            settings: LLM configuration settings.
        """
        self.settings = settings
        self.client = None
        self.async_client = None
        self.fallback_model: Optional[str] = FALLBACK_MODELS.get(settings.model_name)

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client for the provider."""
        pass

    @abstractmethod
    def _call_api(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make a synchronous API call."""
        pass

    @abstractmethod
    async def _call_api_async(self, model: str, messages: list) -> Tuple[str, int, int]:
        """Make an asynchronous API call."""
        pass

    @abstractmethod
    def _build_messages(self, image_path: Path, prompt: str) -> list:
        """Build the messages payload for the API call."""
        pass

    def _call_with_retry(
        self, model: str, messages: list, max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Make a synchronous API call with retries."""
        total_input = 0
        total_output = 0
        for _ in range(max_retries):
            text, inp, out = self._call_api(model, messages)
            total_input += inp
            total_output += out
            if text and text.strip():
                return text, total_input, total_output
        return None, total_input, total_output

    async def _call_with_retry_async(
        self, model: str, messages: list, max_retries: int = 3, timeout: int = 60
    ) -> Tuple[str, int, int]:
        """Make an asynchronous API call with retries and per-request timeout."""
        total_input = 0
        total_output = 0
        for attempt in range(max_retries):
            try:
                text, inp, out = await asyncio.wait_for(
                    self._call_api_async(model, messages), timeout=timeout
                )
                total_input += inp
                total_output += out
                if text and text.strip():
                    return text, total_input, total_output
            except asyncio.TimeoutError:
                logger.warning(
                    f"Request to {model} timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
                )
        return None, total_input, total_output

    def ocr_image(self, image_path: Path, prompt: str) -> Tuple[str, int, int]:
        """
        Perform OCR on an image using the LLM with fallback support.

        Args:
            image_path: Path to the image file.
            prompt: Prompt template for OCR extraction.

        Returns:
            Tuple of (transcribed text, input tokens, output tokens).
        """
        messages = self._build_messages(image_path, prompt)
        text, input_tokens, output_tokens = self._call_with_retry(
            self.settings.model_name, messages
        )
        current_model = self.settings.model_name
        while (text is None or text.strip() == "") and current_model in FALLBACK_MODELS:
            fallback = FALLBACK_MODELS[current_model]
            logger.debug(
                f"Empty response from {current_model} after retries, trying fallback {fallback}"
            )
            text, fb_input, fb_output = self._call_with_retry(fallback, messages)
            input_tokens += fb_input
            output_tokens += fb_output
            current_model = fallback
        return text, input_tokens, output_tokens

    async def ocr_image_async(
        self, image_path: Path, prompt: str
    ) -> Tuple[str, int, int]:
        """
        Perform OCR on an image asynchronously with fallback support.

        Args:
            image_path: Path to the image file.
            prompt: Prompt template for OCR extraction.

        Returns:
            Tuple of (transcribed text, input tokens, output tokens).
        """
        timeout = self.settings.request_timeout
        messages = self._build_messages(image_path, prompt)
        text, input_tokens, output_tokens = await self._call_with_retry_async(
            self.settings.model_name, messages, timeout=timeout
        )
        current_model = self.settings.model_name
        while (text is None or text.strip() == "") and current_model in FALLBACK_MODELS:
            fallback = FALLBACK_MODELS[current_model]
            logger.debug(
                f"Empty response from {current_model} after retries, trying fallback {fallback}"
            )
            text, fb_input, fb_output = await self._call_with_retry_async(
                fallback, messages, timeout=timeout
            )
            input_tokens += fb_input
            output_tokens += fb_output
            current_model = fallback
        return text, input_tokens, output_tokens

    # ------------------------------------------------------------------
    # Text-only methods (for NER extraction with optional function calling)
    # ------------------------------------------------------------------

    def _build_text_messages(self, text: str, system_prompt: str) -> list:
        """Build a text-only messages payload (no image)."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

    async def call_text_async(
        self,
        text: str,
        system_prompt: str,
        tools: list = None,
        tool_choice: str = "auto",
    ) -> Tuple[str, int, int]:
        """Make an async text-only API call with optional function calling.

        Args:
            text: User text input.
            system_prompt: System prompt.
            tools: Optional list of tool/function definitions for structured output.
            tool_choice: Tool choice strategy ("auto", "required", or "none").

        Returns:
            Tuple of (response text or tool call arguments JSON, input tokens, output tokens).
        """
        messages = self._build_text_messages(text, system_prompt)
        timeout = self.settings.request_timeout
        text_result, input_tokens, output_tokens = (
            await self._call_text_with_retry_async(
                self.settings.model_name,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
            )
        )
        # Fallback if empty
        current_model = self.settings.model_name
        while (
            text_result is None or text_result.strip() == ""
        ) and current_model in FALLBACK_MODELS:
            fallback = FALLBACK_MODELS[current_model]
            logger.debug(
                f"Empty response from {current_model}, trying fallback {fallback}"
            )
            text_result, fb_input, fb_output = await self._call_text_with_retry_async(
                fallback,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
            )
            input_tokens += fb_input
            output_tokens += fb_output
            current_model = fallback
        return text_result, input_tokens, output_tokens

    async def _call_text_api_async(
        self, model: str, messages: list, tools: list = None, tool_choice: str = "auto"
    ) -> Tuple[str, int, int]:
        """Make an async text API call. Subclasses can override for provider-specific behavior."""
        raise NotImplementedError("Subclass must implement _call_text_api_async")

    async def _call_text_with_retry_async(
        self,
        model: str,
        messages: list,
        tools: list = None,
        tool_choice: str = "auto",
        max_retries: int = 3,
        timeout: int = 60,
    ) -> Tuple[str, int, int]:
        """Async text API call with retries and timeout."""
        total_input = 0
        total_output = 0
        for attempt in range(max_retries):
            try:
                text, inp, out = await asyncio.wait_for(
                    self._call_text_api_async(model, messages, tools, tool_choice),
                    timeout=timeout,
                )
                total_input += inp
                total_output += out
                if text and text.strip():
                    return text, total_input, total_output
            except asyncio.TimeoutError:
                logger.warning(
                    f"Text request to {model} timed out after {timeout}s (attempt {attempt + 1}/{max_retries})"
                )
        return None, total_input, total_output

    def _encode_image_base64(self, image_path: Path) -> str:
        """
        Encode an image file to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded string of the image.
        """
        import base64

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        """
        Get MIME type based on file extension.

        Args:
            image_path: Path to the image file.

        Returns:
            MIME type string.
        """
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/jpeg")
