from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


FALLBACK_MODELS = {
    "gemini-3-pro-preview": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.0-flash",
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
        self, model: str, messages: list, max_retries: int = 3
    ) -> Tuple[str, int, int]:
        """Make an asynchronous API call with retries."""
        total_input = 0
        total_output = 0
        for _ in range(max_retries):
            text, inp, out = await self._call_api_async(model, messages)
            total_input += inp
            total_output += out
            if text and text.strip():
                return text, total_input, total_output
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
        messages = self._build_messages(image_path, prompt)
        text, input_tokens, output_tokens = await self._call_with_retry_async(
            self.settings.model_name, messages
        )
        current_model = self.settings.model_name
        while (text is None or text.strip() == "") and current_model in FALLBACK_MODELS:
            fallback = FALLBACK_MODELS[current_model]
            logger.debug(
                f"Empty response from {current_model} after retries, trying fallback {fallback}"
            )
            text, fb_input, fb_output = await self._call_with_retry_async(
                fallback, messages
            )
            input_tokens += fb_input
            output_tokens += fb_output
            current_model = fallback
        return text, input_tokens, output_tokens

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
