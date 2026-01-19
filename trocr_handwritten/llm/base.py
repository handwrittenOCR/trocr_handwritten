from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from trocr_handwritten.llm.settings import LLMSettings


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

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client for the provider."""
        pass

    @abstractmethod
    def ocr_image(self, image_path: Path, prompt: str) -> Tuple[str, int, int]:
        """
        Perform OCR on an image using the LLM.

        Args:
            image_path: Path to the image file.
            prompt: Prompt template for OCR extraction.

        Returns:
            Tuple of (transcribed text, input tokens, output tokens).
        """
        pass

    @abstractmethod
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
        pass

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
