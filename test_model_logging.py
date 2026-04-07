"""Quick test to verify actual model name is logged from API response."""

import asyncio
from pathlib import Path
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.llm.factory import get_provider
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

IMAGE = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\OCR_gem31\vieux_fort\1839"
    r"\FRAD971_1E35_136_130_002_C\Marge\001.jpg"
)


async def main():
    settings = LLMSettings()
    print(f"Requested model: {settings.model_name}")
    provider = get_provider(settings)
    prompt = open("config/ocr.prompt").read()
    text, inp, out, think = await provider.ocr_image_async(IMAGE, prompt)
    actual = getattr(provider, "actual_model_name", settings.model_name)
    print(f"Actual model used: {actual}")
    print(f"\nTranscription: {text}")
    print(f"Tokens — input: {inp}, output: {out}, thinking: {think}")


asyncio.run(main())
