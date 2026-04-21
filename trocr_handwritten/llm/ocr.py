import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from tqdm.asyncio import tqdm_asyncio

from trocr_handwritten.utils.logging_config import get_logger
from trocr_handwritten.utils.cost_tracker import CostTracker
from trocr_handwritten.llm.settings import LLMSettings, OCRSettings
from trocr_handwritten.llm.factory import get_provider

logger = get_logger(__name__)

failed_images: Dict[str, str] = {}


def find_images(
    input_dir: Path, pattern: str, limit: Optional[int] = None
) -> List[Path]:
    """
    Find all images matching the pattern in the input directory.

    Args:
        input_dir: Root directory to search.
        pattern: Glob pattern to match images.
        limit: Maximum number of images to return.

    Returns:
        List of paths to image files.
    """
    images = sorted(input_dir.glob(pattern))
    if limit:
        images = images[:limit]
    return images


def load_prompt(prompt_path: str) -> str:
    """
    Load the OCR prompt template from file.

    Args:
        prompt_path: Path to the prompt file.

    Returns:
        Prompt template string.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


async def process_image_async(
    image_path: Path,
    provider,
    prompt: str,
    output_extension: str,
    cost_tracker: CostTracker,
    semaphore: asyncio.Semaphore,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
) -> bool:
    """
    Process a single image asynchronously and save the transcription.

    Args:
        image_path: Path to the image file.
        provider: LLM provider instance.
        prompt: Prompt template for OCR.
        output_extension: Extension for output file.
        cost_tracker: Cost tracker instance.
        semaphore: Semaphore for concurrency control.
        output_dir: If set, write predictions here preserving subfolder structure.
        input_dir: Root input directory, used to compute relative paths for output_dir.

    Returns:
        True if processed successfully, False otherwise.
    """
    if output_dir and input_dir:
        rel = image_path.relative_to(input_dir)
        output_path = output_dir / rel.with_suffix(output_extension)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = image_path.with_suffix(output_extension)

    if output_path.exists():
        logger.debug(f"Skipping {image_path.name}, output already exists")
        return True

    async with semaphore:
        try:
            transcription, input_tokens, output_tokens = await provider.ocr_image_async(
                image_path, prompt
            )
            cost_tracker.add_usage(input_tokens, output_tokens)
            if transcription is None or transcription.strip() == "":
                logger.warning(f"Empty response for {image_path} after fallbacks")
                failed_images[str(image_path)] = "empty_response"
                return False
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.debug(f"Saved transcription to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            failed_images[str(image_path)] = str(e)
            return False


async def process_all_images(
    images: List[Path],
    provider,
    prompt: str,
    output_extension: str,
    cost_tracker: CostTracker,
    max_concurrent: int = 10,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
) -> None:
    """
    Process all images concurrently with rate limiting.

    Args:
        images: List of image paths to process.
        provider: LLM provider instance.
        prompt: Prompt template for OCR.
        output_extension: Extension for output file.
        cost_tracker: Cost tracker instance.
        max_concurrent: Maximum number of concurrent requests.
        output_dir: If set, write predictions here preserving subfolder structure.
        input_dir: Root input directory, used to compute relative paths for output_dir.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        process_image_async(
            image_path,
            provider,
            prompt,
            output_extension,
            cost_tracker,
            semaphore,
            output_dir=output_dir,
            input_dir=input_dir,
        )
        for image_path in images
    ]
    await tqdm_asyncio.gather(*tasks, desc="Processing images")


def main():
    parser = argparse.ArgumentParser(
        description="Apply LLM-based OCR to processed images."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/images",
        help="Root directory containing processed images",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/*/*.jpg",
        help="Glob pattern to find images",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Limit number of images to process",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["openai", "gemini", "mistral"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default depends on provider)",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="config/ocr.prompt",
        help="Path to prompt template",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write predictions (preserves subfolder structure). "
        "If not set, predictions are written next to images.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for Gemini thinking models",
    )
    args = parser.parse_args()

    model_defaults = {
        "openai": "gpt-5.2",
        "gemini": "gemini-3-pro-preview",
        "mistral": "mistral-large-latest",
    }

    model_name = args.model or model_defaults.get(args.provider, "gemini-2.0-flash")

    llm_settings = LLMSettings(
        provider=args.provider,
        model_name=model_name,
        prompt_path=args.prompt_path,
        reasoning_effort=args.reasoning_effort,
    )

    ocr_settings = OCRSettings(
        input_dir=args.input_dir,
        image_pattern=args.pattern,
        llm_settings=llm_settings,
    )

    logger.info(f"Using provider: {llm_settings.provider}")
    logger.info(f"Using model: {llm_settings.model_name}")

    prompt = load_prompt(ocr_settings.llm_settings.prompt_path)
    provider = get_provider(llm_settings)
    cost_tracker = CostTracker(model_name=model_name)

    input_path = Path(ocr_settings.input_dir)
    images = find_images(input_path, ocr_settings.image_pattern, limit=args.n)
    logger.info(f"Found {len(images)} images to process")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        process_all_images(
            images,
            provider,
            prompt,
            ocr_settings.output_extension,
            cost_tracker,
            max_concurrent=args.max_concurrent,
            output_dir=output_dir,
            input_dir=input_path,
        )
    )

    if hasattr(provider, "total_thinking_tokens"):
        cost_tracker.thinking_tokens = provider.total_thinking_tokens

    cost_tracker.log_summary()

    if output_dir:
        summary_path = output_dir / "summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": args.provider,
            "reasoning_effort": args.reasoning_effort,
            "total_calls": cost_tracker.total_calls,
            "input_tokens": cost_tracker.input_tokens,
            "output_tokens": cost_tracker.output_tokens,
            "thinking_tokens": cost_tracker.thinking_tokens,
            "estimated_cost_usd": cost_tracker.get_cost(),
            "elapsed_seconds": round(cost_tracker.elapsed_seconds(), 1),
            "failed_count": len(failed_images),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    if failed_images:
        failed_path = input_path / "failed_ocr.json"
        failed_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": args.provider,
            "failed_count": len(failed_images),
            "images": failed_images,
        }
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.warning(
            f"Failed to process {len(failed_images)} images. See {failed_path}"
        )

    logger.info("OCR processing completed")


if __name__ == "__main__":
    main()
