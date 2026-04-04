import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from tqdm.asyncio import tqdm_asyncio

from trocr_handwritten.utils.logging_config import get_logger
from trocr_handwritten.utils.cost_tracker import CostTracker, BudgetExceeded
from trocr_handwritten.llm.settings import LLMSettings, OCRSettings
from trocr_handwritten.llm.factory import get_provider

logger = get_logger(__name__)

failed_images: Dict[str, str] = {}


def find_images(
    input_dir: Path,
    pattern: str,
    limit: Optional[int] = None,
    output_extension: str = ".md",
) -> List[Path]:
    """
    Find images matching the pattern that have not yet been transcribed.

    Args:
        input_dir: Root directory to search.
        pattern: Glob pattern to match images.
        limit: Maximum number of *untranscribed* images to return.
        output_extension: Extension of output files (to filter already-done).

    Returns:
        List of paths to image files.
    """
    all_images = sorted(input_dir.rglob(pattern))
    untranscribed = [
        img for img in all_images if not img.with_suffix(output_extension).exists()
    ]
    logger.info(
        f"Found {len(all_images)} total images, {len(untranscribed)} untranscribed"
    )
    if limit:
        untranscribed = untranscribed[:limit]
    return untranscribed


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

    Returns:
        True if processed successfully, False otherwise.
    """
    output_path = image_path.with_suffix(output_extension)

    if output_path.exists():
        logger.debug(f"Skipping {image_path.name}, output already exists")
        return True

    async with semaphore:
        try:
            transcription, input_tokens, output_tokens, thinking_tokens = (
                await provider.ocr_image_async(image_path, prompt)
            )
            cost_tracker.add_usage(input_tokens, output_tokens, thinking_tokens)
            if transcription is None or transcription.strip() == "":
                logger.warning(f"Empty response for {image_path} after fallbacks")
                failed_images[str(image_path)] = "empty_response"
                return False
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            logger.debug(f"Saved transcription to {output_path}")
            return True
        except BudgetExceeded:
            raise
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
) -> None:
    """
    Process all images concurrently with rate limiting.

    Filters out already-processed images before starting, then processes
    remaining ones with a semaphore for concurrency control.

    Args:
        images: List of image paths to process.
        provider: LLM provider instance.
        prompt: Prompt template for OCR.
        output_extension: Extension for output file.
        cost_tracker: Cost tracker instance.
        max_concurrent: Maximum number of concurrent requests.
    """
    # Filter out already-processed images upfront
    todo = [img for img in images if not img.with_suffix(output_extension).exists()]
    skipped = len(images) - len(todo)
    if skipped > 0:
        logger.info(f"Skipping {skipped} already transcribed, {len(todo)} remaining")

    if not todo:
        logger.info("All images already transcribed")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        process_image_async(
            image_path, provider, prompt, output_extension, cost_tracker, semaphore
        )
        for image_path in todo
    ]
    try:
        await tqdm_asyncio.gather(*tasks, desc="Processing images")
    except BudgetExceeded:
        logger.warning("Budget exceeded — stopping. Completed work has been saved.")


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
        default="*.jpg",
        help="Glob pattern to find images",
        # modified to use *.jpg, works better with testing file organization
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
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds before retry (default: 60)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="Maximum spend in EUR before stopping (default: 0 = no limit)",
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
        request_timeout=args.timeout,
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
    cost_tracker = CostTracker(model_name=model_name, budget_eur=args.budget)

    input_path = Path(ocr_settings.input_dir)
    images = find_images(
        input_path,
        ocr_settings.image_pattern,
        limit=args.n,
        output_extension=ocr_settings.output_extension,
    )

    asyncio.run(
        process_all_images(
            images,
            provider,
            prompt,
            ocr_settings.output_extension,
            cost_tracker,
            max_concurrent=args.max_concurrent,
        )
    )

    cost_tracker.model_name = getattr(provider, "actual_model_name", model_name)
    cost_tracker.log_summary()

    # Reconcile against disk: any .jpg without a corresponding .md is failed
    all_jpgs = sorted(input_path.rglob(ocr_settings.image_pattern))
    still_failed = {
        str(img): failed_images.get(str(img), "no corresponding output file")
        for img in all_jpgs
        if not img.with_suffix(ocr_settings.output_extension).exists()
    }

    failed_path = input_path / "failed_ocr.json"
    if still_failed:
        failed_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": args.provider,
            "failed_count": len(still_failed),
            "images": still_failed,
        }
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.warning(
            f"{len(still_failed)} images still without transcription. See {failed_path}"
        )
    elif failed_path.exists():
        failed_path.unlink()
        logger.info("All images transcribed — removed stale failed_ocr.json")

    logger.info("OCR processing completed")


if __name__ == "__main__":
    main()
