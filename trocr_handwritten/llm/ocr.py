import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from tqdm.asyncio import tqdm_asyncio

from trocr_handwritten.utils.logging_config import get_logger
from trocr_handwritten.utils.cost_tracker import CostTracker
from trocr_handwritten.utils.prompt import load_prompt
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


def run_once(
    images,
    provider,
    prompt,
    output_extension,
    model_name,
    provider_name,
    max_concurrent,
    output_dir,
    input_path,
    reasoning_effort,
):
    """
    Run one full inference pass and write predictions, a summary and failures.

    Returns the CostTracker so callers can aggregate throughput/cost across passes.
    """
    failed_images.clear()
    cost_tracker = CostTracker(model_name=model_name)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        process_all_images(
            images,
            provider,
            prompt,
            output_extension,
            cost_tracker,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
            input_dir=input_path,
        )
    )

    if hasattr(provider, "total_thinking_tokens"):
        cost_tracker.thinking_tokens = provider.total_thinking_tokens
    cost_tracker.log_summary()

    if output_dir:
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": provider_name,
            "reasoning_effort": reasoning_effort,
            "total_calls": cost_tracker.total_calls,
            "input_tokens": cost_tracker.input_tokens,
            "output_tokens": cost_tracker.output_tokens,
            "thinking_tokens": cost_tracker.thinking_tokens,
            "estimated_cost_usd": cost_tracker.get_cost(),
            "elapsed_seconds": round(cost_tracker.elapsed_seconds(), 1),
            "failed_count": len(failed_images),
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    if failed_images:
        failed_root = output_dir if output_dir else input_path
        failed_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "provider": provider_name,
            "failed_count": len(failed_images),
            "images": dict(failed_images),
        }
        with open(failed_root / "failed_ocr.json", "w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.warning(
            f"Failed to process {len(failed_images)} images. "
            f"See {failed_root / 'failed_ocr.json'}"
        )

    return cost_tracker


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
        choices=["openai", "gemini", "mistral", "vllm"],
        help="LLM provider to use",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help="Base URL of the vLLM server (overrides VLLM_BASE_URL env var). "
        "Required when --provider=vllm.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default=None,
        help="API key for the vLLM server (overrides VLLM_API_KEY env var).",
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
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for Gemini models. Use 'none' to disable thinking "
        "(Flash models; Pro models require thinking).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max output tokens per call (defaults to LLMSettings.max_tokens).",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=None,
        help="Per-request HTTP timeout in seconds (defaults to 120).",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="vLLM-only: disable <think> block for Qwen3/Qwen3-Next models.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run inference N times into <output_dir>/run{i} to measure "
        "variability. Requires --output_dir when > 1.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (defaults to LLMSettings). Set > 0 so repeated "
        "runs actually differ; greedy (0.0) makes --repeat a no-op.",
    )
    args = parser.parse_args()

    if args.repeat > 1 and not args.output_dir:
        parser.error("--repeat > 1 requires --output_dir")

    model_defaults = {
        "openai": "gpt-5.2",
        "gemini": "gemini-3-pro-preview",
        "mistral": "mistral-large-latest",
    }

    if args.provider == "vllm" and not args.model:
        parser.error(
            "--model is required when --provider=vllm (HF repo id served by vLLM)"
        )

    model_name = args.model or model_defaults.get(args.provider, "gemini-2.0-flash")

    llm_kwargs = {
        "provider": args.provider,
        "model_name": model_name,
        "prompt_path": args.prompt_path,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.vllm_base_url:
        llm_kwargs["vllm_base_url"] = args.vllm_base_url
    if args.vllm_api_key:
        llm_kwargs["vllm_api_key"] = args.vllm_api_key
    if args.max_tokens is not None:
        llm_kwargs["max_tokens"] = args.max_tokens
    if args.request_timeout is not None:
        llm_kwargs["request_timeout"] = args.request_timeout
    if args.temperature is not None:
        llm_kwargs["temperature"] = args.temperature
    if args.disable_thinking:
        llm_kwargs["disable_thinking"] = True

    llm_settings = LLMSettings(**llm_kwargs)

    ocr_settings = OCRSettings(
        input_dir=args.input_dir,
        image_pattern=args.pattern,
        llm_settings=llm_settings,
    )

    logger.info(f"Using provider: {llm_settings.provider}")
    logger.info(f"Using model: {llm_settings.model_name}")

    prompt = load_prompt(ocr_settings.llm_settings.prompt_path)
    provider = get_provider(llm_settings)

    input_path = Path(ocr_settings.input_dir)
    images = find_images(input_path, ocr_settings.image_pattern, limit=args.n)
    logger.info(f"Found {len(images)} images to process")

    base_dir = Path(args.output_dir) if args.output_dir else None
    for r in range(args.repeat):
        if base_dir and args.repeat > 1:
            run_dir = base_dir / f"run{r + 1}"
            logger.info(f"=== Run {r + 1}/{args.repeat} -> {run_dir} ===")
        else:
            run_dir = base_dir
        run_once(
            images,
            provider,
            prompt,
            ocr_settings.output_extension,
            model_name,
            args.provider,
            args.max_concurrent,
            run_dir,
            input_path,
            args.reasoning_effort,
        )

    logger.info("OCR processing completed")


if __name__ == "__main__":
    main()
