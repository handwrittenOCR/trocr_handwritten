from pathlib import Path
from image_processor import ImageProcessor
import urllib.request
from datetime import datetime
from settings import SegmentationSettings
import asyncio
import time
from datetime import timedelta
import logging
import sys


def setup_logging(log_file: Path = None, verbose: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("line_segmentation")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def download_model(model_path: Path, logger: logging.Logger):
    """Download the Kraken blla model if it doesn't exist"""
    if not model_path.exists():
        logger.info(f"Downloading model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/mittagessen/kraken/raw/main/kraken/blla.mlmodel"
        urllib.request.urlretrieve(url, model_path)
        logger.info("Model downloaded successfully!")


async def main(settings: SegmentationSettings):
    # Setup logging
    log_dir = settings.root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)

    start_time = time.time()
    logger.info("Starting image processing pipeline...")
    logger.info("-" * 50)

    # Configure paths
    model_path = settings.model_path
    # download_model(model_path, logger)
    input_dir = settings.input_dir

    # Collect all valid directories
    image_folders = [p for p in input_dir.glob("*") if p.is_dir()]
    if not image_folders:
        logger.warning(f"No valid directories found in {input_dir}")
        return

    # Create processor and process all directories
    processor = ImageProcessor(
        str(model_path), settings, max_workers=4, device="cpu", logger=logger
    )

    # Process all folders at once
    await processor.process_all_directories(image_folders)

    total_time = time.time() - start_time
    logger.info(
        f"\nTotal pipeline execution time: {timedelta(seconds=int(total_time))}"
    )


if __name__ == "__main__":
    settings = SegmentationSettings()
    asyncio.run(main(settings))
