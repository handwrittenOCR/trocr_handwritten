from pathlib import Path
import json
from typing import Dict, List, NamedTuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import logging
from line_segmenter.line_segmenter import LineSegmenter, LineData
from line_segmenter.settings import SegmentationSettings


class ImageTask(NamedTuple):
    """Structure to keep track of image processing tasks"""

    input_path: Path
    output_dir: Path
    is_margin: bool
    folder_name: str


class ImageProcessor:
    def __init__(
        self,
        model_path: str,
        settings: SegmentationSettings = None,
        verbose: bool = True,
        max_workers: int = 4,
        device: str = "cpu",
        logger: logging.Logger = None,
    ):
        """
        Initialize the image processor

        Args:
            model_path: Path to the kraken model file
            settings: Segmentation settings (uses defaults if None)
            verbose: Whether to show progress bars and info
            max_workers: Maximum number of parallel workers
            device: Device to use ('cpu' or 'cuda')
            logger: Logger instance to use
        """
        self.settings = settings or SegmentationSettings()
        self.verbose = verbose
        self.max_workers = max_workers
        self.device = device
        self.model_path = model_path
        # We'll create line segmenters per worker to avoid sharing issues
        self.semaphore = asyncio.Semaphore(max_workers)
        self.batch_times = {}
        self.logger = logger or logging.getLogger(__name__)

    async def process_all_directories(self, image_folders: List[Path]) -> None:
        """
        Process all images from multiple directories in parallel

        Args:
            image_folders: List of directories containing images to process
        """
        start_time = time.time()

        # Collect all image tasks
        all_tasks: List[ImageTask] = []
        for folder in image_folders:
            tasks = self._collect_image_tasks(folder)
            all_tasks.extend(tasks)

        if not all_tasks:
            self.logger.warning("No images found to process")
            return

        # Create batches of tasks
        batch_size = max(1, len(all_tasks) // self.max_workers)
        batches = [
            all_tasks[i : i + batch_size] for i in range(0, len(all_tasks), batch_size)
        ]

        self.logger.info(
            f"Starting processing of {len(all_tasks)} images in {len(batches)} "
            f"batches with {self.max_workers} workers on {self.device}"
        )
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create and run processing tasks
        processing_tasks = []
        for batch_idx, batch in enumerate(batches):
            task = asyncio.create_task(self._process_batch_images(batch, batch_idx))
            processing_tasks.append(task)

        # Wait for all batches to complete
        await asyncio.gather(*processing_tasks)

        # Log summary
        end_time = time.time()
        total_time = end_time - start_time
        self._log_summary(total_time, len(all_tasks), len(batches))

    def _collect_image_tasks(self, folder: Path) -> List[ImageTask]:
        """Collect all image tasks from a folder"""
        tasks = []

        for subdir in ["Marge", "Plein Texte"]:
            subdir_path = folder / subdir
            if not subdir_path.exists() or not subdir_path.is_dir():
                self.logger.debug(
                    f"Skipping non-existent or invalid directory: {subdir_path}"
                )
                continue

            is_margin = subdir == "Marge"

            for img_path in subdir_path.glob("*.jpg"):
                # Create output directory structure
                image_name = img_path.stem
                image_output_dir = subdir_path / image_name

                tasks.append(
                    ImageTask(
                        input_path=img_path,
                        output_dir=image_output_dir,
                        is_margin=is_margin,
                        folder_name=folder.name,
                    )
                )

        return tasks

    async def _process_batch_images(
        self, image_tasks: List[ImageTask], batch_idx: int
    ) -> None:
        """Process a batch of images"""
        batch_start_time = time.time()

        async with self.semaphore:
            line_segmenter = LineSegmenter(self.model_path, device=self.device)

            for idx, task in enumerate(image_tasks):
                self.logger.debug(
                    f"Batch {batch_idx}, processing image {idx+1}/{len(image_tasks)}: "
                    f"{task.folder_name}/{task.input_path.parent.name}/{task.input_path.name}"
                )
                await self._process_single_image(task, line_segmenter)

        batch_time = time.time() - batch_start_time
        self.batch_times[batch_idx] = batch_time
        self.logger.info(f"Completed batch {batch_idx} in {batch_time:.2f} seconds")

    async def _process_single_image(
        self, task: ImageTask, line_segmenter: LineSegmenter
    ) -> None:
        """Process a single image task"""
        # Create output directories
        task.output_dir.mkdir(parents=True, exist_ok=True)
        lines_dir = task.output_dir / "lines"
        lines_dir.mkdir(exist_ok=True)

        # Get parameters based on image type
        params = self.settings.get_params(task.is_margin)

        # Process image
        with ThreadPoolExecutor() as executor:
            line_data_list = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: line_segmenter.segment_image(
                    str(task.input_path), lines_dir, **params
                ),
            )

        # Save metadata
        metadata = {
            "original_image": str(task.input_path),
            "lines": [self._line_data_to_dict(data) for data in line_data_list],
            "folder": task.folder_name,
            "type": "margin" if task.is_margin else "main_text",
        }

        metadata_path = task.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _line_data_to_dict(self, line_data: LineData) -> Dict:
        """Convert LineData object to dictionary for JSON serialization"""
        return {
            "id": line_data.id,
            "coordinates": line_data.coordinates,
            "image_path": line_data.image_path,
        }

    def _log_summary(
        self, total_time: float, total_images: int, total_batches: int
    ) -> None:
        """Log processing summary"""
        self.logger.info("\nProcessing Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Total processing time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Total images processed: {total_images}")
        self.logger.info(
            f"Average time per image: {total_time/total_images:.2f} seconds"
        )
        self.logger.info(
            f"Average time per batch: {total_time/total_batches:.2f} seconds"
        )

        self.logger.info("\nBatch Processing Times:")
        self.logger.info("-" * 50)
        for batch_idx, batch_time in sorted(self.batch_times.items()):
            self.logger.info(f"Batch {batch_idx}: {batch_time:.2f} seconds")

        self.logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
