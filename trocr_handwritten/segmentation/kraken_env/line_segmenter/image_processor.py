from pathlib import Path
import json
from typing import Dict
from tqdm import tqdm
from line_segmenter.line_segmenter import LineSegmenter, LineData
from line_segmenter.settings import SegmentationSettings


class ImageProcessor:
    def __init__(
        self,
        model_path: str,
        settings: SegmentationSettings = None,
        verbose: bool = True,
    ):
        """
        Initialize the image processor

        Args:
            model_path: Path to the kraken model file
            settings: Segmentation settings (uses defaults if None)
            verbose: Whether to show progress bars and info
        """
        self.settings = settings or SegmentationSettings()
        self.line_segmenter = LineSegmenter(model_path)
        self.verbose = verbose

    def process_directory(self, input_dir: str) -> None:
        """
        Process all images in the input directory structure

        Args:
            input_dir: Directory containing 'Marge' and 'Plein Texte' subdirectories
        """
        input_path = Path(input_dir)

        # Process each subdirectory (Marge and Plein Texte)
        for subdir in ["Marge", "Plein Texte"]:
            subdir_path = input_path / subdir
            if not subdir_path.exists():
                continue

            # Process each image in the subdirectory
            image_paths = list(subdir_path.glob("*.jpg"))
            if self.verbose:
                image_paths = tqdm(image_paths, desc=f"Processing {subdir}")

            for img_path in image_paths:
                self._process_single_image(img_path)

    def _process_single_image(self, img_path: Path) -> None:
        """
        Process a single image and save its line segments

        Args:
            img_path: Path to the input image
        """
        # Create output directory for this image
        image_name = img_path.stem
        image_output_dir = img_path.parent / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Create lines and metadata directories
        lines_dir = image_output_dir / "lines"
        lines_dir.mkdir(exist_ok=True)

        # Determine if this is a margin image
        is_margin = "Marge" in str(img_path)

        # Get parameters based on image type
        params = self.settings.get_params(is_margin)

        # Segment the image and get line data
        line_data_list = self.line_segmenter.segment_image(
            str(img_path), lines_dir, **params
        )

        # Save metadata
        metadata = {
            "original_image": str(img_path),
            "lines": [self._line_data_to_dict(data) for data in line_data_list],
        }

        metadata_path = image_output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _line_data_to_dict(self, line_data: LineData) -> Dict:
        """Convert LineData object to dictionary for JSON serialization"""
        return {
            "id": line_data.id,
            "coordinates": line_data.coordinates,
            "image_path": line_data.image_path,
        }
