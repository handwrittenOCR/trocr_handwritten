from pathlib import Path
from PIL import Image
from kraken import blla
from kraken.lib import vgsl
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class LineData:
    """Class to store information about a segmented line"""

    id: int
    coordinates: Dict[str, List[Tuple[int, int]]]
    image_path: str


class LineSegmenter:
    def __init__(self, model_path: str, y_threshold: int = 10):
        """
        Initialize the line segmenter

        Args:
            model_path: Path to the kraken model file
            y_threshold: Threshold for merging lines vertically
        """
        self.model = vgsl.TorchVGSLModel.load_model(model_path)
        self.y_threshold = y_threshold

    def _should_merge_lines(self, line1: Any, line2: Any) -> bool:
        """Determine if two lines should be merged based on vertical position"""
        y1 = sum(p[1] for p in line1.baseline) / len(line1.baseline)
        y2 = sum(p[1] for p in line2.baseline) / len(line2.baseline)
        return abs(y1 - y2) < self.y_threshold

    def _merge_lines(self, lines: List[Any]) -> List[Any]:
        """Merge lines that are at similar vertical positions"""
        if not lines:
            return []

        sorted_lines = sorted(
            lines,
            key=lambda line: sum(p[1] for p in line.baseline) / len(line.baseline),
        )
        merged_lines = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if self._should_merge_lines(current_group[-1], line):
                # Merge with current group
                current_group.append(line)
            else:
                # Create new merged line from current group
                if current_group:
                    # Sort segments by x position
                    current_group.sort(key=lambda line: line.baseline[0][0])

                    # Create merged line
                    merged_baseline = [
                        current_group[0].baseline[0],
                        current_group[0].baseline[1],
                        current_group[-1].baseline[1],
                    ]
                    merged_boundary = []
                    for segment in current_group:
                        merged_boundary.extend(segment.boundary)

                    # Create new BaselineLine object with merged data
                    merged_line = type(current_group[0])(
                        id=current_group[0].id,
                        baseline=merged_baseline,
                        boundary=merged_boundary,
                        text=None,
                        base_dir=None,
                        type="baselines",
                        imagename=None,
                        tags={"type": "default"},
                        split=None,
                        regions=current_group[0].regions,
                    )
                    merged_lines.append(merged_line)

                # Start new group
                current_group = [line]

        # Handle last group
        if current_group:
            current_group.sort(key=lambda line: line.baseline[0][0])
            merged_baseline = []
            merged_boundary = []
            for segment in current_group:
                merged_baseline.extend(segment.baseline)
                merged_boundary.extend(segment.boundary)

            merged_line = type(current_group[0])(
                id=current_group[0].id,
                baseline=merged_baseline,
                boundary=merged_boundary,
                text=None,
                base_dir=None,
                type="baselines",
                imagename=None,
                tags={"type": "default"},
                split=None,
                regions=current_group[0].regions,
            )
            merged_lines.append(merged_line)

        return merged_lines

    def calculate_iou(self, box1: Tuple[int, int], box2: Tuple[int, int]) -> float:
        """Calculate IoU between two boxes represented as (y_min, y_max)"""
        y_min_inter = max(box1[0], box2[0])
        y_max_inter = min(box1[1], box2[1])

        if y_max_inter <= y_min_inter:
            return 0.0

        intersection = y_max_inter - y_min_inter
        box1_area = box1[1] - box1[0]
        box2_area = box2[1] - box2[0]
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0

    def segment_image(
        self,
        image_path: str,
        output_dir: Path,
        is_margin: bool = False,
        padding: int = 15,
        iou_threshold: float = 0.5,
    ) -> List[LineData]:
        """
        Segment an image into lines and save them

        Args:
            image_path: Path to the input image
            output_dir: Directory to save segmented lines
            is_margin: If True, process as margin text, else as main text
            padding: Padding for line extraction (width for margins, height for main text)
            iou_threshold: IoU threshold for filtering overlapping lines (main text only)

        Returns:
            List of LineData objects containing information about each line
        """
        im = Image.open(image_path)

        segmentation = blla.segment(im, model=self.model)

        merged_lines = self._merge_lines(segmentation.lines)

        line_data_list = []
        crop_boxes = []  # To store y_min, y_max for IoU calculation

        for idx, line in enumerate(merged_lines):
            x_coords = [p[0] for p in line.boundary]
            y_coords = [p[1] for p in line.boundary]

            if is_margin:
                # For margins, use detected boundaries with width padding
                x1 = max(0, min(x_coords) - padding)
                x2 = min(im.width, max(x_coords) + padding)
                y1 = min(y_coords)
                y2 = max(y_coords)
            else:
                # For main text, use full width and check IoU
                x1 = 0
                x2 = im.width
                y1 = max(0, min(y_coords) - padding)
                y2 = min(im.height, max(y_coords) + padding)

                # Check IoU with previous crops
                current_box = (y1, y2)
                if any(
                    self.calculate_iou(current_box, prev_box) > iou_threshold
                    for prev_box in crop_boxes
                ):
                    continue
                crop_boxes.append(current_box)

            # Crop and save line image
            line_img = im.crop((x1, y1, x2, y2))
            line_filename = f"line_{idx+1}.jpg"
            line_path = output_dir / line_filename
            line_img.save(line_path)

            # Create line data object
            line_data = LineData(
                id=idx + 1,
                coordinates={
                    "boundary": [(int(x), int(y)) for x, y in line.boundary],
                    "baseline": [(int(x), int(y)) for x, y in line.baseline],
                    "crop": [(x1, y1), (x2, y2)],
                },
                image_path=str(line_path.relative_to(output_dir)),
            )
            line_data_list.append(line_data)

        return line_data_list
