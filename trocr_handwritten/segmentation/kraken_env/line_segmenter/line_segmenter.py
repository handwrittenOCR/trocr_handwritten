from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from kraken import blla
from kraken.lib import vgsl
from kraken.containers import Segmentation
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
import copy


@dataclass
class LineData:
    """Class to store information about a segmented line"""

    id: int
    coordinates: Dict[str, List[Tuple[int, int]]]
    image_path: str


class LineSegmenter:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the line segmenter

        Args:
            model_path: Path to the kraken model file
            device: Device to use for inference ('cpu' or 'cuda')
        """
        self.device = device
        self.model = vgsl.TorchVGSLModel.load_model(model_path)
        self.model.to(self.device)

    def merge_similar_lines(
        self,
        image_path: str,
        segmentation: Segmentation,
        y_threshold: int = 20,
        padding: int = 20,
    ) -> Tuple[Segmentation, List[Image.Image]]:
        """
        Merges segmentation lines that are on the same approximate y-level

        Parameters:
            image_path: Path to the original image
            segmentation: Segmentation object containing lines
            y_threshold: Maximum difference in baseline mean y values for lines to be considered on the same level
            padding: Padding to add to line crops

        Returns:
            Tuple of (new_segmentation, line_images) where:
                new_segmentation: A new segmentation object with merged lines
                line_images: List of PIL Images containing the cropped line images
        """
        im = Image.open(image_path)

        # Create a copy of the segmentation object to avoid modifying the original
        merged_segmentation = copy.deepcopy(segmentation)

        # Create an empty segmentation object
        new_segmentation = Segmentation(
            type=segmentation.type,
            imagename=segmentation.imagename,
            text_direction=segmentation.text_direction,
            script_detection=segmentation.script_detection,
        )

        # Calculate mean y-value of each baseline
        baseline_y_means = []
        for line in merged_segmentation.lines:
            y_values = [point[1] for point in line.baseline]
            mean_y = np.mean(y_values)
            baseline_y_means.append(mean_y)

        # Group lines by similar y-value
        groups = defaultdict(list)
        processed = set()

        for i, y_mean in enumerate(baseline_y_means):
            if i in processed:
                continue

            current_group = [i]
            processed.add(i)

            for j, other_y_mean in enumerate(baseline_y_means):
                if j not in processed and abs(y_mean - other_y_mean) <= y_threshold:
                    current_group.append(j)
                    processed.add(j)

            group_id = len(groups)
            groups[group_id] = current_group

        # Create new merged lines
        new_lines = []

        for group_id, line_indices in groups.items():
            if len(line_indices) == 1:
                # No merging needed, just keep the original line
                new_lines.append(merged_segmentation.lines[line_indices[0]])
            else:
                # Merge the lines
                lines_to_merge = [merged_segmentation.lines[i] for i in line_indices]

                # Sort by x-coordinate of first baseline point to handle left-to-right order
                lines_to_merge.sort(key=lambda line: line.baseline[0][0])

                # Merge baseline points, ensuring x-coordinates are in ascending order
                merged_baseline = []
                for line in lines_to_merge:
                    # Only add baseline points that extend beyond our current baseline
                    if (
                        not merged_baseline
                        or line.baseline[0][0] > merged_baseline[-1][0]
                    ):
                        merged_baseline.extend(line.baseline)
                    else:
                        # Find where to insert the new baseline points
                        insert_idx = next(
                            (
                                i
                                for i, pt in enumerate(merged_baseline)
                                if pt[0] > line.baseline[0][0]
                            ),
                            len(merged_baseline),
                        )
                        merged_baseline[insert_idx:insert_idx] = line.baseline

                # Sort baseline points by x-coordinate for consistency
                merged_baseline.sort(key=lambda pt: pt[0])

                # Merge boundaries
                all_boundary_points = []
                for line in lines_to_merge:
                    all_boundary_points.extend(line.boundary)

                # Create a convex hull from all boundary points to get a clean merged boundary
                try:
                    from scipy.spatial import ConvexHull

                    points = np.array(all_boundary_points)
                    hull = ConvexHull(points)
                    merged_boundary = [tuple(points[i]) for i in hull.vertices]

                    # Ensure the boundary points are in clockwise/counter-clockwise order
                    centroid = np.mean(merged_boundary, axis=0)
                    merged_boundary.sort(
                        key=lambda pt: np.arctan2(
                            pt[1] - centroid[1], pt[0] - centroid[0]
                        )
                    )
                    merged_boundary = [tuple(map(int, pt)) for pt in merged_boundary]

                except Exception:
                    # If convex hull fails, just use all boundary points (less clean but functional)
                    merged_boundary = list(
                        set(all_boundary_points)
                    )  # Remove duplicates

                # Create a new line with merged properties
                merged_line = copy.deepcopy(lines_to_merge[0])
                merged_line.baseline = merged_baseline
                merged_line.boundary = merged_boundary

                # Update the ID
                if hasattr(merged_line, "id"):
                    merged_line.id = (
                        f"merged_{'_'.join(str(line.id) for line in lines_to_merge)}"
                    )

                new_lines.append(merged_line)

        # Update the segmentation object with new lines
        new_segmentation.lines = new_lines

        # Create visualizations and crop line images
        img_cv = np.array(im)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        vis_image = img_cv.copy()

        line_images = []
        for line in new_segmentation.lines:
            # Draw baseline in red
            baseline_points = np.array(line.baseline, dtype=np.int32)
            for i in range(len(baseline_points) - 1):
                pt1 = tuple(baseline_points[i])
                pt2 = tuple(baseline_points[i + 1])
                cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)

            # Draw boundary polygon in green
            boundary_points = np.array(line.boundary, dtype=np.int32)
            cv2.polylines(vis_image, [boundary_points], True, (0, 255, 0), 2)

            # Get bounding box from boundary points
            x1 = min(p[0] for p in line.boundary)
            y1 = min(p[1] for p in line.boundary)
            x2 = max(p[0] for p in line.boundary)
            y2 = max(p[1] for p in line.boundary)

            # Add padding
            x1 = max(0, x1 - padding)
            x2 = min(im.width, x2 + padding)

            # Crop the line from the original image
            line_img = im.crop((x1, y1, x2, y2))
            line_images.append(line_img)

        return new_segmentation, line_images

    def segment_image(
        self,
        image_path: str,
        output_dir: Path,
        is_margin: bool = False,
        padding: int = 15,
        iou_threshold: float = 0.5,
        y_threshold: int = 20,
    ) -> List[LineData]:
        """
        Segment an image into lines and save them

        Args:
            image_path: Path to the input image
            output_dir: Directory to save segmented lines
            is_margin: If True, process as margin text, else as main text
            padding: Padding for line extraction (width for margins, height for main text)
            iou_threshold: IoU threshold for filtering overlapping lines (main text only)
            y_threshold: Vertical threshold for merging lines

        Returns:
            List of LineData objects containing information about each line
        """
        im = Image.open(image_path)

        # Get initial segmentation
        segmentation = blla.segment(im, model=self.model, device=self.device)

        # Apply new line merging algorithm
        merged_segmentation, line_images = self.merge_similar_lines(
            image_path, segmentation, y_threshold=y_threshold, padding=padding
        )

        line_data_list = []

        for idx, (line, line_img) in enumerate(
            zip(merged_segmentation.lines, line_images)
        ):
            # Get crop coordinates
            x1 = min(p[0] for p in line.boundary) - padding
            y1 = min(p[1] for p in line.boundary)
            x2 = max(p[0] for p in line.boundary) + padding
            y2 = max(p[1] for p in line.boundary)

            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            x2 = min(im.width, x2)

            # Apply IoU filtering for main text if specified
            if not is_margin and idx > 0:
                current_box = (y1, y2)
                # Check for overlaps with previous lines
                skip = False
                for prev_idx in range(idx):
                    prev_line = merged_segmentation.lines[prev_idx]
                    prev_y1 = min(p[1] for p in prev_line.boundary)
                    prev_y2 = max(p[1] for p in prev_line.boundary)
                    prev_box = (prev_y1, prev_y2)

                    # Calculate IoU for vertical overlap
                    y_overlap = max(
                        0,
                        min(current_box[1], prev_box[1])
                        - max(current_box[0], prev_box[0]),
                    )
                    y_union = (
                        (current_box[1] - current_box[0])
                        + (prev_box[1] - prev_box[0])
                        - y_overlap
                    )
                    iou = y_overlap / y_union if y_union > 0 else 0

                    if iou > iou_threshold:
                        skip = True
                        break

                if skip:
                    continue

            # Save line image
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
                image_path=str(
                    line_path.relative_to(output_dir.parent)
                    if output_dir.parent in line_path.parents
                    else str(line_path)
                ),
            )
            line_data_list.append(line_data)

        return line_data_list
