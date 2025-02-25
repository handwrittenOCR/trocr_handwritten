from pathlib import Path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from settings import LayoutParserSettings
import logging
import cv2
import json


class YOLOv10Model:
    """
    YOLOv10 model class for layout parsing of historical documents
    """

    def __init__(self, settings: LayoutParserSettings, logger: logging.Logger):
        self.model = None
        self.device = settings.device
        self.conf = settings.conf
        self.iou = settings.iou
        if settings.path_model:
            self.load_model_from_local(settings.path_model, logger)
        elif settings.hf_repo and settings.hf_filename:
            self.load_model_from_hf(settings.hf_repo, settings.hf_filename, logger)
        else:
            raise ValueError(
                "Either path_model or hf_repo and hf_filename must be provided"
            )

    def load_model_from_hf(self, hf_repo, hf_filename, logger):
        """
        Load a YOLOv10 model from a HuggingFace Hub repository

        Args:
            hf_repo: Repository name on HuggingFace Hub
            hf_filename: Filename of the model on HuggingFace Hub
            logger: Logger
        Returns:
            YOLOv10: YOLOv10 model
        """
        try:
            logger.info("Trying to load model from HuggingFace Hub...")
            filepath = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
            logger.info(f"Model loaded from HuggingFace Hub: {filepath}")
            self.model = YOLOv10(filepath)
        except Exception as e:
            logger.error(f"Error loading model from HuggingFace Hub: {e}")
            raise ValueError(f"Error loading model from HuggingFace Hub: {e}")

    def load_model_from_local(self, path_model, logger):
        """
        Load a YOLOv10 model from a local file

        Args:
            path_model: Path to the model file
            logger: Logger
        Returns:
            YOLOv10: YOLOv10 model
        """
        try:
            logger.info("Loading model from local file...")
            self.model = YOLOv10(path_model)
        except Exception as e:
            logger.error(f"Error loading model from local file: {e}")
            raise ValueError(f"Error loading model from local file: {e}")

    def predict(self, path_folder):
        return self.model.predict(
            path_folder, imgsz=1024, conf=self.conf, iou=self.iou, device=self.device
        )


def calculate_iou(box1, box2):
    """
    Calculate the IoU between two boxes.

    Args:
        box1: Box 1 (x, y, w, h)
        box2: Box 2 (x, y, w, h)
    Returns:
        float: IoU between the two boxes
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the corners
    x1_1, y1_1, x1_2, y1_2 = x1, y1, x1 + w1, y1 + h1
    x2_1, y2_1, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

    # Calculate the intersection
    xi1, yi1 = max(x1_1, x2_1), max(y1_1, y2_1)
    xi2, yi2 = min(x1_2, x2_2), min(y1_2, y2_2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate the union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def create_via_json(detection_results, class_names, iou=0.5):
    """
    Create a JSON file in VIA format from the YOLO detection results

    Args:
        detection_results: List of YOLO detection results
        class_names: Dictionary of mapping classes
        iou: Intersection over union threshold for predictions (default: 0.5)
    Returns:
        dict: JSON in VIA format
    """
    via_json = {
        "_via_settings": {
            "ui": {
                "annotation_editor_height": 25,
                "annotation_editor_fontsize": 0.8,
                "leftsidebar_width": 18,
                "image_grid": {
                    "img_height": 80,
                    "rshape_fill": "none",
                    "rshape_fill_opacity": 0.3,
                    "rshape_stroke": "yellow",
                    "rshape_stroke_width": 2,
                    "show_region_shape": True,
                    "show_image_policy": "all",
                },
                "image": {
                    "region_label": "Classes",
                    "region_color": "__via_default_region_color__",
                    "region_label_font": "10px Sans",
                    "on_image_annotation_editor_placement": "NEAR_REGION",
                },
            },
            "core": {"buffer_size": 18, "filepath": {}, "default_filepath": ""},
            "project": {"name": "yolo_predictions"},
        },
        "_via_attributes": {
            "region": {
                "Classes": {
                    "type": "dropdown",
                    "description": "",
                    "options": class_names,
                    "default_options": {},
                }
            },
            "file": {},
        },
        "_via_data_format_version": "2.0.10",
        "_via_img_metadata": {},
        "_via_image_id_list": [],
    }

    for res in detection_results:
        filename = Path(res.path).name
        filesize = Path(res.path).stat().st_size
        image_id = f"{filename}{filesize}"

        regions = []
        boxes = res.__dict__["boxes"].xywh
        classes = res.__dict__["boxes"].cls

        for box, cls in zip(boxes, classes):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())

            new_box = [int(x - w / 2), int(y - h / 2), int(w), int(h)]
            is_duplicate = False

            for existing_region in regions:
                existing_box = [
                    existing_region["shape_attributes"]["x"],
                    existing_region["shape_attributes"]["y"],
                    existing_region["shape_attributes"]["width"],
                    existing_region["shape_attributes"]["height"],
                ]
                if calculate_iou(new_box, existing_box) > iou:
                    is_duplicate = True
                    break

            if not is_duplicate:
                region = {
                    "shape_attributes": {
                        "name": "rect",
                        "x": new_box[0],
                        "y": new_box[1],
                        "width": new_box[2],
                        "height": new_box[3],
                    },
                    "region_attributes": {"Classes": str(class_id)},
                }
                regions.append(region)

        image_metadata = {
            "filename": filename,
            "size": filesize,
            "regions": regions,
            "file_attributes": {},
        }

        via_json["_via_img_metadata"][image_id] = image_metadata
        via_json["_via_image_id_list"].append(image_id)

    return via_json


def create_structured_crops(detection_results, class_names, path_output, iou=0.5):
    """
    Create a structured folder system with cropped images and metadata from YOLO detection results

    Args:
        detection_results: List of YOLO detection results
        class_names: Dictionary of mapping classes
        path_output: Path where to create the folder structure
        iou: Intersection over union threshold for predictions (default: 0.5)
    """

    # Create the base output directory if it doesn't exist
    base_output = Path(path_output)
    base_output.parent.mkdir(parents=True, exist_ok=True)
    base_output.mkdir(exist_ok=True)

    for res in detection_results:
        # Read the image
        img = cv2.imread(str(res.path))

        # Create folder for this image
        image_name = Path(res.path).stem
        image_folder = base_output / image_name
        image_folder.mkdir(exist_ok=True)

        # Initialize metadata dictionary
        metadata = {}

        boxes = res.__dict__["boxes"].xywh
        classes = res.__dict__["boxes"].cls

        # Track processed boxes to avoid duplicates
        processed_boxes = []

        for idx, (box, cls) in enumerate(zip(boxes, classes)):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())
            class_name = class_names[str(class_id)]

            # Convert to integer coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            w = int(w)
            h = int(h)

            # Check for duplicates using IOU
            new_box = [x1, y1, w, h]
            is_duplicate = False

            for existing_box in processed_boxes:
                if calculate_iou(new_box, existing_box) > iou:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            processed_boxes.append(new_box)

            # Create class folder if it doesn't exist
            class_folder = image_folder / class_name
            class_folder.mkdir(exist_ok=True)

            # Crop the image
            cropped = img[y1 : y1 + h, x1 : x1 + w]

            # Generate crop filename
            crop_filename = f"{idx:03d}.jpg"
            crop_path = class_folder / crop_filename

            # Save cropped image
            cv2.imwrite(str(crop_path), cropped)

            # Add to metadata
            if class_name not in metadata:
                metadata[class_name] = []

            metadata[class_name].append(
                {
                    "cropped_image_name": crop_filename,
                    "coordinates": {"x": x1, "y": y1, "width": w, "height": h},
                }
            )

        # Save metadata JSON
        metadata_path = image_folder / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
