import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import torch
import logging
import cv2
import json

from trocr_handwritten.parse.settings import LayoutParserSettings

load_dotenv()


def _get_hf_token():
    """
    Resolve the HuggingFace token from environment variables.
    Checks HF_TOKEN, then HUGGINGFACE_API_KEY.
    """
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")


def _detect_model_backend(model_path):
    """
    Detect whether a model checkpoint was trained with doclayout-yolo or ultralytics.

    Args:
        model_path: Path to the .pt model file

    Returns:
        str: "doclayout" or "ultralytics"
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        module = getattr(ckpt["model"].__class__, "__module__", "")
        if module.startswith("doclayout_yolo"):
            return "doclayout"
    return "ultralytics"


def _load_model(model_path, backend=None):
    """
    Load a YOLO model with the appropriate backend.

    Args:
        model_path: Path to the .pt model file
        backend: Force a backend ("doclayout" or "ultralytics"). Auto-detects if None.

    Returns:
        YOLO model instance
    """
    if backend is None:
        backend = _detect_model_backend(model_path)

    if backend == "doclayout":
        from doclayout_yolo import YOLOv10

        return YOLOv10(model_path)

    from ultralytics import YOLO

    return YOLO(model_path)


class YOLOModel:
    """
    YOLO model class for layout parsing of historical documents.
    Supports both doclayout-yolo (YOLOv10) and ultralytics (YOLO11) backends.
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
        Load a YOLO model from a HuggingFace Hub repository.

        Args:
            hf_repo: Repository name on HuggingFace Hub
            hf_filename: Filename of the model on HuggingFace Hub
            logger: Logger
        """
        try:
            logger.info("Trying to load model from HuggingFace Hub...")
            token = _get_hf_token()
            filepath = hf_hub_download(
                repo_id=hf_repo, filename=hf_filename, token=token
            )
            logger.info(f"Model downloaded from HuggingFace Hub: {filepath}")
            self.model = _load_model(filepath)
            logger.info(f"Model loaded with backend: {_detect_model_backend(filepath)}")
        except Exception as e:
            logger.error(f"Error loading model from HuggingFace Hub: {e}")
            raise ValueError(f"Error loading model from HuggingFace Hub: {e}")

    def load_model_from_local(self, path_model, logger):
        """
        Load a YOLO model from a local file.

        Args:
            path_model: Path to the model file
            logger: Logger
        """
        try:
            logger.info("Loading model from local file...")
            self.model = _load_model(path_model)
            logger.info(
                f"Model loaded with backend: {_detect_model_backend(path_model)}"
            )
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

    x1_1, y1_1, x1_2, y1_2 = x1, y1, x1 + w1, y1 + h1
    x2_1, y2_1, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

    xi1, yi1 = max(x1_1, x2_1), max(y1_1, y2_1)
    xi2, yi2 = min(x1_2, x2_2), min(y1_2, y2_2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def create_predictions_json(detection_results, class_names, iou=0.5):
    """
    Create a JSON with detection results per image.

    Args:
        detection_results: List of YOLO detection results
        class_names: Dictionary of mapping classes
        iou: Intersection over union threshold for dedup (default: 0.5)

    Returns:
        list: List of per-image detection dicts
    """
    output = []

    for res in detection_results:
        filename = Path(res.path).name
        boxes = res.__dict__["boxes"].xywh
        classes = res.__dict__["boxes"].cls
        confs = res.__dict__["boxes"].conf

        regions = []
        processed_boxes = []

        for box, cls, conf in zip(boxes, classes, confs):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())
            class_name = class_names.get(str(class_id))
            if class_name is None:
                continue

            new_box = [int(x - w / 2), int(y - h / 2), int(w), int(h)]
            is_duplicate = False
            for existing_box in processed_boxes:
                if calculate_iou(new_box, existing_box) > iou:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            processed_boxes.append(new_box)
            regions.append(
                {
                    "class_id": class_id,
                    "label": class_name,
                    "confidence": round(float(conf.item()), 4),
                    "x": new_box[0],
                    "y": new_box[1],
                    "width": new_box[2],
                    "height": new_box[3],
                }
            )

        output.append(
            {
                "filename": filename,
                "n_regions": len(regions),
                "regions": regions,
            }
        )

    return output


def create_structured_crops(detection_results, class_names, path_output, iou=0.5):
    """
    Create a structured folder system with cropped images and metadata from YOLO detection results

    Args:
        detection_results: List of YOLO detection results
        class_names: Dictionary of mapping classes
        path_output: Path where to create the folder structure
        iou: Intersection over union threshold for predictions (default: 0.5)
    """

    base_output = Path(path_output)
    base_output.parent.mkdir(parents=True, exist_ok=True)
    base_output.mkdir(exist_ok=True)

    for res in detection_results:
        img = cv2.imread(str(res.path))

        image_name = Path(res.path).stem
        image_folder = base_output / image_name
        image_folder.mkdir(exist_ok=True)

        metadata = {}

        boxes = res.__dict__["boxes"].xywh
        classes = res.__dict__["boxes"].cls

        processed_boxes = []

        for idx, (box, cls) in enumerate(zip(boxes, classes)):
            x, y, w, h = box.tolist()
            class_id = int(cls.item())
            class_name = class_names.get(str(class_id))
            if class_name is None:
                continue

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            w = int(w)
            h = int(h)

            new_box = [x1, y1, w, h]
            is_duplicate = False

            for existing_box in processed_boxes:
                if calculate_iou(new_box, existing_box) > iou:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            processed_boxes.append(new_box)

            class_folder = image_folder / class_name
            class_folder.mkdir(exist_ok=True)

            cropped = img[y1 : y1 + h, x1 : x1 + w]

            crop_filename = f"{idx:03d}.jpg"
            crop_path = class_folder / crop_filename

            cv2.imwrite(str(crop_path), cropped)

            if class_name not in metadata:
                metadata[class_name] = []

            metadata[class_name].append(
                {
                    "cropped_image_name": crop_filename,
                    "coordinates": {"x": x1, "y": y1, "width": w, "height": h},
                }
            )

        metadata_path = image_folder / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
