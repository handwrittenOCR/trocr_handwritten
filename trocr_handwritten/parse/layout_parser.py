#!/usr/bin/env python3
import argparse
import json
from os.path import join
from pathlib import Path

from trocr_handwritten.parse.utils import (
    YOLOModel,
    create_predictions_json,
    create_structured_crops,
)
from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main(settings: LayoutParserSettings, logger):
    """
    Detect regions on every image of path_folder, batch by batch.

    Predicting a whole folder at once keeps every page in memory, so images are
    processed in batches of settings.batch_size; crops are written per batch and
    predictions.json gathers all batches.
    """
    model = YOLOModel(settings, logger)

    images = sorted(
        str(p)
        for p in Path(settings.path_folder).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    logger.info(f"Predicting on {len(images)} images in {settings.path_folder}...")

    predictions = []
    for start in range(0, len(images), settings.batch_size):
        batch = images[start : start + settings.batch_size]
        det_res = model.predict(batch)
        create_structured_crops(det_res, settings.class_names, settings.path_output)
        if settings.create_annotation_json:
            predictions.extend(
                create_predictions_json(det_res, settings.class_names, settings.iou)
            )
        logger.info(f"Processed {start + len(batch)}/{len(images)} images")

    if settings.create_annotation_json:
        output_path = join(settings.path_output, "predictions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        logger.info(f"Predictions JSON created: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply YOLO layout parser on images")
    parser.add_argument("path_folder", nargs="?", default="data/raw/images")
    parser.add_argument("--output", type=str, default="data/processed/images/")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default="agomberto/historical-layout-ft")
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="20241119_v2_yolov10_50_finetuned.pt",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images predicted per batch (bounds memory on large folders)",
    )
    parser.add_argument(
        "--no-annotation-json", action="store_true", help="Skip VIA JSON generation"
    )
    args = parser.parse_args()

    settings = LayoutParserSettings(
        path_folder=args.path_folder,
        path_output=args.output,
        path_model=args.model,
        hf_repo=args.hf_repo if args.model is None else None,
        hf_filename=args.hf_filename if args.model is None else None,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
        create_annotation_json=not args.no_annotation_json,
    )

    main(settings, logger)
