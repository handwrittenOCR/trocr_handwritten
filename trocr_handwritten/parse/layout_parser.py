#!/usr/bin/env python3
import argparse
import json
from os.path import join

from trocr_handwritten.parse.utils import (
    YOLOModel,
    create_predictions_json,
    create_structured_crops,
)
from trocr_handwritten.parse.settings import LayoutParserSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def main(settings: LayoutParserSettings, logger):

    model = YOLOModel(settings, logger)

    logger.info(f"Predicting on {settings.path_folder}...")
    det_res = model.predict(settings.path_folder)

    logger.info("Creating structured crops...")
    create_structured_crops(det_res, settings.class_names, settings.path_output)

    if settings.create_annotation_json:
        predictions = create_predictions_json(
            det_res, settings.class_names, settings.iou
        )
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
        "--no-annotation-json", action="store_true", help="Skip VIA JSON generation"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific class names (e.g. --classes Marge 'Plein Texte')",
    )
    args = parser.parse_args()

    from trocr_handwritten.parse.settings import CLASS_NAMES

    class_names = dict(CLASS_NAMES)
    if args.classes:
        class_names = {k: v for k, v in class_names.items() if v in args.classes}

    settings = LayoutParserSettings(
        path_folder=args.path_folder,
        path_output=args.output,
        path_model=args.model,
        hf_repo=args.hf_repo if args.model is None else None,
        hf_filename=args.hf_filename if args.model is None else None,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        create_annotation_json=not args.no_annotation_json,
        class_names=class_names,
    )

    main(settings, logger)
