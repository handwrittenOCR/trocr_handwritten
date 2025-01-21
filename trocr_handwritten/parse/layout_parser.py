#!/usr/bin/env python3
import json
from os.path import join
from utils import YOLOv10Model, create_via_json, create_structured_crops
from settings import LayoutParserSettings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(settings: LayoutParserSettings, logger):

    model = YOLOv10Model(settings, logger)

    logger.info(f"Predicting on {settings.path_folder}...")
    det_res = model.predict(settings.path_folder)

    # Create JSON
    logger.info("Creating JSON file...")

    create_structured_crops(det_res, settings.class_names, settings.path_output)

    # Save JSON for annotation
    if settings.create_annotation_json:
        via_json = create_via_json(det_res, settings.class_names, settings.iou)
        output_path = join(settings.path_output, "yolo_predictions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(via_json, f, ensure_ascii=False, indent=2)

        logger.info(f"Json file for annotation created: {output_path}")


if __name__ == "__main__":

    settings = LayoutParserSettings(logger=logger)
    main(settings, logger)
