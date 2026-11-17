#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from trocr_handwritten.hw_filter.features import Embedder
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def main(path_folder, model_path, output):
    """
    Classify the crops written by the layout parser as handwritten or not.

    path_folder is the output of parse.layout_parser: one folder per page, one
    subfolder per class, one JPEG per region. Only the classes the filter was
    trained on are classified.

    Args:
        path_folder: Layout parser output folder.
        model_path: hw_filter.joblib written by hw_filter.train.
        output: CSV path (page, class, crop, p_manuscrit, handwriting).
    """
    bundle = joblib.load(model_path)
    classes = bundle["classes"]
    crops = sorted(p for c in classes for p in Path(path_folder).glob(f"*/{c}/*.jpg"))
    logger.info(f"{len(crops)} crops des classes {classes} dans {path_folder}")
    embedder = Embedder(bundle["embed_model"], bundle["image_size"])
    emb = embedder.embed(
        (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in crops), total=len(crops)
    )
    onehot = np.array([[int(p.parent.name == c) for c in classes] for p in crops])
    X = pd.DataFrame(
        np.hstack([emb, onehot]), columns=bundle["model"].feature_names_in_
    )
    proba = bundle["model"].predict_proba(X)[:, 1]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "class", "crop", "p_manuscrit", "handwriting"])
        for p, q in zip(crops, proba):
            w.writerow(
                [
                    p.parent.parent.name,
                    p.parent.name,
                    p.name,
                    round(float(q), 4),
                    "manuscrit" if q >= bundle["threshold"] else "non_manuscrit",
                ]
            )
    logger.info(
        f"{int((proba >= bundle['threshold']).sum())}/{len(crops)} manuscrits -> {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply the handwriting filter to layout parser crops"
    )
    parser.add_argument(
        "path_folder", help="Output folder of trocr_handwritten.parse.layout_parser"
    )
    parser.add_argument("--model", default="data/processed/hw_filter/hw_filter.joblib")
    parser.add_argument(
        "--output", default="data/processed/hw_filter/predictions_crops.csv"
    )
    args = parser.parse_args()
    main(args.path_folder, args.model, args.output)
