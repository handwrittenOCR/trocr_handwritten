import argparse
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from trocr_handwritten.parse.settings import (
    EvaluationSettings,
)
from trocr_handwritten.parse.utils import _load_model, _get_hf_token
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def _parse_yolo_label(label_path):
    """
    Parse a YOLO label file into xyxy boxes and class ids.

    Args:
        label_path: Path to the .txt label file

    Returns:
        tuple: (boxes_xyxy as np.ndarray, classes as np.ndarray)
    """
    boxes = []
    classes = []
    if not label_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls = int(parts[0])
        cx, cy, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        classes.append(cls)
    if boxes:
        return np.array(boxes), np.array(classes, dtype=int)
    return np.zeros((0, 4)), np.zeros(0, dtype=int)


def _compute_iou_matrix(boxes_a, boxes_b):
    """
    Compute IoU matrix between two sets of xyxy boxes.

    Args:
        boxes_a: Nx4 array of xyxy boxes
        boxes_b: Mx4 array of xyxy boxes

    Returns:
        np.ndarray: NxM IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = np.array(boxes_a)
    b = np.array(boxes_b)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def _match_predictions(
    gt_boxes,
    gt_classes,
    pred_boxes,
    pred_classes,
    pred_confs,
    num_classes,
    iou_threshold=0.5,
):
    """
    Match predictions to ground truth and compute per-class TP/FP/FN.

    Args:
        gt_boxes: Ground truth boxes (xyxy)
        gt_classes: Ground truth class ids
        pred_boxes: Predicted boxes (xyxy)
        pred_classes: Predicted class ids
        pred_confs: Prediction confidences
        num_classes: Total number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        dict: {cls_id: {"tp": int, "fp": int, "fn": int}}
    """
    results = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(num_classes)}

    for cls_id in range(num_classes):
        gt_mask = gt_classes == cls_id
        pred_mask = pred_classes == cls_id

        gt_cls = gt_boxes[gt_mask]
        pred_cls = pred_boxes[pred_mask]

        if len(gt_cls) == 0:
            results[cls_id]["fp"] += len(pred_cls)
            continue
        if len(pred_cls) == 0:
            results[cls_id]["fn"] += len(gt_cls)
            continue

        iou_matrix = _compute_iou_matrix(pred_cls, gt_cls)
        gt_matched = set()

        pred_conf_cls = (
            pred_confs[pred_mask] if len(pred_confs) > 0 else np.ones(len(pred_cls))
        )
        order = np.argsort(-pred_conf_cls)

        for pred_idx in order:
            best_gt = np.argmax(iou_matrix[pred_idx])
            if (
                iou_matrix[pred_idx, best_gt] >= iou_threshold
                and best_gt not in gt_matched
            ):
                results[cls_id]["tp"] += 1
                gt_matched.add(best_gt)
            else:
                results[cls_id]["fp"] += 1

        results[cls_id]["fn"] += len(gt_cls) - len(gt_matched)

    return results


def evaluate(settings, logger):
    """
    Evaluate a YOLO model on a dataset split with P/R/F1 per class.

    Args:
        settings: EvaluationSettings instance
        logger: Logger instance

    Returns:
        dict: Per-class TP/FP/FN totals
    """
    if settings.path_model:
        model = _load_model(settings.path_model)
    else:
        token = _get_hf_token()
        filepath = hf_hub_download(
            repo_id=settings.hf_repo, filename=settings.hf_filename, token=token
        )
        model = _load_model(filepath)

    class_names_list = [
        settings.class_names[str(i)] for i in range(len(settings.class_names))
    ]
    num_classes = len(class_names_list)

    split_dir = Path(settings.path_data) / settings.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        logger.error(f"Images dir not found: {images_dir}")
        return None

    image_paths = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )
    totals = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(num_classes)}

    import cv2

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes_norm, gt_classes = _parse_yolo_label(label_path)

        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        gt_boxes = gt_boxes_norm.copy()
        if len(gt_boxes) > 0:
            gt_boxes[:, [0, 2]] *= w_img
            gt_boxes[:, [1, 3]] *= h_img

        results = model.predict(
            str(img_path),
            imgsz=settings.imgsz,
            conf=settings.conf,
            iou=settings.iou,
            device=settings.device,
            verbose=False,
        )

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
            pred_confs = results[0].boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))
            pred_classes = np.zeros(0, dtype=int)
            pred_confs = np.zeros(0)

        img_results = _match_predictions(
            gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confs, num_classes
        )
        for cls_id in range(num_classes):
            for k in ("tp", "fp", "fn"):
                totals[cls_id][k] += img_results[cls_id][k]

    logger.info(
        f"\nSplit: {settings.split} | conf={settings.conf} | iou={settings.iou}"
    )
    logger.info(f"{'':>12}  {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    logger.info("-" * 60)

    sum_tp = sum_fp = sum_fn = 0
    for cls_id in range(num_classes):
        tp = totals[cls_id]["tp"]
        fp = totals[cls_id]["fp"]
        fn = totals[cls_id]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        logger.info(
            f"{class_names_list[cls_id]:>12}  {p:>7.4f} {r:>7.4f} {f1:>7.4f} {tp:>5} {fp:>5} {fn:>5}"
        )
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    p_all = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    r_all = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0
    logger.info("-" * 60)
    logger.info(
        f"{'ALL':>12}  {p_all:>7.4f} {r_all:>7.4f} {f1_all:>7.4f} {sum_tp:>5} {sum_fp:>5} {sum_fn:>5}"
    )

    return totals


def main():
    """CLI entry point for layout model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO layout model")
    parser.add_argument("--path-data", type=str, default="data/layout")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    settings = EvaluationSettings(
        path_data=args.path_data,
        path_model=args.model,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    evaluate(settings, logger)


if __name__ == "__main__":
    main()
