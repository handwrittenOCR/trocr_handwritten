import argparse
import json
import time
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


def _match_tp_flags(gt_boxes, pred_boxes, pred_confs, iou_thresholds):
    """
    COCO-style greedy matching of one class on one image at several IoU thresholds.

    Predictions are visited by decreasing confidence; each one takes the unmatched
    ground-truth box with the highest IoU, if that IoU reaches the threshold.

    Args:
        gt_boxes: Ground truth boxes (xyxy) of one class
        pred_boxes: Predicted boxes (xyxy) of the same class
        pred_confs: Prediction confidences
        iou_thresholds: Sequence of IoU thresholds

    Returns:
        np.ndarray: (n_pred, n_thresholds) boolean TP flags, in the input order
    """
    flags = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return flags
    iou_matrix = _compute_iou_matrix(pred_boxes, gt_boxes)
    for t_idx, thr in enumerate(iou_thresholds):
        flags[:, t_idx] = _greedy_match(iou_matrix, pred_confs, thr) >= 0
    return flags


def _greedy_match(iou_matrix, pred_confs, iou_threshold):
    """
    Greedy one-to-one matching of predictions to ground truth by decreasing confidence.

    Each prediction takes the unmatched ground-truth box with the highest IoU, if
    that IoU reaches the threshold.

    Args:
        iou_matrix: (n_pred, n_gt) IoU matrix
        pred_confs: Prediction confidences
        iou_threshold: IoU threshold

    Returns:
        np.ndarray: Matched ground-truth index per prediction (-1 if unmatched),
            in the input order
    """
    match = np.full(iou_matrix.shape[0], -1, dtype=int)
    matched = np.zeros(iou_matrix.shape[1], dtype=bool)
    if iou_matrix.shape[1] == 0:
        return match
    for pred_idx in np.argsort(-pred_confs, kind="stable"):
        ious = np.where(matched, -1.0, iou_matrix[pred_idx])
        best_gt = int(np.argmax(ious))
        if ious[best_gt] >= iou_threshold:
            matched[best_gt] = True
            match[pred_idx] = best_gt
    return match


def _confusion_matrix(
    gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confs, num_classes, iou
):
    """
    Confusion matrix of one image from a class-agnostic greedy matching.

    Rows are ground-truth classes plus a last "background" row (predictions that
    match no ground-truth box); columns are predicted classes plus a last "missed"
    column (ground-truth boxes matched by no prediction).

    Args:
        gt_boxes: Ground truth boxes (xyxy)
        gt_classes: Ground truth class ids
        pred_boxes: Predicted boxes (xyxy)
        pred_classes: Predicted class ids
        pred_confs: Prediction confidences
        num_classes: Total number of classes
        iou: IoU threshold for a match

    Returns:
        np.ndarray: (num_classes + 1, num_classes + 1) integer counts
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    match = _greedy_match(_compute_iou_matrix(pred_boxes, gt_boxes), pred_confs, iou)
    for pred_idx, gt_idx in enumerate(match):
        row = gt_classes[gt_idx] if gt_idx >= 0 else num_classes
        matrix[row, pred_classes[pred_idx]] += 1
    missed = np.setdiff1d(np.arange(len(gt_boxes)), match[match >= 0])
    for gt_idx in missed:
        matrix[gt_classes[gt_idx], num_classes] += 1
    return matrix


def _average_precision(tp_flags, confs, n_gt, recall_points):
    """
    COCO interpolated average precision for one class and one IoU threshold.

    Args:
        tp_flags: Boolean TP flag per prediction, pooled over all images
        confs: Confidence per prediction
        n_gt: Number of ground-truth boxes of the class
        recall_points: Number of recall points of the interpolation (COCO: 101)

    Returns:
        float: AP, or nan if the class has no ground truth
    """
    if n_gt == 0:
        return float("nan")
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-confs, kind="stable")
    tp = np.cumsum(tp_flags[order])
    fp = np.cumsum(~tp_flags[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, recall_points), side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def _prf(tp, fp, fn):
    """
    Precision, recall and F1 from counts.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        tuple: (precision, recall, f1)
    """
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def _prf_table(counts, class_names_list, n_gt):
    """
    Build the P/R/F1 table of one IoU threshold from per-class TP/FP/FN counts.

    Args:
        counts: {cls_id: {"tp": int, "fp": int, "fn": int}}
        class_names_list: Ordered class names
        n_gt: Ground-truth count per class (macro average over classes with ground truth)

    Returns:
        dict: {class name: {p, r, f1, tp, fp, fn}, "ALL": micro, "MACRO": macro}
    """
    table = {}
    sums = {"tp": 0, "fp": 0, "fn": 0}
    for cls_id, name in enumerate(class_names_list):
        c = counts[cls_id]
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        table[name] = {"p": p, "r": r, "f1": f1, **c}
        for k in sums:
            sums[k] += c[k]
    p, r, f1 = _prf(sums["tp"], sums["fp"], sums["fn"])
    table["ALL"] = {"p": p, "r": r, "f1": f1, **sums}
    with_gt = [class_names_list[i] for i in range(len(class_names_list)) if n_gt[i] > 0]
    table["MACRO"] = {
        k: float(np.mean([table[c][k] for c in with_gt])) if with_gt else 0.0
        for k in ("p", "r", "f1")
    }
    return table


def _predict(model, img_path, settings):
    """
    Run one prediction and time it (preprocessing, inference, NMS).
    Models exposing detect(img_path, conf, iou, device) (torchvision detectors) are
    called through it; others go through the ultralytics predict API.

    Args:
        model: Loaded ultralytics / doclayout-yolo model or FasterRCNNDetector
        img_path: Path to the page image
        settings: EvaluationSettings instance

    Returns:
        tuple: (boxes_xyxy, classes, confs, seconds)
    """
    start = time.perf_counter()
    if hasattr(model, "detect"):
        out = model.detect(img_path, settings.ap_conf, settings.iou, settings.device)
        return (*out, time.perf_counter() - start)
    results = model.predict(
        str(img_path),
        imgsz=settings.imgsz,
        conf=settings.ap_conf,
        iou=settings.iou,
        device=settings.device,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        out = (
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        )
    else:
        out = (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
    return (*out, time.perf_counter() - start)


def _log_report(report, logger):
    """
    Log the evaluation report: AP table, P/R/F1 tables per IoU threshold, timing.

    Args:
        report: Dict returned by evaluate
        logger: Logger instance
    """
    logger.info(
        f"\nSplit: {report['split']} | pages={report['n_pages']} | "
        f"conf={report['conf']} | nms_iou={report['nms_iou']} | "
        f"device={report['device']}"
    )
    logger.info(f"{'':>12}  {'n_gt':>5} {'AP50':>7} {'AP50:95':>7}")
    logger.info("-" * 36)
    for name, c in report["per_class"].items():
        logger.info(
            f"{name:>12}  {c['n_gt']:>5} {c['ap50']:>7.4f} {c['ap50_95']:>7.4f}"
        )
    logger.info("-" * 36)
    logger.info(
        f"{'mAP':>12}  {report['n_gt']:>5} {report['map50']:>7.4f} "
        f"{report['map50_95']:>7.4f}"
    )
    for key in report["prf"]:
        logger.info(f"\nIoU {key} (conf >= {report['conf']})")
        logger.info(
            f"{'':>12}  {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
        )
        logger.info("-" * 60)
        for name, c in report["prf"][key].items():
            if name == "ALL":
                logger.info("-" * 60)
            counts = f"{c['tp']:>5} {c['fp']:>5} {c['fn']:>5}" if "tp" in c else ""
            logger.info(
                f"{name:>12}  {c['p']:>7.4f} {c['r']:>7.4f} {c['f1']:>7.4f} {counts}"
            )
    if report.get("conf_sweep"):
        thrs = list(next(iter(report["conf_sweep"].values())))
        logger.info("\nF1 (micro) by confidence threshold")
        logger.info(f"{'conf':>12}  " + " ".join(f"{'IoU ' + t:>9}" for t in thrs))
        for conf, by_thr in report["conf_sweep"].items():
            logger.info(
                f"{conf:>12}  "
                + " ".join(f"{by_thr[t]['ALL']['f1']:>9.4f}" for t in thrs)
            )
    cm = report["confusion"]
    logger.info(
        f"\nConfusion at IoU {cm['iou']} (conf >= {cm['conf']}): "
        f"rows = ground truth, columns = prediction"
    )
    logger.info(f"{'':>12}  " + " ".join(f"{c[:7]:>7}" for c in cm["cols"]))
    for name, row in zip(cm["rows"], cm["matrix"]):
        logger.info(f"{name:>12}  " + " ".join(f"{v:>7}" for v in row))
    t = report["time_per_page"]
    logger.info(
        f"\nTime per page ({report['device']}): mean {t['mean']:.4f}s | "
        f"median {t['median']:.4f}s | total {t['total']:.1f}s"
    )


def _list_images(images_dir):
    """
    List the page images of a split directory, sorted.

    Args:
        images_dir: Directory holding .png / .jpg / .jpeg images

    Returns:
        list: Sorted list of image Paths
    """
    images_dir = Path(images_dir)
    return sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
    )


def evaluate(settings, logger, model=None, verbose=True):
    """
    Evaluate a layout model on a dataset split.

    Computes per-class AP at the first threshold and averaged over
    iou_thresholds_ap (COCO interpolation, all predictions above ap_conf), mAP over
    classes with ground truth, P/R/F1 at each of iou_thresholds_prf on predictions
    above conf, and the wall-clock prediction time per page.

    Args:
        settings: EvaluationSettings instance
        logger: Logger instance
        model: Already loaded model (e.g. during training); loaded from
            settings.path_model or the HF repo when None
        verbose: Log the full report tables

    Returns:
        dict: Evaluation report (also written to settings.out_json if set)
    """
    if model is None and settings.path_model:
        model = _load_model(settings.path_model)
    elif model is None:
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

    image_paths = _list_images(images_dir)
    if not image_paths:
        logger.error(f"No images in {images_dir}")
        return None

    import cv2

    _predict(model, image_paths[0], settings)

    totals = {
        thr: {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(num_classes)}
        for thr in settings.iou_thresholds_prf
    }
    sweep = {
        conf: {
            thr: {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(num_classes)}
            for thr in settings.iou_thresholds_prf
        }
        for conf in settings.conf_sweep
    }
    ap_flags = {i: [] for i in range(num_classes)}
    ap_confs = {i: [] for i in range(num_classes)}
    n_gt = np.zeros(num_classes, dtype=int)
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    times = []

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes_norm, gt_classes = _parse_yolo_label(label_path)

        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        gt_boxes = gt_boxes_norm.copy()
        if len(gt_boxes) > 0:
            gt_boxes[:, [0, 2]] *= w_img
            gt_boxes[:, [1, 3]] *= h_img

        pred_boxes, pred_classes, pred_confs, seconds = _predict(
            model, img_path, settings
        )
        times.append(seconds)

        for cls_id in range(num_classes):
            gt_mask = gt_classes == cls_id
            pred_mask = pred_classes == cls_id
            n_gt[cls_id] += int(gt_mask.sum())
            ap_flags[cls_id].append(
                _match_tp_flags(
                    gt_boxes[gt_mask],
                    pred_boxes[pred_mask],
                    pred_confs[pred_mask],
                    settings.iou_thresholds_ap,
                )
            )
            ap_confs[cls_id].append(pred_confs[pred_mask])

        keep = pred_confs >= settings.conf
        confusion += _confusion_matrix(
            gt_boxes,
            gt_classes,
            pred_boxes[keep],
            pred_classes[keep],
            pred_confs[keep],
            num_classes,
            settings.confusion_iou,
        )
        for thr in settings.iou_thresholds_prf:
            img_results = _match_predictions(
                gt_boxes,
                gt_classes,
                pred_boxes[keep],
                pred_classes[keep],
                pred_confs[keep],
                num_classes,
                iou_threshold=thr,
            )
            for cls_id in range(num_classes):
                for k in ("tp", "fp", "fn"):
                    totals[thr][cls_id][k] += img_results[cls_id][k]
        for conf in settings.conf_sweep:
            above = pred_confs >= conf
            for thr in settings.iou_thresholds_prf:
                img_results = _match_predictions(
                    gt_boxes,
                    gt_classes,
                    pred_boxes[above],
                    pred_classes[above],
                    pred_confs[above],
                    num_classes,
                    iou_threshold=thr,
                )
                for cls_id in range(num_classes):
                    for k in ("tp", "fp", "fn"):
                        sweep[conf][thr][cls_id][k] += img_results[cls_id][k]

    per_class = {}
    ap_matrix = np.full((num_classes, len(settings.iou_thresholds_ap)), np.nan)
    for cls_id in range(num_classes):
        flags = np.concatenate(ap_flags[cls_id])
        confs = np.concatenate(ap_confs[cls_id])
        for t_idx in range(len(settings.iou_thresholds_ap)):
            ap_matrix[cls_id, t_idx] = _average_precision(
                flags[:, t_idx], confs, n_gt[cls_id], settings.ap_recall_points
            )
        per_class[class_names_list[cls_id]] = {
            "n_gt": int(n_gt[cls_id]),
            "ap50": float(ap_matrix[cls_id, 0]),
            "ap50_95": float(np.mean(ap_matrix[cls_id])),
        }

    prf = {
        str(thr): _prf_table(totals[thr], class_names_list, n_gt)
        for thr in settings.iou_thresholds_prf
    }
    conf_sweep = {
        str(conf): {
            str(thr): _prf_table(sweep[conf][thr], class_names_list, n_gt)
            for thr in settings.iou_thresholds_prf
        }
        for conf in settings.conf_sweep
    }

    report = {
        "model": str(settings.path_model or settings.hf_filename),
        "split": settings.split,
        "n_pages": len(image_paths),
        "conf": settings.conf,
        "ap_conf": settings.ap_conf,
        "nms_iou": settings.iou,
        "imgsz": settings.imgsz,
        "device": settings.device,
        "n_gt": int(n_gt.sum()),
        "map50": float(np.nanmean(ap_matrix[:, 0])),
        "map50_95": float(np.nanmean(ap_matrix.mean(axis=1))),
        "per_class": per_class,
        "prf": prf,
        "conf_sweep": conf_sweep,
        "confusion": {
            "iou": settings.confusion_iou,
            "conf": settings.conf,
            "rows": class_names_list + ["background"],
            "cols": class_names_list + ["missed"],
            "matrix": confusion.tolist(),
        },
        "time_per_page": {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "total": float(np.sum(times)),
        },
    }

    if verbose:
        _log_report(report, logger)

    if settings.out_json:
        out = Path(settings.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"Report written to {out}")

    return report


def main():
    """CLI entry point for layout model evaluation."""
    defaults = EvaluationSettings()
    parser = argparse.ArgumentParser(description="Evaluate a layout detection model")
    parser.add_argument("--path-data", type=str, default=defaults.path_data)
    parser.add_argument("--model", type=str, default=defaults.path_model)
    parser.add_argument(
        "--split", type=str, default=defaults.split, choices=["train", "dev", "test"]
    )
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--ap-conf", type=float, default=defaults.ap_conf)
    parser.add_argument("--iou", type=float, default=defaults.iou)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--out-json", type=str, default=defaults.out_json)
    parser.add_argument(
        "--conf-sweep", type=float, nargs="*", default=list(defaults.conf_sweep)
    )
    args = parser.parse_args()

    settings = EvaluationSettings(
        path_data=args.path_data,
        path_model=args.model,
        split=args.split,
        conf=args.conf,
        ap_conf=args.ap_conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        out_json=args.out_json,
        conf_sweep=tuple(args.conf_sweep),
    )

    evaluate(settings, logger)


if __name__ == "__main__":
    main()
