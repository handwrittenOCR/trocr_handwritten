import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from trocr_handwritten.parse.settings import (
    FasterRCNNSettings,
    RFDETRSettings,
    TrainingSettings,
)
from trocr_handwritten.parse.utils import _load_model, _detect_model_backend
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def _prepare_yolo_data(path_data, images_dir, logger):
    """
    Read annotations.json from each split and create YOLO images/ + labels/ directories.
    Images are symlinked from the source images_dir. Labels are generated as .txt YOLO format.

    Args:
        path_data: Root layout directory (e.g. data/layout)
        images_dir: Source directory where the original images live
        logger: Logger instance
    """
    layout_dir = Path(path_data)
    source_dir = Path(images_dir).resolve()

    for split in ["train", "test", "dev"]:
        ann_path = layout_dir / split / "annotations.json"
        if not ann_path.exists():
            continue

        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)

        images_out = layout_dir / split / "images"
        labels_out = layout_dir / split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for entry in annotations:
            filename = entry["filename"]
            img_w = entry.get("image_width", 1)
            img_h = entry.get("image_height", 1)
            boxes = entry.get("boxes", [])

            if not boxes:
                continue

            src_image = source_dir / filename
            if not src_image.exists():
                continue

            dst_image = images_out / filename
            if not dst_image.exists():
                dst_image.symlink_to(src_image)

            stem = Path(filename).stem
            label_lines = []
            for box in boxes:
                class_id = box["class_id"]
                cx = (box["x"] + box["width"] / 2) / img_w
                cy = (box["y"] + box["height"] / 2) / img_h
                w = box["width"] / img_w
                h = box["height"] / img_h
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path = labels_out / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines))
            count += 1

        logger.info(f"  {split}: prepared {count} images+labels from annotations")


def _build_dataset_yaml(path_data, class_names_list):
    """
    Generate a dataset.yaml file for YOLO training.
    Picks the first non-empty split for val (dev > test > train).

    Args:
        path_data: Root directory containing train/test/dev splits
        class_names_list: Ordered list of class names

    Returns:
        Path: Path to the generated dataset.yaml
    """
    layout_dir = Path(path_data).resolve()
    yaml_path = layout_dir / "dataset.yaml"

    val_split = "train"
    for candidate in ["dev", "test"]:
        candidate_dir = layout_dir / candidate / "images"
        if candidate_dir.exists() and any(candidate_dir.iterdir()):
            val_split = candidate
            break

    test_split = val_split
    for candidate in ["test", "dev"]:
        candidate_dir = layout_dir / candidate / "images"
        if candidate_dir.exists() and any(candidate_dir.iterdir()):
            test_split = candidate
            break

    names_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(class_names_list))
    content = (
        f"path: {layout_dir}\n"
        f"train: train/images\n"
        f"val: {val_split}/images\n"
        f"test: {test_split}/images\n"
        f"\n"
        f"nc: {len(class_names_list)}\n"
        f"names:\n"
        f"{names_str}\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def _count_split(path_data, split):
    """
    Count images and labels in a dataset split.

    Args:
        path_data: Root directory of the dataset
        split: Split name (train or test)

    Returns:
        tuple: (num_images, num_labels)
    """
    layout_dir = Path(path_data)
    images = list((layout_dir / split / "images").glob("*.[jp][pn]g"))
    labels = list((layout_dir / split / "labels").glob("*.txt"))
    return len(images), len(labels)


def _restore_numpy_trapz():
    """
    Alias np.trapz to np.trapezoid when NumPy no longer provides it (removed in 2.4).
    doclayout-yolo 0.0.4 calls np.trapz in its validation AP computation.
    """
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid


def _skip_doclayout_amp_check():
    """
    Replace the doclayout-yolo 0.0.4 AMP check by "AMP on when the model is on CUDA".
    The check downloads yolov8n.pt from the non-existent doclayout_yolo/assets GitHub
    repo and its FileNotFoundError is not caught, which aborts training.
    """
    from doclayout_yolo.engine import trainer

    trainer.check_amp = lambda model: next(model.parameters()).device.type == "cuda"


def _get_device():
    """
    Detect the best available device.

    Returns:
        str: Device string (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(settings, images_dir, logger):
    """
    Train a layout detector picked from model_base: YOLO, RT-DETR, DocLayout-YOLO
    (ultralytics / doclayout-yolo), Faster R-CNN ("fasterrcnn_*", torchvision) or
    RF-DETR ("rfdetr_*", rfdetr).

    Args:
        settings: TrainingSettings, FasterRCNNSettings or RFDETRSettings instance
        images_dir: Source directory containing the original images
        logger: Logger instance

    Returns:
        Path: Path to the best model checkpoint
    """
    logger.info("Preparing YOLO data from annotations...")
    _prepare_yolo_data(settings.path_data, images_dir, logger)

    class_names_list = [
        settings.class_names[str(i)] for i in range(len(settings.class_names))
    ]
    yaml_path = _build_dataset_yaml(settings.path_data, class_names_list)
    logger.info(f"Dataset YAML created: {yaml_path}")

    for split in ["train", "test"]:
        ni, nl = _count_split(settings.path_data, split)
        logger.info(f"  {split}: {ni} images, {nl} labels")

    ni_train, _ = _count_split(settings.path_data, "train")
    if ni_train == 0:
        logger.error("No training images found.")
        return None

    device = settings.device if settings.device != "auto" else _get_device()
    logger.info(
        f"  device: {device}, imgsz: {settings.imgsz}, "
        f"batch: {settings.batch}, freeze: {settings.freeze}"
    )

    models_dir = Path(settings.path_data) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if settings.model_base.startswith("fasterrcnn"):
        from trocr_handwritten.parse.frcnn import train_frcnn

        logger.info(f"  model: {settings.model_base} (torchvision)")
        return _copy_best(train_frcnn(settings, device, logger), models_dir, logger)

    if settings.model_base.startswith("rfdetr"):
        from trocr_handwritten.parse.rfdetr import train_rfdetr

        logger.info(f"  model: {settings.model_base} (rfdetr)")
        return _copy_best(train_rfdetr(settings, device, logger), models_dir, logger)

    backend = _detect_model_backend(settings.model_base)
    if backend == "doclayout":
        _restore_numpy_trapz()
        _skip_doclayout_amp_check()
    model = _load_model(settings.model_base, backend=backend)
    logger.info(f"  model: {settings.model_base} ({backend}, {type(model).__name__})")

    model.train(
        data=str(yaml_path),
        epochs=settings.epochs,
        imgsz=settings.imgsz,
        batch=settings.batch,
        device=device,
        project=str(models_dir.resolve()),
        name=settings.run_name,
        exist_ok=True,
        patience=settings.patience,
        save=True,
        plots=True,
        workers=settings.workers,
        freeze=settings.freeze,
        dropout=settings.dropout,
        weight_decay=settings.weight_decay,
    )

    return _copy_best(
        models_dir / settings.run_name / "weights" / "best.pt", models_dir, logger
    )


def _copy_best(best_pt, models_dir, logger):
    """
    Copy a run's best.pt to <models_dir>/best.pt.

    Args:
        best_pt: Path to the run's best checkpoint
        models_dir: Directory holding all runs
        logger: Logger instance

    Returns:
        Path: Path to the copy, or None if best_pt does not exist
    """
    if best_pt.exists():
        out = models_dir / "best.pt"
        shutil.copy2(best_pt, out)
        logger.info(f"Best model saved: {out}")
        return out

    logger.warning("best.pt not found after training")
    return None


def main():
    """
    CLI entry point for layout detector training.
    Unset options keep the defaults of TrainingSettings, of FasterRCNNSettings
    when --model-base starts with "fasterrcnn", or of RFDETRSettings when it starts
    with "rfdetr".
    """
    parser = argparse.ArgumentParser(description="Train a layout detector")
    parser.add_argument("--path-data", type=str)
    parser.add_argument("--images-dir", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--name", dest="run_name", type=str)
    overrides = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    model_base = overrides.get("model_base", TrainingSettings.model_base)
    settings_cls = TrainingSettings
    if model_base.startswith("fasterrcnn"):
        settings_cls = FasterRCNNSettings
    elif model_base.startswith("rfdetr"):
        settings_cls = RFDETRSettings
    settings = settings_cls(**overrides)
    if "run_name" not in overrides:
        settings.run_name = f"e{settings.epochs}_b{settings.batch}_f{settings.freeze}"

    train(settings, settings.images_dir, logger)


if __name__ == "__main__":
    main()
