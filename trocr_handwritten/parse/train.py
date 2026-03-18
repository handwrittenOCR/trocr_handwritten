import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from trocr_handwritten.parse.settings import TrainingSettings
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
    Train a YOLO11 model on layout detection data.

    Args:
        settings: TrainingSettings instance
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

    model = YOLO(settings.model_base)

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
        workers=4,
        freeze=settings.freeze,
        dropout=settings.dropout,
        weight_decay=settings.weight_decay,
    )

    best_pt = models_dir / settings.run_name / "weights" / "best.pt"
    if best_pt.exists():
        out = models_dir / "best.pt"
        shutil.copy2(best_pt, out)
        logger.info(f"Best model saved: {out}")
        return out

    logger.warning("best.pt not found after training")
    return None


def main():
    """CLI entry point for layout YOLO training."""
    parser = argparse.ArgumentParser(description="Train YOLO11 on layout annotations")
    parser.add_argument("--path-data", type=str, default="data/layout")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--model-base", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.name or f"e{args.epochs}_b{args.batch}_f{args.freeze}"

    settings = TrainingSettings(
        path_data=args.path_data,
        model_base=args.model_base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        freeze=args.freeze,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        run_name=run_name,
    )

    train(settings, args.images_dir, logger)


if __name__ == "__main__":
    main()
