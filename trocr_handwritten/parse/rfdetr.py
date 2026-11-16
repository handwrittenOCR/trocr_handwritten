import os
import shutil
from pathlib import Path

import numpy as np

from trocr_handwritten.parse.metrics import _list_images

SPLITS = {"train": "train", "dev": "valid", "test": "test"}


def build_rfdetr_dataset(path_data, out_dir, class_names_list):
    """
    Build the Roboflow-style YOLO layout that RF-DETR reads, without touching path_data.

    RF-DETR expects data.yaml with train/, valid/, test/ splits and refuses split
    directories that resolve outside the dataset root, so each split gets real
    images/ and labels/ directories filled with relative per-file symlinks to
    <path_data>/<split>/{images,labels}. dev/ becomes valid/.

    Args:
        path_data: Root layout directory with YOLO train/dev/test splits
        out_dir: Directory to create (e.g. <path_data>/models/<run>/dataset)
        class_names_list: Ordered list of class names

    Returns:
        Path: out_dir
    """
    root, out = Path(path_data), Path(out_dir)
    for split, name in SPLITS.items():
        images_dst = out / name / "images"
        labels_dst = out / name / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)
        for img in _list_images(root / split / "images"):
            label = root / split / "labels" / f"{img.stem}.txt"
            pairs = [(img, images_dst)] + (
                [(label, labels_dst)] if label.exists() else []
            )
            for src, dst_dir in pairs:
                dst = dst_dir / src.name
                if not dst.is_symlink():
                    dst.symlink_to(os.path.relpath(src, dst_dir))
    names = "\n".join(f"  - {c}" for c in class_names_list)
    (out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(class_names_list)}\nnames:\n{names}\n"
    )
    return out


class RFDETRDetector:
    """
    Fine-tuned RF-DETR checkpoint predicting the layout classes.

    The model is built on the first detect() call, on the requested device.
    RF-DETR has no NMS, so detect() ignores iou. Class ids are 0-based indices into
    the training class names, i.e. YOLO class ids, so metrics.evaluate scores it
    like the ultralytics models. antialias=True matches the torchvision resize used
    in training when rfdetr[augment] is not installed.
    """

    def __init__(self, checkpoint, antialias=True):
        self.checkpoint = str(checkpoint)
        self.antialias = antialias
        self.model = None
        self.device = None

    def detect(self, img_path, conf, iou, device):
        """
        Predict the regions of one page.

        Args:
            img_path: Path to the page image
            conf: Minimum score kept
            iou: Unused (RF-DETR has no NMS)
            device: Torch device string

        Returns:
            tuple: (boxes_xyxy, classes, confs) as numpy arrays, YOLO class ids
        """
        if self.model is None or self.device != device:
            from rfdetr import RFDETR

            self.model = RFDETR.from_checkpoint(self.checkpoint, device=device)
            self.device = device
        det = self.model.predict(
            str(img_path),
            threshold=conf,
            include_source_image=False,
            antialias=self.antialias,
        )
        if len(det) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        return det.xyxy, det.class_id.astype(int), det.confidence


def train_rfdetr(settings, device, logger):
    """
    Fine-tune RF-DETR Medium on a YOLO-format layout dataset.

    Trains on train/, validates on dev/ each epoch, keeps the best dev mAP@.5:.95
    (checkpoint_best_total.pth, the ultralytics criterion) and stops after
    settings.patience epochs without improvement. Outputs go to
    <path_data>/models/<run_name>/; the best checkpoint is copied to weights/best.pt.

    Args:
        settings: RFDETRSettings instance
        device: Resolved torch device string
        logger: Logger instance

    Returns:
        Path: Path to best.pt
    """
    from rfdetr import RFDETRMedium

    run_dir = Path(settings.path_data) / "models" / settings.run_name
    class_names_list = [
        settings.class_names[str(i)] for i in range(len(settings.class_names))
    ]
    dataset_dir = build_rfdetr_dataset(
        settings.path_data, run_dir / "dataset", class_names_list
    )
    logger.info(f"  RF-DETR dataset: {dataset_dir}, resolution {settings.imgsz}")
    model = RFDETRMedium(resolution=settings.imgsz)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=settings.epochs,
        batch_size=settings.batch,
        grad_accum_steps=settings.grad_accum_steps,
        lr=settings.lr,
        lr_encoder=settings.lr_encoder,
        weight_decay=settings.weight_decay,
        num_workers=settings.workers,
        early_stopping=True,
        early_stopping_patience=settings.patience,
        seed=settings.seed,
        device=device,
    )
    best = run_dir / "weights" / "best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "checkpoint_best_total.pth", best)
    return best
