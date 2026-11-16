import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image, read_file
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trocr_handwritten.parse.metrics import _list_images, _parse_yolo_label, evaluate
from trocr_handwritten.parse.settings import EvaluationSettings

ARCH = "fasterrcnn_resnet50_fpn"


def _read_image(img_path):
    """
    Read a page as a float RGB tensor in [0, 1].

    Args:
        img_path: Path to the image

    Returns:
        torch.Tensor: (3, H, W) float tensor
    """
    data = read_file(str(img_path))
    return decode_image(data, mode=ImageReadMode.RGB).float() / 255.0


def _collate(batch):
    """
    Keep images and targets as lists, as torchvision detection models expect.

    Args:
        batch: List of (image, target) pairs

    Returns:
        tuple: (images, targets)
    """
    return tuple(zip(*batch))


class YoloDetectionDataset(Dataset):
    """
    Adapter from a YOLO split (images/ + labels/) to torchvision detection targets.

    Boxes become absolute xyxy pixels; class ids are shifted by one since
    torchvision reserves label 0 for the background. Degenerate boxes are dropped
    and pages without boxes yield empty targets.
    """

    def __init__(self, split_dir, hflip=0.0):
        self.split_dir = Path(split_dir)
        self.image_paths = _list_images(self.split_dir / "images")
        self.hflip = hflip

    def __len__(self):
        """Number of pages in the split."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load one page and its target, with an optional horizontal flip.

        Args:
            idx: Page index

        Returns:
            tuple: (image tensor, {"boxes": (N, 4) float, "labels": (N,) int64})
        """
        img_path = self.image_paths[idx]
        image = _read_image(img_path)
        h, w = image.shape[1:]
        boxes, classes = _parse_yolo_label(
            self.split_dir / "labels" / f"{img_path.stem}.txt"
        )
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes *= torch.tensor([w, h, w, h], dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64) + 1
        if self.hflip and torch.rand(1).item() < self.hflip:
            image = image.flip(-1)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return image, {"boxes": boxes[keep], "labels": labels[keep]}


class FasterRCNNDetector:
    """
    Faster R-CNN R50-FPN (torchvision) predicting the layout classes.

    Images are resized so that their longest side equals imgsz, like the YOLO
    letterbox. detect() returns YOLO class ids, so metrics.evaluate scores it like
    the ultralytics models.
    """

    def __init__(
        self,
        num_classes,
        imgsz,
        max_det,
        pretrained=False,
        trainable_backbone_layers=None,
    ):
        self.num_classes = num_classes
        self.imgsz = imgsz
        self.max_det = max_det
        if pretrained:
            kwargs = {
                "weights": "DEFAULT",
                "trainable_backbone_layers": trainable_backbone_layers,
            }
        else:
            kwargs = {"weights": None, "weights_backbone": None}
        self.model = fasterrcnn_resnet50_fpn(
            min_size=imgsz, max_size=imgsz, box_detections_per_img=max_det, **kwargs
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

    @torch.no_grad()
    def detect(self, img_path, conf, iou, device):
        """
        Predict the regions of one page.

        Args:
            img_path: Path to the page image
            conf: Minimum score kept
            iou: NMS IoU threshold (per class)
            device: Torch device string

        Returns:
            tuple: (boxes_xyxy, classes, confs) as numpy arrays, YOLO class ids
        """
        self.model.to(device).eval()
        self.model.roi_heads.score_thresh = conf
        self.model.roi_heads.nms_thresh = iou
        out = self.model([_read_image(img_path).to(device)])[0]
        return (
            out["boxes"].cpu().numpy(),
            out["labels"].cpu().numpy().astype(int) - 1,
            out["scores"].cpu().numpy(),
        )

    def save(self, path, **extra):
        """
        Save the weights and the architecture parameters.

        Args:
            path: Output .pt path
            **extra: Extra scalar fields stored in the checkpoint (epoch, metric)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": ARCH,
                "num_classes": self.num_classes,
                "imgsz": self.imgsz,
                "max_det": self.max_det,
                "state_dict": self.model.state_dict(),
                **extra,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path):
        """
        Rebuild a detector from a checkpoint written by save().

        Args:
            path: Checkpoint path

        Returns:
            FasterRCNNDetector: Detector on CPU
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        detector = cls(ckpt["num_classes"], ckpt["imgsz"], ckpt["max_det"])
        detector.model.load_state_dict(ckpt["state_dict"])
        return detector


def _lr_lambda(warmup_iters, total_iters):
    """
    Linear warmup then cosine decay to zero, as a LambdaLR factor.

    Args:
        warmup_iters: Number of warmup iterations
        total_iters: Total number of iterations

    Returns:
        callable: iteration -> learning-rate factor
    """

    def factor(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def train_frcnn(settings, device, logger):
    """
    Fine-tune Faster R-CNN R50-FPN on a YOLO-format layout dataset.

    Trains on train/, validates on dev/ after each epoch with metrics.evaluate and
    keeps the checkpoint with the best dev mAP@.5:.95 (the ultralytics criterion).
    Stops after settings.patience epochs without improvement. Checkpoints go to
    <path_data>/models/<run_name>/weights/{best,last}.pt.

    Args:
        settings: FasterRCNNSettings instance
        device: Resolved torch device string
        logger: Logger instance

    Returns:
        Path: Path to best.pt
    """
    torch.manual_seed(settings.seed)
    root = Path(settings.path_data)
    weights_dir = root / "models" / settings.run_name / "weights"
    detector = FasterRCNNDetector(
        len(settings.class_names),
        settings.imgsz,
        settings.max_det,
        pretrained=settings.pretrained,
        trainable_backbone_layers=settings.trainable_backbone_layers,
    )
    model = detector.model.to(device)

    loader = DataLoader(
        YoloDetectionDataset(root / "train", settings.hflip),
        batch_size=settings.batch,
        shuffle=True,
        num_workers=settings.workers,
        collate_fn=_collate,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=settings.lr,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
    )
    total_iters = settings.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(min(settings.warmup_iters, total_iters), total_iters)
    )
    device_type = torch.device(device).type
    amp = device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=amp)
    eval_settings = EvaluationSettings(
        path_data=settings.path_data,
        split="dev",
        imgsz=settings.imgsz,
        device=device,
        class_names=settings.class_names,
    )

    best, stale = -1.0, 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        start = time.perf_counter()
        running = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device_type, enabled=amp):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        report = evaluate(eval_settings, logger, model=detector, verbose=False)
        fitness = report["map50_95"]
        logger.info(
            f"epoch {epoch}/{settings.epochs} | loss {running / len(loader):.4f} | "
            f"dev mAP50 {report['map50']:.4f} mAP50:95 {fitness:.4f} | "
            f"{time.perf_counter() - start:.0f}s"
        )
        detector.save(weights_dir / "last.pt", epoch=epoch, map50_95=fitness)
        if fitness > best:
            best, stale = fitness, 0
            detector.save(weights_dir / "best.pt", epoch=epoch, map50_95=fitness)
        else:
            stale += 1
            if stale >= settings.patience:
                logger.info(f"Early stopping: no improvement for {stale} epochs")
                break

    logger.info(f"Best dev mAP50:95 {best:.4f}")
    return weights_dir / "best.pt"
