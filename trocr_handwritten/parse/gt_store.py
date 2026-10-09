"""Ground-truth layout boxes, stored apart from the YOLO output so that no YOLO rerun, crop
rebuild or deleted output folder can touch them.

Layout of the store:
    <gt_dir>/<commune>/<year>/<page_id>.json      current ground truth of a page
    <gt_dir>/_history/<commune>/<year>/<page_id>_<timestamp>.json   every saved version
A page's file is never overwritten without its previous version being kept in _history.
"""

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from trocr_handwritten.parse.settings import CLASS_NAMES

LABEL_TO_ID = {v: int(k) for k, v in CLASS_NAMES.items()}
BOX_KEYS = ("label", "x", "y", "width", "height")


def gt_path(gt_dir: Path, commune: str, year: str, page_id: str) -> Path:
    """Path of a page's ground-truth file."""
    return Path(gt_dir) / commune / str(year) / f"{page_id}.json"


def load_gt(path: Path) -> dict | None:
    """Ground-truth record of a page, or None when the page has none."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def boxes_from_metadata(metadata_path: Path) -> list[dict]:
    """YOLO boxes of a page (metadata.json of the crop step) in the editor's box format."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return []
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    boxes = []
    for label, regions in meta.items():
        if label not in LABEL_TO_ID or not isinstance(regions, list):
            continue
        for r in regions:
            c = r["coordinates"]
            boxes.append(
                {
                    "class_id": LABEL_TO_ID[label],
                    "label": label,
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                    "width": float(c["width"]),
                    "height": float(c["height"]),
                }
            )
    return boxes


def _rounded(boxes: list[dict]) -> list[tuple]:
    """Order-free, rounded view of a box list, to tell an accepted prefill from an edit."""
    return sorted(
        tuple(round(float(b[k])) if k != "label" else b[k] for k in BOX_KEYS)
        for b in boxes
    )


def save_gt(
    gt_dir: Path,
    commune: str,
    year: str,
    page_id: str,
    boxes: list[dict],
    image_size: tuple[int, int],
    yolo_boxes: list[dict],
    annotator: str = "",
) -> Path:
    """Write a page's ground truth; keep the previous version in _history; return the path.

    source is 'yolo_accepted' when the saved boxes equal the YOLO prefill, else 'drawn'.
    """
    path = gt_path(gt_dir, commune, year, page_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    hist = Path(gt_dir) / "_history" / commune / str(year)
    hist.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, hist / f"{page_id}_{stamp}_previous.json")
    record = {
        "commune": commune,
        "year": str(year),
        "page_id": page_id,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "boxes": [
            {k: (b[k] if k == "label" else round(float(b[k]))) for k in BOX_KEYS}
            for b in boxes
        ],
        "source": (
            "yolo_accepted" if _rounded(boxes) == _rounded(yolo_boxes) else "drawn"
        ),
        "n_yolo_boxes": len(yolo_boxes),
        "annotator": annotator,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    text = json.dumps(record, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    (hist / f"{page_id}_{stamp}.json").write_text(text, encoding="utf-8")
    return path


def read_queue(queue_csv: Path) -> list[dict]:
    """Pages to annotate: rows with commune, year, page_id, image_path, metadata_path."""
    with open(queue_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))
