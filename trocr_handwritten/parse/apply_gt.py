"""Rebuild page crops from ground-truth boxes, so OCR reads the boxes drawn in the editor.

For every ground-truth page <gt_dir>/<commune>/<year>/<page_id>.json drawn in the editor
(source "drawn"), the current output of the page in <ocr_dir>/<commune>/<year>/<page_id>/ (class
folders, metadata.json, metadata_reading_order.json) is moved to
<replaced_dir>/<commune>/<year>/<page_id>/<version>/ ("yolo", then "gt_<save time>" when a page
is redrawn), kept and never deleted, and Marge / Plein Texte crops are cut from the ground-truth
boxes with the file layout and reading order of layout_parser. gt_applied.json in the page folder
records the ground-truth file and its save time. The replaced folder must sit outside <ocr_dir>:
the OCR step transcribes every */*/*.jpg below its input folder.

Skipped: pages whose ground truth is the accepted YOLO output (source "yolo_accepted"), pages
already rebuilt from the same ground-truth version, and pages with an OCR transcription (.md),
whose text came from the old crops.

Usage:
    python -m trocr_handwritten.parse.apply_gt --gt-dir <ECES>/layout_gt \
        --ocr-dir <ECES>/OCR_gem31 --image-root D:/ECES \
        --replaced-dir <ECES>/OCR_gem31_replaced [--folders moule/1837 ...] [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

from trocr_handwritten.parse.utils import build_reading_order

TEXT_CLASSES = ("Marge", "Plein Texte")
YOLO_FILES = ("metadata.json", "metadata_reading_order.json")


def page_image(image_root: Path, commune: str, year: str, page_id: str) -> Path:
    """Staged page image, searched in the commune's other year folders when the year moved."""
    island = "Martinique" if commune.startswith("mar_") else "Guadeloupe"
    base = image_root / island / commune
    path = base / year / "pages" / f"{page_id}.jpg"
    return path if path.exists() else next(base.glob(f"*/pages/{page_id}.jpg"), path)


def write_crops(img, boxes: list[dict], page_dir: Path) -> dict:
    """Cut the Marge / Plein Texte boxes into <class>/<idx>.jpg and return the metadata."""
    h_img, w_img = img.shape[:2]
    metadata = {}
    for idx, b in enumerate(boxes):
        if b["label"] not in TEXT_CLASSES:
            continue
        x0, y0 = max(0, int(b["x"])), max(0, int(b["y"]))
        x1 = min(w_img, int(b["x"] + b["width"]))
        y1 = min(h_img, int(b["y"] + b["height"]))
        if x1 <= x0 or y1 <= y0:
            continue
        folder = page_dir / b["label"]
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{idx:03d}.jpg"
        cv2.imwrite(str(folder / name), img[y0:y1, x0:x1])
        metadata.setdefault(b["label"], []).append(
            {
                "cropped_image_name": name,
                "coordinates": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            }
        )
    metadata["reading_order"] = build_reading_order(metadata)
    return metadata


def apply_page(
    gt_file: Path, ocr_dir: Path, image_root: Path, replaced_dir: Path, dry_run: bool
) -> str:
    """Rebuild one page from its ground truth; returns what was done."""
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    commune, year, page_id = gt["commune"], str(gt["year"]), gt["page_id"]
    page_dir = ocr_dir / commune / year / page_id
    if gt.get("source") != "drawn":
        return "skip: yolo accepted"
    if not page_dir.is_dir():
        return "skip: no OCR page folder"
    marker = page_dir / "gt_applied.json"
    if marker.exists():
        done = json.loads(marker.read_text(encoding="utf-8"))
        if done.get("saved_at") == gt.get("saved_at"):
            return "skip: already applied"
    if any(page_dir.rglob("*.md")):
        return "skip: already OCRed"
    img_path = page_image(image_root, commune, year, page_id)
    if not img_path.exists():
        return f"skip: no image {img_path}"
    if dry_run:
        return "would apply"
    version = "yolo"
    if marker.exists():
        prev = json.loads(marker.read_text(encoding="utf-8")).get("saved_at", "")
        version = "gt_" + "".join(ch for ch in str(prev) if ch.isalnum())
    keep = replaced_dir / commune / year / page_id / version
    keep.mkdir(parents=True, exist_ok=True)
    for p in list(page_dir.iterdir()):
        if p.is_dir() or p.name in YOLO_FILES or p.name == marker.name:
            shutil.move(str(p), str(keep / p.name))
    metadata = write_crops(cv2.imread(str(img_path)), gt["boxes"], page_dir)
    (page_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "gt_file": str(gt_file),
                "saved_at": gt.get("saved_at"),
                "annotator": gt.get("annotator"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "applied"


def main():
    """Apply every drawn ground-truth page, optionally restricted to commune/year folders."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--ocr-dir", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--replaced-dir", required=True)
    ap.add_argument("--folders", nargs="*", default=None, help="commune/year")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    gt_dir = Path(a.gt_dir)
    files = sorted(
        p for p in gt_dir.glob("*/*/*.json") if not p.parts[-3].startswith("_")
    )
    if a.folders:
        files = [p for p in files if f"{p.parts[-3]}/{p.parts[-2]}" in a.folders]
    counts = {}
    for f in files:
        status = apply_page(
            f, Path(a.ocr_dir), Path(a.image_root), Path(a.replaced_dir), a.dry_run
        )
        key = "skip: no image" if status.startswith("skip: no image") else status
        counts[key] = counts.get(key, 0) + 1
        if status not in ("skip: yolo accepted", "skip: already applied"):
            print(f"{f.parts[-3]}/{f.parts[-2]}/{f.stem}: {status}")
    print(counts)


if __name__ == "__main__":
    main()
