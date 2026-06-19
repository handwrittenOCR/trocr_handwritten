import colorsys
import json
import random
import re
import shutil
from pathlib import Path

SPLIT_WEIGHTS = {"train": 0.7, "test": 0.2, "dev": 0.1}
SPLITS = list(SPLIT_WEIGHTS.keys())


def page_key(filename):
    """
    Return the page a crop belongs to by stripping the trailing crop index.

    A crop is named ``<page>_<crop_index>`` across every corpus (FRANOM, FRCAOM,
    DAFCAOM, p-variant, ...). Removing the final ``_<digits>`` segment yields a
    stable page identifier so that all crops of the same page share one key.

    Args:
        filename: Crop file name or stem.

    Returns:
        str: The page key.
    """
    stem = Path(filename).stem
    m = re.match(r"^(.*)_\d+$", stem)
    return m.group(1) if m else stem


def assign_split(filename=None, annotations=None):
    """
    Assign a split, keeping all crops of a page together to avoid leakage.

    When ``filename`` and ``annotations`` are provided and another crop of the
    same page already has a split, that split is reused so a page never straddles
    train/test/dev. Otherwise a weighted random draw is used (train=0.7, test=0.2,
    dev=0.1).

    Args:
        filename: Crop file name, used to derive the page key.
        annotations: Existing annotation dicts (each with ``filename`` and ``split``).

    Returns:
        str: The assigned split.
    """
    if filename is not None and annotations:
        key = page_key(filename)
        for a in annotations:
            if a.get("split") and page_key(a["filename"]) == key:
                return a["split"]
    r = random.random()
    cumulative = 0.0
    for split, weight in SPLIT_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return split
    return "train"


def generate_colors(n):
    """
    Generate n visually distinct colors using HSV spacing.

    Args:
        n: Number of colors to generate

    Returns:
        list: List of hex color strings
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def load_split_annotations(base_dir):
    """
    Load annotations from all split JSON files into a single list.

    Args:
        base_dir: Root directory containing train/test/dev subdirectories

    Returns:
        list: All annotations with 'split' field added
    """
    all_annotations = []
    base = Path(base_dir)
    for split in SPLITS:
        path = base / split / "annotations.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
                for entry in entries:
                    entry["split"] = split
                all_annotations.extend(entries)
    return all_annotations


def save_split_annotations(annotations, base_dir):
    """
    Save annotations to per-split JSON files.

    Args:
        annotations: List of annotation dicts, each with a 'split' field
        base_dir: Root directory for splits
    """
    base = Path(base_dir)
    by_split = {s: [] for s in SPLITS}
    for a in annotations:
        split = a.get("split", "train")
        by_split[split].append(a)

    for split, entries in by_split.items():
        split_dir = base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / "annotations.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)


def _collect_split_crops(base):
    """
    Scan train/dev/test on disk and return one record per crop.

    Disk is the source of truth: annotations.json may be incomplete, so crops are
    enumerated from the image files and matched with their label text when present.

    Args:
        base: Root directory containing the split subdirectories.

    Returns:
        list: Records with filename, subfolder, split, page and text.
    """
    extensions = ("*.jpg", "*.jpeg", "*.png")
    records = []
    for split in SPLITS:
        images_root = base / split / "images"
        if not images_root.exists():
            continue
        imgs = []
        for ext in extensions:
            imgs.extend(images_root.rglob(ext))
        for img in sorted(imgs):
            subfolder = img.parent.name
            label = base / split / "labels" / subfolder / (img.stem + ".txt")
            text = label.read_text(encoding="utf-8").strip() if label.exists() else ""
            records.append(
                {
                    "filename": img.name,
                    "subfolder": subfolder,
                    "split": split,
                    "page": page_key(img.name),
                    "text": text,
                }
            )
    return records


def _move_crop(base, subfolder, filename, old_split, new_split):
    """
    Move a crop's image and label file from one split to another.

    Args:
        base: Root directory containing the split subdirectories.
        subfolder: Region subfolder the crop lives in.
        filename: Crop image file name.
        old_split: Current split.
        new_split: Destination split.
    """
    stem = Path(filename).stem
    pairs = [
        (
            base / old_split / "images" / subfolder / filename,
            base / new_split / "images" / subfolder / filename,
        ),
        (
            base / old_split / "labels" / subfolder / f"{stem}.txt",
            base / new_split / "labels" / subfolder / f"{stem}.txt",
        ),
    ]
    for src, dst in pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def regroup_by_page(base_dir, seed=42, logger=None):
    """
    Re-split existing OCR data grouped by page to remove train/test leakage.

    Every crop of a page is forced into the same split. Whole pages are greedily
    packed to match SPLIT_WEIGHTS by crop count. Image and label files are moved to
    their new split and annotations.json is rebuilt complete and consistent with
    disk.

    Args:
        base_dir: Root directory containing the split subdirectories.
        seed: Seed controlling the page shuffle for reproducibility.
        logger: Optional logger for progress reporting.

    Returns:
        dict: Per-split crop counts.
    """
    base = Path(base_dir)
    records = _collect_split_crops(base)
    if not records:
        if logger:
            logger.warning(f"No crops found under {base_dir}")
        return {s: 0 for s in SPLITS}

    unique = {}
    for r in records:
        unique.setdefault((r["subfolder"], r["filename"]), r)
    total = len(unique)

    pages = {}
    for r in unique.values():
        pages.setdefault(r["page"], []).append(r)

    rng = random.Random(seed)
    items = list(pages.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    counts = {s: 0 for s in SPLITS}
    page_split = {}
    for key, group in items:
        need = {s: SPLIT_WEIGHTS[s] * total - counts[s] for s in SPLITS}
        target = max(need, key=need.get)
        page_split[key] = target
        counts[target] += len(group)

    moved = 0
    for r in records:
        new = page_split[r["page"]]
        if new != r["split"]:
            _move_crop(base, r["subfolder"], r["filename"], r["split"], new)
            moved += 1

    final = _collect_split_crops(base)
    annotations = [
        {
            "filename": r["filename"],
            "subfolder": r["subfolder"],
            "text": r["text"],
            "split": r["split"],
        }
        for r in final
    ]
    save_split_annotations(annotations, base)

    final_counts = {s: sum(1 for r in final if r["split"] == s) for s in SPLITS}
    if logger:
        duplicates = len(records) - total
        logger.info(
            f"Regrouped {total} crops over {len(pages)} pages, moved {moved}, "
            f"dropped {duplicates} duplicate copies"
        )
        for s in SPLITS:
            pct = final_counts[s] / total * 100 if total else 0
            logger.info(f"  {s}: {final_counts[s]} ({pct:.1f}%)")
    return final_counts


def collect_images(images_dir):
    """
    Collect all image files from a directory.

    Args:
        images_dir: Directory to search

    Returns:
        list: Sorted list of Path objects
    """
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in extensions:
        images.extend(Path(images_dir).glob(ext))
    return sorted(images)


def collect_images_recursive(images_dir):
    """
    Collect all image files recursively from a directory (resolved absolute paths).

    Args:
        images_dir: Root directory to search

    Returns:
        list: Sorted list of resolved Path objects
    """
    extensions = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in extensions:
        images.extend(p.resolve() for p in Path(images_dir).rglob(ext))
    return sorted(images)


def import_legacy_annotations(input_dir, output_dir, logger):
    """
    Import annotated data from legacy format ({subfolder}/images/*.jpg + label/*.txt)
    into the new split structure (data/ocr/{split}/images/{subfolder}/ + labels/).

    Args:
        input_dir: Directory with annotated data (subfolders with images/ and label/)
        output_dir: Output directory (e.g. data/ocr)
        logger: Logger instance
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    annotations = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    page_to_split = {}

    subfolders = [
        d
        for d in sorted(input_path.iterdir())
        if d.is_dir() and (d / "images").exists()
    ]

    if not subfolders:
        logger.error(f"No subfolders with images/ found in {input_dir}")
        return

    random.seed(42)

    for subfolder in subfolders:
        subfolder_name = subfolder.name
        images_dir = subfolder / "images"
        labels_dir = subfolder / "label"

        if not labels_dir.exists():
            logger.warning(f"No label/ directory in {subfolder}, skipping")
            continue

        image_files = collect_images(images_dir)

        for image_path in image_files:
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            key = page_key(image_path.name)
            split = page_to_split.get(key)
            if split is None:
                split = assign_split()
                page_to_split[key] = split

            dst_images = output_path / split / "images" / subfolder_name
            dst_labels = output_path / split / "labels" / subfolder_name
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, dst_images / image_path.name)
            shutil.copy2(label_path, dst_labels / (image_path.stem + ".txt"))

            annotations[split].append(
                {
                    "filename": image_path.name,
                    "subfolder": subfolder_name,
                    "text": text,
                }
            )
            counts[split] += 1

    for split in SPLITS:
        if not annotations[split]:
            continue
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        ann_path = split_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations[split], f, indent=2, ensure_ascii=False)

    total = sum(counts.values())
    logger.info(f"Imported {total} samples from {input_dir}")
    for split in SPLITS:
        logger.info(f"  {split}: {counts[split]}")


ANNOTATE_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column;
       overflow: hidden; user-select: none; }
.header { background: #16213e; padding: 0.6rem 1.5rem; display: flex;
          align-items: center; justify-content: space-between; flex-shrink: 0; }
.header h1 { font-size: 0.95rem; font-weight: 500; color: #a8b2d1; }
.header-right { display: flex; align-items: center; gap: 1rem; }
.progress { font-size: 0.8rem; color: #8892b0; }
.stats { font-size: 0.75rem; color: #64ffda; }
.controls { background: #0f1729; padding: 0.5rem 1rem; display: flex;
            justify-content: center; gap: 0.6rem; flex-shrink: 0; }
.btn { padding: 0.5rem 1.5rem; border: none; border-radius: 5px; font-size: 0.85rem;
       font-weight: 600; cursor: pointer; transition: all 0.1s; }
.btn-nav { background: #495670; color: #ccd6f6; }
.btn-nav:hover { background: #5a6a8a; }
.btn-save { background: #64ffda; color: #0a192f; }
.btn-save:hover { background: #4cd9b0; }
.btn-clear { background: #f07178; color: #0a192f; }
.btn-clear:hover { background: #d45d63; }
.shortcut { font-size: 0.65rem; color: #8892b0; display: block; margin-top: 0.1rem; }
.toast { position: fixed; top: 1rem; right: 1rem; padding: 0.6rem 1.2rem;
         background: #64ffda; color: #0a192f; border-radius: 5px; font-weight: 600;
         font-size: 0.85rem; opacity: 0; transition: opacity 0.3s; z-index: 999;
         pointer-events: none; }
.toast.show { opacity: 1; }
"""


def main():
    """Regroup an existing OCR split by page to remove leakage (CLI entry point)."""
    import argparse
    from trocr_handwritten.utils.logging_config import get_logger

    parser = argparse.ArgumentParser(
        description="Re-split OCR data grouped by page (no train/test leakage)."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="data/ocr",
        help="Root directory containing train/dev/test (default: data/ocr)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    regroup_by_page(args.base_dir, seed=args.seed, logger=get_logger(__name__))


if __name__ == "__main__":
    main()
