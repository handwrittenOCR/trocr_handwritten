"""
simple_preprocess.py
────────────────────
Converts manuscript images to greyscale, increases contrast, and binarises.

Usage
─────
  python simple_preprocess.py
"""

from typing import List, Optional

import cv2
from pathlib import Path


input_dir = Path("data/raw/images")
output_dir = Path("data/raw/images/preprocess")
output_dir.mkdir(parents=True, exist_ok=True)  # create it if needed


def find_images(
    input_dir: Path, pattern: str, limit: Optional[int] = None
) -> List[Path]:
    """
    Find all images matching the pattern in the input directory.

    Args:
        input_dir: Root directory to search.
        pattern: Glob pattern to match images.
        limit: Maximum number of images to return.

    Returns:
        List of paths to image files.
    """
    images = sorted(input_dir.rglob(pattern))
    if limit:
        images = images[:limit]
    return images


def process(input_path: Path) -> None:
    img = cv2.imread(str(input_path))

    # 1. Greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Contrast enhancement (CLAHE — local adaptive, handles uneven parchment)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)

    # 3. Binarisation (adaptive threshold handles background brightness variation)
    binary = cv2.adaptiveThreshold(
        contrasted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=30,
    )

    out_path = output_dir / f"{input_path.stem}_simple.jpg"
    cv2.imwrite(str(out_path), binary, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}")


for image in find_images(Path(input_dir), "*.jpg"):
    process(image)
