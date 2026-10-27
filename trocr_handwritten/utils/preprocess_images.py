import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def divide_background(gray, sigma_ratio=20.0):
    """
    Flatten uneven illumination by dividing the image by its own blurred version.

    The wide Gaussian keeps only the low-frequency lighting -- shadows, page
    curvature, yellowing -- while ink lives in the high frequencies. Dividing
    removes the former and leaves local ink contrast untouched, unlike a global
    histogram stretch.
    """
    sigma = max(gray.shape) / sigma_ratio
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    background = np.maximum(background, 1.0)
    flat = gray.astype(np.float32) / background
    lo, hi = np.percentile(flat, [1, 99])
    if hi - lo < 1e-6:
        return gray
    return np.clip((flat - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def apply_clahe(gray, clip_limit=2.0, tile=8):
    """Boost local contrast with tile-wise adaptive histogram equalisation."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)


METHODS = {
    "bgdiv": lambda g, a: divide_background(g, a.sigma_ratio),
    "clahe": lambda g, a: apply_clahe(g, a.clip_limit, a.tile),
    "bgdiv+clahe": lambda g, a: apply_clahe(
        divide_background(g, a.sigma_ratio), a.clip_limit, a.tile
    ),
}


def process(path, method, args):
    """Read one crop, normalise it, and return a 3-channel image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = METHODS[method](gray, args)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def main():
    """
    Write an illumination-normalised copy of an image tree, mirroring its layout.

    Meant to be measured against the untouched tree with the same model, so the
    three variants stay nested: raw, background division, then division followed
    by CLAHE. Output stays 3-channel JPEG so nothing downstream changes.
    """
    ap = argparse.ArgumentParser(description="Normalise illumination on OCR crops.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=sorted(METHODS), default="bgdiv")
    ap.add_argument("--pattern", default="*/*.jpg")
    ap.add_argument(
        "--sigma-ratio",
        type=float,
        default=20.0,
        help="Background blur sigma = longest side / this. Lower = flatter.",
    )
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    images = sorted(src.glob(args.pattern))
    logger.info(f"{len(images)} images, method={args.method} -> {dst}")

    written = failed = 0
    for path in tqdm(images, desc=args.method):
        out = process(path, args.method, args)
        if out is None:
            failed += 1
            continue
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        written += 1
    logger.info(f"wrote {written}, failed {failed}")


if __name__ == "__main__":
    main()
