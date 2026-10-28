import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from trocr_handwritten.htr.kraken_engine import KrakenEngine
from trocr_handwritten.htr.pylaia_engine import PyLaiaEngine
from trocr_handwritten.htr.settings import HTRSettings
from trocr_handwritten.llm.ocr import find_images
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

ENGINES = {"kraken": KrakenEngine, "pylaia": PyLaiaEngine}


def main():
    """Zero-shot line-based HTR over crops: blla segmentation, then Kraken or PyLaia."""
    parser = argparse.ArgumentParser(
        description="Apply Kraken or PyLaia HTR to multi-line crops."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/ocr/20260807-ft-data/test/images",
        help="Root directory containing crops (category subfolders).",
    )
    parser.add_argument(
        "--pattern", type=str, default="*/*.jpg", help="Glob pattern to find images"
    )
    parser.add_argument(
        "-n", type=int, default=None, help="Limit number of images to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write .md predictions (preserves subfolder structure).",
    )
    parser.add_argument("--engine", type=str, default="kraken", choices=sorted(ENGINES))
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Kraken .mlmodel path, or PyLaia model directory.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Kraken OpenMP/BLAS threads (default: all cores).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    settings = HTRSettings(
        engine=args.engine,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        **({"threads": args.threads} if args.threads else {}),
    )
    cache_dir = Path(settings.seg_cache_dir or input_dir.parent / "segmentation")

    images = find_images(input_dir, args.pattern, limit=args.n)
    logger.info(f"Found {len(images)} images; engine={args.engine} model={args.model}")

    engine = ENGINES[args.engine](settings)
    records = engine.transcribe(images, input_dir, output_dir, cache_dir)

    failed = [r["image"] for r in records if r["failed"]]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "engine": args.engine,
        "model": args.model,
        "device": args.device,
        "n_images": len(records),
        "mean_lines_per_crop": (
            round(mean(r["n_lines"] for r in records), 2) if records else 0
        ),
        **engine.timings,
        "failed_count": len(failed),
        "failed": failed,
        "lines_per_crop": {r["image"]: r["n_lines"] for r in records},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Wrote {len(records)} predictions to {output_dir} "
        f"({len(failed)} failed); timings {engine.timings}"
    )


if __name__ == "__main__":
    main()
