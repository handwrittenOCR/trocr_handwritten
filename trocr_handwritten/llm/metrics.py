import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import evaluate
import jiwer

from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def _collect_pairs(predictions_dir, labels_dir, pred_ext=".md", label_ext=".txt"):
    """
    Match prediction files with label files by stem name, recursively.
    Preserves subfolder structure for per-class metrics.

    Args:
        predictions_dir: Directory with prediction files (e.g. .md)
        labels_dir: Directory with reference label files (e.g. .txt)
        pred_ext: Extension of prediction files
        label_ext: Extension of label files

    Returns:
        list: List of (subfolder, stem, prediction_text, reference_text) tuples
    """
    pred_path = Path(predictions_dir)
    label_path = Path(labels_dir)

    label_map = {}
    for f in label_path.rglob(f"*{label_ext}"):
        rel = f.relative_to(label_path)
        subfolder = str(rel.parent) if rel.parent != Path(".") else ""
        label_map[(subfolder, f.stem)] = f

    pairs = []
    for f in sorted(pred_path.rglob(f"*{pred_ext}")):
        rel = f.relative_to(pred_path)
        subfolder = str(rel.parent) if rel.parent != Path(".") else ""
        key = (subfolder, f.stem)
        if key in label_map:
            pred_text = f.read_text(encoding="utf-8").strip()
            ref_text = label_map[key].read_text(encoding="utf-8").strip()
            if pred_text and ref_text:
                pairs.append((subfolder, f.stem, pred_text, ref_text))

    return pairs


def _levenshtein_distance(ref, hyp):
    """
    Compute character-level Levenshtein distance.

    Args:
        ref: Reference string
        hyp: Hypothesis string

    Returns:
        int: Edit distance
    """
    measures = jiwer.process_characters(ref, hyp)
    return measures.substitutions + measures.insertions + measures.deletions


def compute_metrics(pairs):
    """
    Compute OCR metrics from prediction/reference pairs.

    Args:
        pairs: List of (subfolder, stem, prediction_text, reference_text)

    Returns:
        dict: Per-subfolder and global metrics
    """
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    per_item = []
    by_subfolder = defaultdict(list)

    for subfolder, stem, prediction, reference in pairs:
        item_cer = cer_metric.compute(predictions=[prediction], references=[reference])
        item_wer = wer_metric.compute(predictions=[prediction], references=[reference])
        item_lev = _levenshtein_distance(reference, prediction)
        item_exact = 1.0 if prediction == reference else 0.0

        item = {
            "filename": stem,
            "subfolder": subfolder,
            "cer": item_cer,
            "wer": item_wer,
            "levenshtein": item_lev,
            "exact_match": item_exact,
        }
        per_item.append(item)
        by_subfolder[subfolder].append(item)

    results = {"per_item": per_item, "by_subfolder": {}, "global": {}}

    for subfolder, items in sorted(by_subfolder.items()):
        cers = [i["cer"] for i in items]
        wers = [i["wer"] for i in items]
        levs = [i["levenshtein"] for i in items]
        exacts = [i["exact_match"] for i in items]

        results["by_subfolder"][subfolder] = {
            "n": len(items),
            "cer_mean": mean(cers),
            "cer_median": median(cers),
            "cer_std": stdev(cers) if len(cers) > 1 else 0.0,
            "wer_mean": mean(wers),
            "wer_median": median(wers),
            "lev_mean": mean(levs),
            "exact_pct": mean(exacts) * 100,
        }

    if per_item:
        all_cers = [i["cer"] for i in per_item]
        all_wers = [i["wer"] for i in per_item]
        all_levs = [i["levenshtein"] for i in per_item]
        all_exacts = [i["exact_match"] for i in per_item]

        results["global"] = {
            "n": len(per_item),
            "cer_mean": mean(all_cers),
            "cer_median": median(all_cers),
            "cer_std": stdev(all_cers) if len(all_cers) > 1 else 0.0,
            "wer_mean": mean(all_wers),
            "wer_median": median(all_wers),
            "lev_mean": mean(all_levs),
            "exact_pct": mean(all_exacts) * 100,
        }

    return results


def log_metrics(results, logger):
    """
    Log metrics table to logger.

    Args:
        results: Output from compute_metrics
        logger: Logger instance
    """
    header = (
        f"{'Subfolder':>15}  {'CER':>7} {'WER':>7} {'Lev':>7} {'Exact%':>7} {'N':>5}"
    )
    logger.info(header)
    logger.info("-" * 55)

    for subfolder, m in results["by_subfolder"].items():
        name = subfolder if subfolder else "(root)"
        logger.info(
            f"{name:>15}  {m['cer_mean']:>7.4f} {m['wer_mean']:>7.4f} "
            f"{m['lev_mean']:>7.1f} {m['exact_pct']:>6.1f}% {m['n']:>5}"
        )

    if results["global"]:
        g = results["global"]
        logger.info("-" * 55)
        logger.info(
            f"{'ALL':>15}  {g['cer_mean']:>7.4f} {g['wer_mean']:>7.4f} "
            f"{g['lev_mean']:>7.1f} {g['exact_pct']:>6.1f}% {g['n']:>5}"
        )


def main():
    """CLI entry point for OCR metrics."""
    parser = argparse.ArgumentParser(
        description="Compute OCR metrics: predictions vs labels"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory with prediction files (e.g. data/ocr/test/images with .md files)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory with reference label files (e.g. data/ocr/test/labels with .txt files)",
    )
    parser.add_argument("--pred-ext", type=str, default=".md")
    parser.add_argument("--label-ext", type=str, default=".txt")
    args = parser.parse_args()

    pairs = _collect_pairs(args.predictions, args.labels, args.pred_ext, args.label_ext)
    if not pairs:
        logger.error("No matching prediction/label pairs found")
        return

    logger.info(f"Found {len(pairs)} prediction/label pairs")
    results = compute_metrics(pairs)
    log_metrics(results, logger)


if __name__ == "__main__":
    main()
