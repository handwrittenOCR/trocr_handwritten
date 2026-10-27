import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import jiwer

from trocr_handwritten.llm.settings import MetricsSettings
from trocr_handwritten.utils.annotation import register_key
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

# Normalization ladder (paper 2). One frozen transform per level, applied
# identically to hypothesis and reference. Keys are the paper codes (see
# MetricsSettings.levels for the legend); N0 and N2 are the co-primaries, N1/N3
# live in appendices. Delta_conv = N0 - N2; Delta_read = N2.
_MARKUP_STRIKE = re.compile(r"~~(.*?)~~")
_MARKUP_ELLIPSIS = re.compile(r"[.·]{3,}")
_APOSTROPHE = re.compile(r"[’‘ʼ′`´]")
_HYPHEN = re.compile(r"[‐-―−]")
_WHITESPACE = re.compile(r"[ \t ]+")


def _strip_markup(text):
    """Remove model markdown artefacts (strikethrough, ellipsis runs)."""
    text = _MARKUP_STRIKE.sub(r"\1", text)
    return _MARKUP_ELLIPSIS.sub("", text)


def _collapse_whitespace(text):
    """Collapse intra-line whitespace and drop blank lines."""
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def norm_n0(text):
    """N0 raw: markup stripped, whitespace collapsed; case/punct/accents kept."""
    return _collapse_whitespace(_strip_markup(text))


def norm_n1(text):
    """N1: N0 plus case folding (isolates the case convention)."""
    return norm_n0(text).lower()


def _strip_punctuation(text):
    """Remove Unicode punctuation, keeping hyphens and apostrophes (normalized)."""
    text = _APOSTROPHE.sub("'", text)
    text = _HYPHEN.sub("-", text)
    kept = []
    for char in text:
        if char in "-'\n" or not unicodedata.category(char).startswith("P"):
            kept.append(char)
    return "".join(kept)


def norm_n2(text):
    """N2 canonical: N0 plus punctuation strip; case and accents preserved."""
    return _collapse_whitespace(_strip_punctuation(norm_n0(text)))


def norm_n3(text):
    """N3 ascii: N2 plus diacritics stripped (accents are payload; appendix only)."""
    decomposed = unicodedata.normalize("NFKD", norm_n2(text))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


NORMALIZERS = {"N0": norm_n0, "N1": norm_n1, "N2": norm_n2, "N3": norm_n3}


def _collect_pairs(predictions_dir, labels_dir, pred_ext=".md", label_ext=".txt"):
    """
    Match prediction files with label files by (subfolder, stem), recursively.

    Failures are kept, not dropped: an empty prediction or a wildly long output
    is flagged and scored as CER=1.0 (per-item clip), so failure counts are honest.

    Args:
        predictions_dir: Directory with prediction files (e.g. .md)
        labels_dir: Directory with reference label files (e.g. .txt)
        pred_ext: Extension of prediction files
        label_ext: Extension of label files

    Returns:
        list: (subfolder, stem, prediction, reference) tuples.
    """
    pred_path = Path(predictions_dir)
    label_path = Path(labels_dir)

    label_map = {}
    for f in label_path.rglob(f"*{label_ext}"):
        rel = f.relative_to(label_path)
        subfolder = str(rel.parent) if rel.parent != Path(".") else ""
        label_map[(subfolder, f.stem)] = f

    pairs = []
    seen = set()
    for f in sorted(pred_path.rglob(f"*{pred_ext}")):
        rel = f.relative_to(pred_path)
        subfolder = str(rel.parent) if rel.parent != Path(".") else ""
        key = (subfolder, f.stem)
        if key not in label_map:
            continue
        seen.add(key)
        reference = label_map[key].read_text(encoding="utf-8").strip()
        if not reference:
            continue
        prediction = f.read_text(encoding="utf-8").strip()
        pairs.append((subfolder, f.stem, prediction, reference))
    for key in sorted(set(label_map) - seen):
        reference = label_map[key].read_text(encoding="utf-8").strip()
        if reference:
            pairs.append((key[0], key[1], "", reference))
    return pairs


def _edits(reference, hypothesis, unit):
    """Return (edit_count, reference_length) for characters or words."""
    if unit == "char":
        measures = jiwer.process_characters(reference, hypothesis)
        ref_len = len(reference)
    else:
        measures = jiwer.process_words(reference, hypothesis)
        ref_len = len(reference.split())
    edits = measures.substitutions + measures.insertions + measures.deletions
    return edits, ref_len


def _score_item(subfolder, stem, prediction, reference, outlier_ratio):
    """
    Score one pair across the whole normalization ladder.

    Returns a dict with the register, a `failure` flag and, per level, the
    reference length and clipped edit counts for characters and words (enabling
    both micro and macro aggregation, and register-cluster bootstrap, downstream).
    """
    raw_ref = norm_n0(reference)
    raw_hyp = norm_n0(prediction)
    ratio = len(raw_hyp) / max(len(raw_ref), 1)
    failure = (not raw_hyp) or ratio > outlier_ratio

    levels = {}
    for code, normalize in NORMALIZERS.items():
        ref = normalize(reference)
        hyp = normalize(prediction)
        char_edits, ref_chars = _edits(ref, hyp, "char")
        word_edits, ref_words = _edits(ref, hyp, "word")
        char_edits = min(char_edits, ref_chars)
        word_edits = min(word_edits, ref_words)
        levels[code] = {
            "ref_chars": ref_chars,
            "char_edits": char_edits,
            "cer": (char_edits / ref_chars) if ref_chars else 0.0,
            "ref_words": ref_words,
            "word_edits": word_edits,
            "wer": (word_edits / ref_words) if ref_words else 0.0,
        }
    return {
        "register": register_key(stem),
        "subfolder": subfolder,
        "stem": stem,
        "failure": failure,
        "exact_match": 1.0 if raw_hyp == raw_ref else 0.0,
        "levels": levels,
    }


def _aggregate(items, level):
    """Micro (char-weighted) and macro (per-item mean) CER/WER for one level."""
    cers = [i["levels"][level]["cer"] for i in items]
    wers = [i["levels"][level]["wer"] for i in items]
    char_edits = sum(i["levels"][level]["char_edits"] for i in items)
    ref_chars = sum(i["levels"][level]["ref_chars"] for i in items)
    word_edits = sum(i["levels"][level]["word_edits"] for i in items)
    ref_words = sum(i["levels"][level]["ref_words"] for i in items)
    return {
        "n": len(items),
        "ref_chars": ref_chars,
        "micro_cer": (char_edits / ref_chars) if ref_chars else 0.0,
        "macro_cer": mean(cers) if cers else 0.0,
        "cer_median": median(cers) if cers else 0.0,
        "cer_std": stdev(cers) if len(cers) > 1 else 0.0,
        "micro_wer": (word_edits / ref_words) if ref_words else 0.0,
        "macro_wer": mean(wers) if wers else 0.0,
    }


def _block(items):
    """Per-level aggregates plus failure count and exact-match share for a group."""
    out = {level: _aggregate(items, level) for level in NORMALIZERS}
    out["failures"] = sum(i["failure"] for i in items)
    out["exact_pct"] = mean(i["exact_match"] for i in items) * 100 if items else 0.0
    return out


def _build_record(per_item, levels_legend):
    """Assemble the full metrics record (legend, per_item, aggregates) from items."""
    by_subfolder = defaultdict(list)
    by_register = defaultdict(list)
    for item in per_item:
        by_subfolder[item["subfolder"]].append(item)
        by_register[item["register"]].append(item)
    return {
        "levels": levels_legend,
        "per_item": per_item,
        "by_subfolder": {k: _block(v) for k, v in sorted(by_subfolder.items())},
        "by_register": {k: _block(v) for k, v in sorted(by_register.items())},
        "global": _block(per_item) if per_item else {},
    }


def compute_metrics(pairs, settings=None):
    """
    Compute the full metrics record from prediction/reference pairs.

    Args:
        pairs: Output of `_collect_pairs`.
        settings: MetricsSettings; defaults are used when None.

    Returns:
        dict: levels legend, per_item scores, and per-subfolder / per-register /
        global aggregates broken down by normalization level.
    """
    settings = settings or MetricsSettings()
    per_item = [_score_item(*p, settings.outlier_length_ratio) for p in pairs]
    return _build_record(per_item, settings.levels)


def _average_item(group):
    """Average edit counts of one item across runs (references are identical)."""
    first = group[0]
    levels = {}
    for code, ref_lv in first["levels"].items():
        ref_chars = ref_lv["ref_chars"]
        ref_words = ref_lv["ref_words"]
        char_edits = mean(g["levels"][code]["char_edits"] for g in group)
        word_edits = mean(g["levels"][code]["word_edits"] for g in group)
        levels[code] = {
            "ref_chars": ref_chars,
            "char_edits": char_edits,
            "cer": (char_edits / ref_chars) if ref_chars else 0.0,
            "ref_words": ref_words,
            "word_edits": word_edits,
            "wer": (word_edits / ref_words) if ref_words else 0.0,
        }
    return {
        "register": first["register"],
        "subfolder": first["subfolder"],
        "stem": first["stem"],
        "failure": mean(g["failure"] for g in group),
        "exact_match": mean(g["exact_match"] for g in group),
        "levels": levels,
        "n_runs": len(group),
    }


def _run_variability(records):
    """
    Inter-run mean and std of the headline metrics, global and per subfolder.

    Complements the register-cluster bootstrap: this is the spread *between* runs
    (optimization/decoding variance), computed from each run's point estimate.
    """

    def summarize(blocks):
        out = {}
        for level in NORMALIZERS:
            out[level] = {}
            for metric in ("micro_cer", "macro_cer", "micro_wer"):
                vals = [b[level][metric] for b in blocks]
                out[level][f"{metric}_mean"] = mean(vals)
                out[level][f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        return out

    subfolders = sorted(set().union(*[set(r["by_subfolder"]) for r in records]))
    return {
        "n_runs": len(records),
        "global": summarize([r["global"] for r in records if r.get("global")]),
        "by_subfolder": {
            s: summarize(
                [r["by_subfolder"][s] for r in records if s in r["by_subfolder"]]
            )
            for s in subfolders
        },
    }


def average_records(records):
    """
    Average several run records item-by-item, then rebuild the aggregates.

    Each item's per-level edit counts are averaged over the runs (references, hence
    reference lengths, are identical across runs). The result is a metrics record of
    the same schema, so `llm.stats` bootstraps it directly: run/decoding noise is
    averaged out per item while the register-cluster CI still reflects test-sampling
    variance. A `runs` section carries the inter-run mean/std of the headline
    metrics (global and per subfolder).
    """
    groups = defaultdict(list)
    for record in records:
        for item in record["per_item"]:
            groups[(item["subfolder"], item["stem"])].append(item)
    per_item = [_average_item(g) for g in groups.values()]
    record = _build_record(per_item, records[0]["levels"])
    record["runs"] = _run_variability(records)
    return record


def log_metrics(results, logger):
    """Log a compact micro-CER table across the co-primary levels."""
    header = (
        f"{'Subfolder':>15}  {'micro N0':>8} {'micro N2':>8} "
        f"{'macro N2':>8} {'WER N2':>7} {'fail':>5} {'N':>5}"
    )
    logger.info(header)
    logger.info("-" * 62)

    def row(name, block):
        logger.info(
            f"{name:>15}  {block['N0']['micro_cer']:>8.4f} "
            f"{block['N2']['micro_cer']:>8.4f} {block['N2']['macro_cer']:>8.4f} "
            f"{block['N2']['micro_wer']:>7.4f} {block['failures']:>5} "
            f"{block['N0']['n']:>5}"
        )

    for subfolder, block in results["by_subfolder"].items():
        row(subfolder if subfolder else "(root)", block)

    if results["global"]:
        logger.info("-" * 62)
        row("ALL", results["global"])

    runs = results.get("runs")
    if runs:
        g = runs["global"]["N2"]
        logger.info(
            f"inter-run micro N2: mean {g['micro_cer_mean']:.4f} "
            f"± {g['micro_cer_std']:.4f} (n_runs={runs['n_runs']})"
        )


def main():
    """CLI entry point for OCR metrics."""
    parser = argparse.ArgumentParser(
        description="Compute OCR metrics (micro/macro CER-WER over the N0-N3 ladder)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        nargs="+",
        required=True,
        help="One or more prediction directories. With several (e.g. run1 run2 ...) "
        "each is scored to its own metrics.json and an averaged record is written "
        "to --output.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory with reference label files (e.g. .txt)",
    )
    settings = MetricsSettings()
    parser.add_argument("--pred-ext", type=str, default=settings.pred_ext)
    parser.add_argument("--label-ext", type=str, default=settings.label_ext)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save metrics JSON (default: metrics.json in predictions dir)",
    )
    args = parser.parse_args()

    records = []
    for pred_dir in args.predictions:
        pairs = _collect_pairs(pred_dir, args.labels, args.pred_ext, args.label_ext)
        if not pairs:
            logger.warning(f"No matching pairs in {pred_dir}; skipping")
            continue
        record = compute_metrics(pairs, settings)
        with open(Path(pred_dir) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        records.append(record)
        logger.info(f"{pred_dir}: {len(pairs)} pairs")

    if not records:
        logger.error("No matching prediction/label pairs found")
        return

    if len(records) == 1:
        results = records[0]
        default_output = Path(args.predictions[0]) / "metrics.json"
    else:
        results = average_records(records)
        logger.info(f"Averaged {len(records)} runs")
        if not args.output:
            logger.error("--output is required when averaging several runs")
            return
        default_output = Path(args.output)

    log_metrics(results, logger)
    output_path = Path(args.output) if args.output else default_output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
