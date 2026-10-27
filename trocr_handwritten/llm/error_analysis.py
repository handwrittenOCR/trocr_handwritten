import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import evaluate
import jiwer

from trocr_handwritten.llm.metrics import NORMALIZERS, _collect_pairs
from trocr_handwritten.utils.annotation import register_key
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

FORMULAIC_PATTERNS = [
    "fils de",
    "née? [àa]",
    "arrondissement d",
    "département d",
    "condamné[e]? le",
    "profession de",
    "domicilié[e]? [àa]",
    "ayant exercé",
    "Le nommé",
    "Taille d'un mètre",
    "Cheveux",
    "Sourcils",
    "Front",
    "Yeux",
    "Nez",
    "Bouche",
    "Menton",
    "Barbe",
    "Visage",
    "Teint",
    "Signes particuliers",
    "moyens d'existence",
]

ENTITY_PATTERNS = {
    "numbers": r"\b\d+\b",
    "dates": r"\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b",
    "proper_nouns": r"\b[A-ZÀ-Ü][a-zà-ü]{2,}\b",
}


def character_confusion_matrix(pairs, top_n=30):
    """
    Build a character-level confusion matrix from substitution errors.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples
        top_n: Number of top confusions to return

    Returns:
        dict with 'substitutions', 'insertions', 'deletions' counters
    """
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()

    for _, _, prediction, reference in pairs:
        out = jiwer.process_characters([reference], [prediction])
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                ref_chars = reference[chunk.ref_start_idx : chunk.ref_end_idx]
                hyp_chars = prediction[chunk.hyp_start_idx : chunk.hyp_end_idx]
                for rc, hc in zip(ref_chars, hyp_chars):
                    substitutions[(rc, hc)] += 1
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    deletions[reference[i]] += 1
            elif chunk.type == "insert":
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    insertions[prediction[i]] += 1

    return {
        "substitutions": substitutions.most_common(top_n),
        "insertions": insertions.most_common(top_n),
        "deletions": deletions.most_common(top_n),
    }


def entity_cer(pairs):
    """
    Compute CER on entity segments (numbers, dates, proper nouns) vs rest.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples

    Returns:
        dict with per-entity-type CER and baseline CER
    """
    cer_metric = evaluate.load("cer")
    entity_hits = defaultdict(lambda: {"refs": [], "preds": []})
    all_refs = []
    all_preds = []

    for _, _, prediction, reference in pairs:
        all_refs.append(reference)
        all_preds.append(prediction)

        out = jiwer.process_words([reference], [prediction])
        ref_words = reference.split()
        pred_words = prediction.split()

        ref_to_pred = {}
        for chunk in out.alignments[0]:
            if chunk.type == "equal":
                for ri, pi in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                ):
                    ref_to_pred[ri] = pred_words[pi]
            elif chunk.type == "substitute":
                for ri, pi in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                ):
                    ref_to_pred[ri] = pred_words[pi]

        for entity_type, pattern in ENTITY_PATTERNS.items():
            for match in re.finditer(pattern, reference, re.IGNORECASE):
                start_char = match.start()
                word_idx = len(reference[:start_char].split()) - (
                    1 if start_char > 0 and reference[start_char - 1] != " " else 0
                )
                match_words = match.group().split()
                ref_segment = []
                pred_segment = []
                for offset in range(len(match_words)):
                    idx = word_idx + offset
                    if idx < len(ref_words):
                        ref_segment.append(ref_words[idx])
                        if idx in ref_to_pred:
                            pred_segment.append(ref_to_pred[idx])
                        else:
                            pred_segment.append("")

                if ref_segment:
                    entity_hits[entity_type]["refs"].append(" ".join(ref_segment))
                    entity_hits[entity_type]["preds"].append(" ".join(pred_segment))

    results = {}
    baseline_cer = cer_metric.compute(predictions=all_preds, references=all_refs)
    results["baseline_cer"] = baseline_cer

    for entity_type, data in entity_hits.items():
        if data["refs"]:
            entity_cer_val = cer_metric.compute(
                predictions=data["preds"], references=data["refs"]
            )
            results[entity_type] = {
                "cer": entity_cer_val,
                "n_segments": len(data["refs"]),
            }

    return results


def formulaic_vs_free_cer(pairs):
    """
    Compare CER on formulaic expressions vs free text segments.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples

    Returns:
        dict with formulaic CER, free CER, and per-formula stats
    """
    cer_metric = evaluate.load("cer")
    formula_refs = []
    formula_preds = []
    free_refs = []
    free_preds = []

    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in FORMULAIC_PATTERNS]

    for _, _, prediction, reference in pairs:
        ref_lines = reference.split("\n")

        out = jiwer.process_words([reference], [prediction])
        pred_words = prediction.split()

        ref_to_pred_map = {}
        for chunk in out.alignments[0]:
            if chunk.type in ("equal", "substitute"):
                for ri, pi in zip(
                    range(chunk.ref_start_idx, chunk.ref_end_idx),
                    range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                ):
                    ref_to_pred_map[ri] = pi

        for ref_line in ref_lines:
            if not ref_line.strip():
                continue
            is_formulaic = any(rx.search(ref_line) for _, rx in compiled)

            start_word = (
                len(reference.split(ref_line)[0].split())
                if ref_line in reference
                else 0
            )
            line_word_count = len(ref_line.split())

            matched_pred_words = []
            for offset in range(line_word_count):
                idx = start_word + offset
                if idx in ref_to_pred_map:
                    matched_pred_words.append(pred_words[ref_to_pred_map[idx]])
                else:
                    matched_pred_words.append("")

            pred_line = " ".join(matched_pred_words) if matched_pred_words else ""

            if ref_line.strip():
                if is_formulaic:
                    formula_refs.append(ref_line.strip())
                    formula_preds.append(pred_line.strip())
                else:
                    free_refs.append(ref_line.strip())
                    free_preds.append(pred_line.strip())

    results = {}
    if formula_refs:
        results["formulaic"] = {
            "cer": cer_metric.compute(
                predictions=formula_preds, references=formula_refs
            ),
            "n": len(formula_refs),
        }
    if free_refs:
        results["free_text"] = {
            "cer": cer_metric.compute(predictions=free_preds, references=free_refs),
            "n": len(free_refs),
        }

    return results


def space_apostrophe_errors(pairs, top_n=20):
    """
    Detect space and apostrophe insertion/deletion errors.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples
        top_n: Number of top examples to return

    Returns:
        dict with space merges, splits, apostrophe errors and examples
    """
    space_insertions = []
    space_deletions = []
    apostrophe_errors = []

    for _, stem, prediction, reference in pairs:
        out = jiwer.process_characters([reference], [prediction])
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                ref_seg = reference[chunk.ref_start_idx : chunk.ref_end_idx]
                hyp_seg = prediction[chunk.hyp_start_idx : chunk.hyp_end_idx]
                if " " in ref_seg and " " not in hyp_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 5)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 5)
                    space_deletions.append(
                        (stem, reference[ctx_start:ctx_end], ref_seg, hyp_seg)
                    )
                elif " " not in ref_seg and " " in hyp_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 5)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 5)
                    space_insertions.append(
                        (stem, reference[ctx_start:ctx_end], ref_seg, hyp_seg)
                    )
                if "'" in ref_seg or "'" in hyp_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 8)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 8)
                    apostrophe_errors.append(
                        (stem, reference[ctx_start:ctx_end], ref_seg, hyp_seg)
                    )

            elif chunk.type == "delete":
                ref_seg = reference[chunk.ref_start_idx : chunk.ref_end_idx]
                if " " in ref_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 5)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 5)
                    space_deletions.append(
                        (stem, reference[ctx_start:ctx_end], ref_seg, "∅")
                    )
                if "'" in ref_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 8)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 8)
                    apostrophe_errors.append(
                        (stem, reference[ctx_start:ctx_end], ref_seg, "∅")
                    )

            elif chunk.type == "insert":
                hyp_seg = prediction[chunk.hyp_start_idx : chunk.hyp_end_idx]
                if " " in hyp_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 5)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 5)
                    space_insertions.append(
                        (stem, reference[ctx_start:ctx_end], "∅", hyp_seg)
                    )
                if "'" in hyp_seg:
                    ctx_start = max(0, chunk.ref_start_idx - 8)
                    ctx_end = min(len(reference), chunk.ref_end_idx + 8)
                    apostrophe_errors.append(
                        (stem, reference[ctx_start:ctx_end], "∅", hyp_seg)
                    )

    return {
        "space_merges": space_deletions[:top_n],
        "space_splits": space_insertions[:top_n],
        "apostrophe_errors": apostrophe_errors[:top_n],
        "total_space_merges": len(space_deletions),
        "total_space_splits": len(space_insertions),
        "total_apostrophe_errors": len(apostrophe_errors),
    }


def truncation_errors(pairs, tail_chars=10):
    """
    Measure CER on the last N characters of each line vs the rest.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples
        tail_chars: Number of characters at end of line to analyze

    Returns:
        dict with tail CER vs head CER
    """
    cer_metric = evaluate.load("cer")
    tail_refs = []
    tail_preds = []
    head_refs = []
    head_preds = []

    for _, _, prediction, reference in pairs:
        ref_lines = reference.split("\n")
        pred_lines = prediction.split("\n")

        for i, ref_line in enumerate(ref_lines):
            ref_line = ref_line.strip()
            if len(ref_line) < tail_chars + 5:
                continue

            pred_line = pred_lines[i].strip() if i < len(pred_lines) else ""

            ref_tail = ref_line[-tail_chars:]
            ref_head = ref_line[:-tail_chars]
            pred_tail = (
                pred_line[-tail_chars:] if len(pred_line) >= tail_chars else pred_line
            )
            pred_head = pred_line[:-tail_chars] if len(pred_line) >= tail_chars else ""

            if ref_tail:
                tail_refs.append(ref_tail)
                tail_preds.append(pred_tail)
            if ref_head:
                head_refs.append(ref_head)
                head_preds.append(pred_head)

    results = {}
    if tail_refs:
        results["tail_cer"] = cer_metric.compute(
            predictions=tail_preds, references=tail_refs
        )
        results["tail_n"] = len(tail_refs)
    if head_refs:
        results["head_cer"] = cer_metric.compute(
            predictions=head_preds, references=head_refs
        )
        results["head_n"] = len(head_refs)

    return results


def length_type_buckets(pairs):
    """
    Compute CER by word-length bucket crossed with subfolder type.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples

    Returns:
        dict with CER per (bucket, subfolder) combination
    """
    cer_metric = evaluate.load("cer")

    buckets_def = [
        ("1-3", 1, 3),
        ("4-10", 4, 10),
        ("11-25", 11, 25),
        ("26-50", 26, 50),
        ("51+", 51, 9999),
    ]

    grouped = defaultdict(lambda: {"refs": [], "preds": []})

    for subfolder, _, prediction, reference in pairs:
        word_count = len(reference.split())
        bucket = "51+"
        for name, lo, hi in buckets_def:
            if lo <= word_count <= hi:
                bucket = name
                break

        grouped[(bucket, subfolder)]["refs"].append(reference)
        grouped[(bucket, subfolder)]["preds"].append(prediction)
        grouped[(bucket, "ALL")]["refs"].append(reference)
        grouped[("ALL", subfolder)]["refs"].append(reference)
        grouped[("ALL", subfolder)]["preds"].append(prediction)
        grouped[(bucket, "ALL")]["preds"].append(prediction)

    results = {}
    for key, data in sorted(grouped.items()):
        if data["refs"]:
            results[key] = {
                "cer": cer_metric.compute(
                    predictions=data["preds"], references=data["refs"]
                ),
                "n": len(data["refs"]),
            }

    return results


def hallucination_detection(pairs, min_consecutive=3, top_n=20):
    """
    Detect long inserted sequences (words present in prediction but absent from reference).

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples
        min_consecutive: Minimum consecutive inserted words to flag
        top_n: Number of top hallucinations to return

    Returns:
        dict with hallucination examples and counts
    """
    hallucinations = []

    for subfolder, stem, prediction, reference in pairs:
        out = jiwer.process_words([reference], [prediction])
        pred_words = prediction.split()

        for chunk in out.alignments[0]:
            if chunk.type == "insert":
                inserted = pred_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
                if len(inserted) >= min_consecutive:
                    hallucinations.append(
                        {
                            "file": stem,
                            "subfolder": subfolder,
                            "inserted_text": " ".join(inserted),
                            "n_words": len(inserted),
                        }
                    )

    hallucinations.sort(key=lambda x: x["n_words"], reverse=True)

    return {
        "total_hallucinations": len(hallucinations),
        "total_hallucinated_words": sum(h["n_words"] for h in hallucinations),
        "examples": hallucinations[:top_n],
    }


def log_character_confusion(results, logger):
    """Log character confusion matrix results."""
    logger.info("\n=== CHARACTER CONFUSION MATRIX ===")

    logger.info("\nTop substitutions (ref → pred):")
    logger.info(f"  {'ref':>5} → {'pred':<5}  {'count':>6}")
    logger.info("  " + "-" * 25)
    for (ref_c, hyp_c), count in results["substitutions"]:
        ref_repr = repr(ref_c) if ref_c in (" ", "\n") else ref_c
        hyp_repr = repr(hyp_c) if hyp_c in (" ", "\n") else hyp_c
        logger.info(f"  {ref_repr:>5} → {hyp_repr:<5}  {count:>6}")

    logger.info("\nTop deleted characters:")
    for char, count in results["deletions"][:15]:
        char_repr = repr(char) if char in (" ", "\n") else char
        logger.info(f"  {char_repr:>5}  {count:>6}")

    logger.info("\nTop inserted characters:")
    for char, count in results["insertions"][:15]:
        char_repr = repr(char) if char in (" ", "\n") else char
        logger.info(f"  {char_repr:>5}  {count:>6}")


def log_entity_cer(results, logger):
    """Log entity CER results."""
    logger.info("\n=== ENTITY CER ===")
    logger.info(f"  Baseline CER: {results['baseline_cer']:.4f}")
    for entity_type in ("numbers", "dates", "proper_nouns"):
        if entity_type in results:
            r = results[entity_type]
            logger.info(
                f"  {entity_type:>15}: CER={r['cer']:.4f}  (n={r['n_segments']})"
            )


def log_formulaic(results, logger):
    """Log formulaic vs free text results."""
    logger.info("\n=== FORMULAIC vs FREE TEXT ===")
    for key in ("formulaic", "free_text"):
        if key in results:
            r = results[key]
            logger.info(f"  {key:>12}: CER={r['cer']:.4f}  (n={r['n']})")


def log_space_apostrophe(results, logger):
    """Log space and apostrophe error results."""
    logger.info("\n=== SPACE & APOSTROPHE ERRORS ===")
    logger.info(f"  Space merges (word fusion):  {results['total_space_merges']}")
    logger.info(f"  Space splits (word break):   {results['total_space_splits']}")
    logger.info(f"  Apostrophe errors:           {results['total_apostrophe_errors']}")

    if results["space_merges"]:
        logger.info("\n  Sample merges:")
        for stem, ctx, ref, hyp in results["space_merges"][:10]:
            logger.info(f"    [{stem}] ...{ctx}...  '{ref}' → '{hyp}'")

    if results["apostrophe_errors"]:
        logger.info("\n  Sample apostrophe errors:")
        for stem, ctx, ref, hyp in results["apostrophe_errors"][:10]:
            logger.info(f"    [{stem}] ...{ctx}...  '{ref}' → '{hyp}'")


def log_truncation(results, logger):
    """Log truncation error results."""
    logger.info("\n=== TRUNCATION (last 10 chars vs rest) ===")
    if "head_cer" in results:
        logger.info(f"  Head CER: {results['head_cer']:.4f}  (n={results['head_n']})")
    if "tail_cer" in results:
        logger.info(f"  Tail CER: {results['tail_cer']:.4f}  (n={results['tail_n']})")


def log_length_type(results, logger):
    """Log length x type bucket results."""
    logger.info("\n=== CER by LENGTH × TYPE ===")

    subfolders = sorted({k[1] for k in results.keys()})
    buckets = ["1-3", "4-10", "11-25", "26-50", "51+", "ALL"]

    header = f"  {'bucket':>8}"
    for sf in subfolders:
        header += f"  {sf[:12]:>14}"
    logger.info(header)
    logger.info("  " + "-" * (10 + 16 * len(subfolders)))

    for bucket in buckets:
        row = f"  {bucket:>8}"
        for sf in subfolders:
            key = (bucket, sf)
            if key in results:
                r = results[key]
                row += f"  {r['cer']:>7.4f} ({r['n']:>3})"
            else:
                row += f"  {'—':>14}"
        logger.info(row)


def log_hallucinations(results, logger):
    """Log hallucination detection results."""
    logger.info("\n=== HALLUCINATIONS (inserted sequences ≥3 words) ===")
    logger.info(
        f"  Total: {results['total_hallucinations']} sequences, {results['total_hallucinated_words']} words"
    )

    if results["examples"]:
        logger.info("\n  Examples:")
        for h in results["examples"][:15]:
            logger.info(
                f"    [{h['subfolder']}/{h['file']}] ({h['n_words']} words): \"{h['inserted_text']}\""
            )


def _word_alignment(reference, hypothesis, level="N2"):
    """
    Align two texts word by word on the N2 canonical form.

    Returns (ref_tokens, flags, chunks) where flags[i] tells whether reference
    token i was mis-read, and chunks is the raw jiwer alignment. Sharing this one
    alignment keeps the positional, linguistic and word analyses consistent.
    """
    ref = NORMALIZERS[level](reference).split()
    hyp = NORMALIZERS[level](hypothesis).split()
    if not ref:
        return [], [], []
    out = jiwer.process_words([" ".join(ref)], [" ".join(hyp) or " "])
    flags = [False] * len(ref)
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if i < len(flags):
                    flags[i] = True
    return ref, flags, out.alignments[0]


def word_error_profile(pairs, top_n=25):
    """
    Break word errors into substitutions, insertions and deletions, and rank the
    most frequent confusions.

    The character confusion matrix says which glyphs slip; this says which words
    the model actually gets wrong, which is what a downstream lexicon would fix.
    """
    subs, dels, ins = Counter(), Counter(), Counter()
    totals = Counter()
    for _, _, prediction, reference in pairs:
        ref = NORMALIZERS["N2"](reference).split()
        hyp = NORMALIZERS["N2"](prediction).split()
        if not ref:
            continue
        out = jiwer.process_words([" ".join(ref)], [" ".join(hyp) or " "])
        totals["ref_words"] += len(ref)
        totals["hits"] += out.hits
        totals["substitutions"] += out.substitutions
        totals["insertions"] += out.insertions
        totals["deletions"] += out.deletions
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                for r, h in zip(
                    ref[chunk.ref_start_idx : chunk.ref_end_idx],
                    hyp[chunk.hyp_start_idx : chunk.hyp_end_idx],
                ):
                    subs[(r, h)] += 1
            elif chunk.type == "delete":
                for r in ref[chunk.ref_start_idx : chunk.ref_end_idx]:
                    dels[r] += 1
            elif chunk.type == "insert":
                for h in hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                    ins[h] += 1
    n = max(totals["ref_words"], 1)
    edits = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    return {
        "ref_words": totals["ref_words"],
        "wer": edits / n,
        "share": {
            k: totals[k] / max(edits, 1)
            for k in ("substitutions", "insertions", "deletions")
        },
        "counts": dict(totals),
        "top_substitutions": subs.most_common(top_n),
        "top_deletions": dels.most_common(top_n),
        "top_insertions": ins.most_common(top_n),
    }


def positional_error_profile(pairs, n_bins=10):
    """
    Measure the word error rate as a function of relative position in the text.

    Answers whether the model drifts towards the end (decoding fatigue, early
    EOS) or fumbles the start (header lines, layout entry point).
    """
    hit = [0] * n_bins
    total = [0] * n_bins
    first_last = Counter()
    for _, _, prediction, reference in pairs:
        ref, flags, _ = _word_alignment(reference, prediction)
        if not ref:
            continue
        for i, bad in enumerate(flags):
            b = min(int(i * n_bins / len(ref)), n_bins - 1)
            total[b] += 1
            hit[b] += bad
        first_last["first_5_err"] += sum(flags[:5])
        first_last["first_5_tot"] += len(flags[:5])
        first_last["last_5_err"] += sum(flags[-5:])
        first_last["last_5_tot"] += len(flags[-5:])
    return {
        "n_bins": n_bins,
        "bins": [
            {
                "bin": i,
                "n": total[i],
                "error_rate": hit[i] / total[i] if total[i] else 0.0,
            }
            for i in range(n_bins)
        ],
        "first_5": first_last["first_5_err"] / max(first_last["first_5_tot"], 1),
        "last_5": first_last["last_5_err"] / max(first_last["last_5_tot"], 1),
    }


def linguistic_error_profile(pairs, tags_path):
    """
    Break the word error rate down by part of speech and by named-entity type.

    Uses the LLM-tagged reference cache written by `llm.tag_references`, joined
    by token index: the tagger tokenises with the same N2 whitespace split as the
    alignment, so no realignment is needed.
    """
    tags = json.loads(Path(tags_path).read_text(encoding="utf-8"))
    pos_err, pos_tot = Counter(), Counter()
    ner_err, ner_tot = Counter(), Counter()
    matched, skipped = 0, 0
    for subfolder, stem, prediction, reference in pairs:
        entry = tags.get(f"{subfolder}/{stem}")
        ref, flags, _ = _word_alignment(reference, prediction)
        if entry is None or len(entry) != len(ref):
            skipped += 1
            continue
        matched += 1
        for (_, pos, ner), bad in zip(entry, flags):
            pos_tot[pos] += 1
            ner_tot[ner] += 1
            pos_err[pos] += bad
            ner_err[ner] += bad

    def _rows(err, tot):
        return sorted(
            (
                {"label": k, "n": tot[k], "error_rate": err[k] / tot[k]}
                for k in tot
                if tot[k]
            ),
            key=lambda r: -r["error_rate"],
        )

    overall = sum(pos_err.values()) / max(sum(pos_tot.values()), 1)
    return {
        "documents_matched": matched,
        "documents_skipped": skipped,
        "overall_word_error_rate": overall,
        "by_pos": _rows(pos_err, pos_tot),
        "by_ner": _rows(ner_err, ner_tot),
    }


def document_error_profile(pairs, top_n=15):
    """
    Aggregate errors per source document (register) rather than per crop.

    Two distinct questions: how unevenly the error mass is spread across
    documents, and how likely a given crop is to carry at least one word error.
    """
    by_reg = defaultdict(lambda: {"chars": 0, "edits": 0, "items": 0, "clean": 0})
    clean_items = 0
    for _, stem, prediction, reference in pairs:
        ref_text = NORMALIZERS["N2"](reference)
        hyp_text = NORMALIZERS["N2"](prediction)
        if not ref_text:
            continue
        reg = by_reg[register_key(stem)]
        edits = jiwer.cer(ref_text, hyp_text or " ") * len(ref_text)
        reg["chars"] += len(ref_text)
        reg["edits"] += edits
        reg["items"] += 1
        _, flags, _ = _word_alignment(reference, prediction)
        if not any(flags):
            reg["clean"] += 1
            clean_items += 1
    rows = sorted(
        (
            {
                "register": k,
                "n_items": v["items"],
                "cer": v["edits"] / v["chars"] if v["chars"] else 0.0,
                "clean_rate": v["clean"] / v["items"] if v["items"] else 0.0,
            }
            for k, v in by_reg.items()
        ),
        key=lambda r: -r["cer"],
    )
    cers = [r["cer"] for r in rows]
    total_edits = sum(v["edits"] for v in by_reg.values())
    worst = sorted(by_reg.values(), key=lambda v: -v["edits"])
    top_share = sum(v["edits"] for v in worst[: max(len(worst) // 5, 1)])
    n_items = sum(v["items"] for v in by_reg.values())
    return {
        "n_registers": len(rows),
        "n_items": n_items,
        "item_error_probability": 1 - clean_items / max(n_items, 1),
        "cer_spread": {
            "min": min(cers) if cers else 0.0,
            "median": sorted(cers)[len(cers) // 2] if cers else 0.0,
            "max": max(cers) if cers else 0.0,
        },
        "worst_quintile_edit_share": top_share / max(total_edits, 1),
        "worst": rows[:top_n],
        "best": rows[-top_n:][::-1],
    }


def image_quality_profile(pairs, images_dir, image_ext=".jpg", n_buckets=4):
    """
    Correlate per-crop CER with cheap image-quality proxies.

    Blur is the variance of the Laplacian, contrast the pixel standard deviation,
    exposure the mean level. Reported as quartile buckets plus a Spearman
    coefficient, which is rank-based and so immune to the heavy CER tail.
    """
    import cv2
    import numpy as np
    from scipy.stats import spearmanr

    root = Path(images_dir)
    rows = []
    for subfolder, stem, prediction, reference in pairs:
        path = root / subfolder / f"{stem}{image_ext}"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ref_text = NORMALIZERS["N2"](reference)
        if not ref_text:
            continue
        rows.append(
            {
                "cer": jiwer.cer(ref_text, NORMALIZERS["N2"](prediction) or " "),
                "blur": float(cv2.Laplacian(img, cv2.CV_64F).var()),
                "contrast": float(img.std()),
                "brightness": float(img.mean()),
                "pixels": float(img.size),
            }
        )
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "metrics": {}}
    cers = np.array([r["cer"] for r in rows])
    for key in ("blur", "contrast", "brightness", "pixels"):
        vals = np.array([r[key] for r in rows])
        order = np.argsort(vals)
        buckets = np.array_split(order, n_buckets)
        rho, p = spearmanr(vals, cers)
        out["metrics"][key] = {
            "spearman_rho": float(rho),
            "p_value": float(p),
            "buckets": [
                {
                    "bucket": i,
                    "n": len(b),
                    "value_median": float(np.median(vals[b])),
                    "cer_mean": float(cers[b].mean()),
                }
                for i, b in enumerate(buckets)
            ],
        }
    return out


def log_word_errors(results, logger):
    """Log the word-level edit breakdown and the most frequent confusions."""
    logger.info("\n=== WORD ERRORS ===")
    share = results["share"]
    logger.info(
        f"WER {results['wer']:.4f} on {results['ref_words']} reference words | "
        f"sub {share['substitutions']:.0%} / del {share['deletions']:.0%} / "
        f"ins {share['insertions']:.0%}"
    )
    logger.info("  top substitutions (ref -> hyp):")
    for (r, h), c in results["top_substitutions"][:15]:
        logger.info(f"    {c:>4}x  {r!r} -> {h!r}")
    logger.info("  top deletions:")
    for w, c in results["top_deletions"][:10]:
        logger.info(f"    {c:>4}x  {w!r}")
    logger.info("  top insertions:")
    for w, c in results["top_insertions"][:10]:
        logger.info(f"    {c:>4}x  {w!r}")


def log_positional(results, logger):
    """Log the word error rate across relative position deciles."""
    logger.info("\n=== ERROR BY POSITION ===")
    logger.info(f"{'decile':>8} {'n':>7} {'err rate':>9}")
    for b in results["bins"]:
        bar = "#" * int(b["error_rate"] * 100)
        logger.info(f"{b['bin']:>8} {b['n']:>7} {b['error_rate']:>9.4f}  {bar}")
    logger.info(
        f"first 5 words: {results['first_5']:.4f} | last 5 words: {results['last_5']:.4f}"
    )


def log_linguistic(results, logger):
    """Log the word error rate by POS tag and by entity type."""
    logger.info("\n=== ERROR BY POS / NER ===")
    logger.info(
        f"{results['documents_matched']} documents joined, "
        f"{results['documents_skipped']} skipped (token count mismatch) | "
        f"overall word error rate {results['overall_word_error_rate']:.4f}"
    )
    for key, title in (("by_pos", "POS"), ("by_ner", "NER")):
        logger.info(f"  {title:>4}  {'n':>7} {'err rate':>9}  ratio")
        for row in results[key]:
            ratio = row["error_rate"] / max(results["overall_word_error_rate"], 1e-9)
            logger.info(
                f"  {row['label']:>4}  {row['n']:>7} {row['error_rate']:>9.4f}  x{ratio:.2f}"
            )


def log_documents(results, logger):
    """Log the spread of CER across source documents."""
    logger.info("\n=== ERROR BY DOCUMENT ===")
    spread = results["cer_spread"]
    logger.info(
        f"{results['n_registers']} registers, {results['n_items']} crops | "
        f"CER min {spread['min']:.4f} / median {spread['median']:.4f} / "
        f"max {spread['max']:.4f}"
    )
    logger.info(
        f"P(a crop carries >=1 word error) = {results['item_error_probability']:.4f} | "
        f"worst quintile of registers holds {results['worst_quintile_edit_share']:.0%} "
        f"of all character edits"
    )
    logger.info("  worst registers:")
    for r in results["worst"][:10]:
        logger.info(
            f"    {r['register']:>28}  n={r['n_items']:>3} cer={r['cer']:.4f} "
            f"clean={r['clean_rate']:.2f}"
        )


def log_image_quality(results, logger):
    """Log the correlation between image-quality proxies and CER."""
    logger.info("\n=== IMAGE QUALITY vs CER ===")
    if not results.get("n"):
        logger.info("  no images matched")
        return
    logger.info(f"{results['n']} crops measured")
    for key, block in results["metrics"].items():
        logger.info(
            f"  {key:>10}  spearman rho={block['spearman_rho']:+.3f} "
            f"p={block['p_value']:.4f}"
        )
        for b in block["buckets"]:
            logger.info(
                f"      Q{b['bucket'] + 1} n={b['n']:>3} "
                f"median={b['value_median']:>10.1f}  CER={b['cer_mean']:.4f}"
            )


def run_all(pairs, tags_path=None, images_dir=None):
    """
    Run all error analyses on prediction/reference pairs.

    Args:
        pairs: List of (subfolder, stem, prediction, reference) tuples

    Returns:
        dict with all analysis results
    """
    logger.info(f"Running error analysis on {len(pairs)} pairs...")

    results = {}

    logger.info("  1/7 Character confusion matrix...")
    results["char_confusion"] = character_confusion_matrix(pairs)
    log_character_confusion(results["char_confusion"], logger)

    logger.info("  2/7 Entity CER...")
    results["entity_cer"] = entity_cer(pairs)
    log_entity_cer(results["entity_cer"], logger)

    logger.info("  3/7 Formulaic vs free text...")
    results["formulaic"] = formulaic_vs_free_cer(pairs)
    log_formulaic(results["formulaic"], logger)

    logger.info("  4/7 Space & apostrophe errors...")
    results["space_apostrophe"] = space_apostrophe_errors(pairs)
    log_space_apostrophe(results["space_apostrophe"], logger)

    logger.info("  5/7 Truncation errors...")
    results["truncation"] = truncation_errors(pairs)
    log_truncation(results["truncation"], logger)

    logger.info("  6/7 Length × type buckets...")
    results["length_type"] = length_type_buckets(pairs)
    log_length_type(results["length_type"], logger)

    logger.info("  7/7 Hallucination detection...")
    results["hallucinations"] = hallucination_detection(pairs)
    log_hallucinations(results["hallucinations"], logger)

    logger.info("  + Word-level error breakdown...")
    results["word_errors"] = word_error_profile(pairs)
    log_word_errors(results["word_errors"], logger)

    logger.info("  + Error by position...")
    results["positional"] = positional_error_profile(pairs)
    log_positional(results["positional"], logger)

    logger.info("  + Error by document...")
    results["documents"] = document_error_profile(pairs)
    log_documents(results["documents"], logger)

    if tags_path:
        logger.info("  + Error by POS / NER...")
        results["linguistic"] = linguistic_error_profile(pairs, tags_path)
        log_linguistic(results["linguistic"], logger)

    if images_dir:
        logger.info("  + Image quality vs CER...")
        results["image_quality"] = image_quality_profile(pairs, images_dir)
        log_image_quality(results["image_quality"], logger)

    return results


def _jsonable(obj):
    """Recursively turn tuple dict keys into strings so results can be dumped."""
    if isinstance(obj, dict):
        return {
            (" → ".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main():
    """CLI entry point for OCR error analysis."""
    parser = argparse.ArgumentParser(
        description="Run detailed OCR error analysis: predictions vs labels"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Directory with prediction files",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory with reference label files",
    )
    parser.add_argument("--pred-ext", type=str, default=".md")
    parser.add_argument("--label-ext", type=str, default=".txt")
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="POS/NER cache from llm.tag_references; enables the linguistic split.",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Image directory mirroring the labels; enables the quality analysis.",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write the full results as JSON."
    )
    args = parser.parse_args()

    pairs = _collect_pairs(args.predictions, args.labels, args.pred_ext, args.label_ext)
    if not pairs:
        logger.error("No matching prediction/label pairs found")
        return

    results = run_all(pairs, tags_path=args.tags, images_dir=args.images)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_jsonable(results), ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"\nWrote {out}")


if __name__ == "__main__":
    main()
