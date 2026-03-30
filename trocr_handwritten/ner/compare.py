"""Compare regex vs LLM NER extraction results field-by-field."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from trocr_handwritten.ner.schemas import NERResult, flatten_ner_result

logger = logging.getLogger(__name__)


def compare_field(regex_val, llm_val) -> str:
    """Compare two field values. Returns 'match', 'mismatch', 'regex_only', 'llm_only', or 'both_null'."""
    r_none = regex_val is None or str(regex_val).strip() == ""
    l_none = llm_val is None or str(llm_val).strip() == ""

    if r_none and l_none:
        return "both_null"
    if r_none:
        return "llm_only"
    if l_none:
        return "regex_only"

    # Normalize for comparison: lowercase, strip whitespace
    r_norm = str(regex_val).strip().lower()
    l_norm = str(llm_val).strip().lower()

    if r_norm == l_norm:
        return "match"
    # Partial match: one contains the other
    if r_norm in l_norm or l_norm in r_norm:
        return "match"
    return "mismatch"


def compare_results(
    regex_results: List[NERResult],
    llm_results: List[NERResult],
) -> List[dict]:
    """Compare regex and LLM results field-by-field.

    Returns a list of comparison rows, one per (act_id, field) pair.
    """
    # Index by act_id
    llm_by_id = {r.act_id: r for r in llm_results}

    rows = []
    for regex_r in regex_results:
        llm_r = llm_by_id.get(regex_r.act_id)
        if not llm_r:
            continue

        regex_flat = flatten_ner_result(regex_r)
        llm_flat = flatten_ner_result(llm_r)

        # Compare all fields except raw text and metadata
        skip = {
            "act_id",
            "act_type",
            "extraction_method",
            "raw_marge",
            "raw_plein_texte",
        }
        all_fields = set(regex_flat.keys()) | set(llm_flat.keys())

        for field in sorted(all_fields - skip):
            r_val = regex_flat.get(field)
            l_val = llm_flat.get(field)
            status = compare_field(r_val, l_val)
            rows.append(
                {
                    "act_id": regex_r.act_id,
                    "act_type": regex_r.act_type,
                    "field": field,
                    "regex_value": r_val,
                    "llm_value": l_val,
                    "status": status,
                }
            )

    return rows


def compute_agreement(comparison_rows: List[dict]) -> Dict[str, dict]:
    """Compute agreement rates per field.

    Returns dict of field -> {match, mismatch, regex_only, llm_only, both_null, total, agreement_pct}.
    """
    stats: Dict[str, dict] = {}

    for row in comparison_rows:
        field = row["field"]
        status = row["status"]
        if field not in stats:
            stats[field] = {
                "match": 0,
                "mismatch": 0,
                "regex_only": 0,
                "llm_only": 0,
                "both_null": 0,
                "total": 0,
            }
        stats[field][status] += 1
        stats[field]["total"] += 1

    # Compute agreement percentage (match / (match + mismatch))
    for field, s in stats.items():
        contested = s["match"] + s["mismatch"]
        s["agreement_pct"] = (
            round(100 * s["match"] / contested, 1) if contested > 0 else None
        )

    return stats


def save_comparison(
    comparison_rows: List[dict],
    agreement: Dict[str, dict],
    output_dir: str,
) -> Tuple[Path, Path]:
    """Save comparison CSV and summary JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detailed comparison CSV
    csv_path = output_path / "ner_comparison.csv"
    if comparison_rows:
        fieldnames = [
            "act_id",
            "act_type",
            "field",
            "regex_value",
            "llm_value",
            "status",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

    # Summary JSON
    summary_path = output_path / "ner_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agreement, f, ensure_ascii=False, indent=2)

    logger.info("Saved comparison: %s (%d rows)", csv_path, len(comparison_rows))
    return csv_path, summary_path
