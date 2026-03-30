"""Merge regex and LLM NER results into a consolidated dataset.

Produces one CSV per commune/year with columns:
  act_id, act_type, act_number, {field}_regex, {field}_llm, {field}_final,
  marge_text, plein_texte_text

The _final column uses the LLM value if available, otherwise the regex value.

Usage:
    python -m trocr_handwritten.ner.merge \
        --ner_dir <path_to_ner_outputs> \
        --output_dir <path_to_NER_datasets> \
        --commune abymes --year 1842
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

from trocr_handwritten.ner.schemas import NERResult, flatten_ner_result

logger = logging.getLogger(__name__)

# Fields to merge (excluding metadata)
_SKIP_FIELDS = {
    "act_id",
    "act_type",
    "extraction_method",
    "raw_marge",
    "raw_plein_texte",
}


def merge_results(
    regex_results: List[NERResult],
    llm_results: List[NERResult],
) -> List[dict]:
    """Merge regex and LLM results into consolidated rows.

    Each row has: act_id, act_type, act_number, marge_text, plein_texte_text,
    then for each entity field: {field}_regex, {field}_llm, {field}_final.
    """
    llm_by_id = {r.act_id: r for r in llm_results}

    rows = []
    for regex_r in regex_results:
        llm_r = llm_by_id.get(regex_r.act_id)

        rf = flatten_ner_result(regex_r)
        lf = flatten_ner_result(llm_r) if llm_r else {}

        row = {
            "act_id": regex_r.act_id,
            "act_type": regex_r.act_type,
            "marge_text": regex_r.raw_marge,
            "plein_texte_text": regex_r.raw_plein_texte,
        }

        # Collect all entity field names from both
        all_fields = list(
            dict.fromkeys(k for d in [rf, lf] for k in d if k not in _SKIP_FIELDS)
        )

        for field in all_fields:
            r_val = rf.get(field)
            l_val = lf.get(field)

            row[f"{field}_regex"] = r_val
            row[f"{field}_llm"] = l_val
            # Final: prefer LLM, fall back to regex
            row[f"{field}_final"] = l_val if l_val is not None else r_val

        rows.append(row)

    return rows


def save_consolidated(
    rows: List[dict],
    output_path: Path,
) -> Path:
    """Save consolidated rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No rows to save")
        return output_path

    # Collect ALL field names across all rows for a consistent header
    all_fields = dict.fromkeys(k for row in rows for k in row.keys())
    fieldnames = list(all_fields.keys())

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=";",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved consolidated dataset: %s (%d rows)", output_path, len(rows))
    return output_path


def merge_from_files(
    ner_dir: str,
    output_dir: str,
    commune: str,
    year: str,
) -> Path:
    """Load saved regex and LLM results from disk and produce a consolidated CSV.

    Args:
        ner_dir: Directory containing ner_regex.json and ner_llm.json.
        output_dir: Directory for the final consolidated CSV.
        commune: Commune name (for filename).
        year: Year (for filename).

    Returns:
        Path to the output CSV.
    """
    ner_path = Path(ner_dir)

    regex_path = ner_path / "ner_regex.json"
    llm_path = ner_path / "ner_llm.json"

    if not regex_path.exists():
        raise FileNotFoundError(f"Regex results not found: {regex_path}")

    regex_results = [
        NERResult(**r) for r in json.loads(regex_path.read_text(encoding="utf-8"))
    ]
    logger.info("Loaded %d regex results from %s", len(regex_results), regex_path)

    llm_results = []
    if llm_path.exists():
        llm_results = [
            NERResult(**r) for r in json.loads(llm_path.read_text(encoding="utf-8"))
        ]
        logger.info("Loaded %d LLM results from %s", len(llm_results), llm_path)
    else:
        logger.warning(
            "No LLM results found at %s — final column will use regex only", llm_path
        )

    rows = merge_results(regex_results, llm_results)

    output_path = Path(output_dir) / f"{commune}_{year}.csv"
    save_consolidated(rows, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge regex + LLM NER results into consolidated CSV."
    )
    parser.add_argument(
        "--ner_dir",
        type=str,
        required=True,
        help="Directory with ner_regex.json and ner_llm.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for consolidated CSV",
    )
    parser.add_argument("--commune", type=str, default="abymes")
    parser.add_argument("--year", type=str, default="1842")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = merge_from_files(
        args.ner_dir, args.output_dir, args.commune, args.year
    )
    print(f"Consolidated dataset: {output_path}")


if __name__ == "__main__":
    main()
