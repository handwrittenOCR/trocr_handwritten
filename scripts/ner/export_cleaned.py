"""Export cleaned NER results to per-type CSVs.

Usage:
    python scripts/ner/export_cleaned.py
    python scripts/ner/export_cleaned.py --commune abymes
    python scripts/ner/export_cleaned.py --commune abymes --clusters path/to/abymes_clusters.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.ner.cleaning.export import export_all

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
NER_JSON = BASE / "NER_datasets/llm/ner_llm.json"
PAIRS_DIR = BASE / "NER_datasets/llm/owner_pairs"
OUTPUT_DIR = BASE / "NER_datasets/llm/cleaned"


def _find_clusters(commune: str | None, clusters_arg: str | None) -> Path | None:
    """Return clusters JSON path: explicit arg > auto-detect from commune > None."""
    if clusters_arg:
        return Path(clusters_arg)
    if commune:
        candidate = PAIRS_DIR / f"{commune}_clusters.json"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commune",
        type=str,
        default=None,
        help="Filter to one commune before exporting (default: all)",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to {commune}_clusters.json (auto-detected from --commune if omitted)",
    )
    args = parser.parse_args()

    with open(NER_JSON, encoding="utf-8") as f:
        all_records = json.load(f)

    if args.commune:
        clusters_path = _find_clusters(args.commune, args.clusters)
        print(
            f"Using clusters: {clusters_path.name}"
            if clusters_path
            else "No clusters file — falling back to fuzzy clustering"
        )
        records_to_export = [
            r for r in all_records if r["act_id"].startswith(args.commune + "_")
        ]
        print(
            f"Filtering to commune '{args.commune}': {len(records_to_export)} records"
        )
        tmp_path = NER_JSON.parent / f"ner_llm_{args.commune}_tmp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records_to_export, f, ensure_ascii=False)
        export_all(tmp_path, OUTPUT_DIR, clusters_json_path=clusters_path)
        tmp_path.unlink()
    else:
        by_commune: dict[str, list] = defaultdict(list)
        for r in all_records:
            commune = r["act_id"].split("_")[0]
            by_commune[commune].append(r)

        for idx, (commune, records) in enumerate(sorted(by_commune.items())):
            clusters_path = _find_clusters(commune, None)
            label = clusters_path.name if clusters_path else "fuzzy"
            print(f"  {commune}: {len(records)} records [{label}]")
            tmp_path = NER_JSON.parent / f"ner_llm_{commune}_tmp.json"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            export_all(
                tmp_path, OUTPUT_DIR, clusters_json_path=clusters_path, append=(idx > 0)
            )
            tmp_path.unlink()


if __name__ == "__main__":
    main()
