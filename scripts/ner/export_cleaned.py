"""Export cleaned NER results to per-type CSVs.

Usage:
    python scripts/ner/export_cleaned.py
    python scripts/ner/export_cleaned.py --commune abymes
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.ner.cleaning.export import export_all

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
NER_JSON = BASE / "NER_datasets/ner_llm.json"
OUTPUT_DIR = BASE / "NER_datasets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commune",
        type=str,
        default=None,
        help="Filter to one commune before exporting (default: all)",
    )
    args = parser.parse_args()

    if args.commune:
        with open(NER_JSON, encoding="utf-8") as f:
            all_records = json.load(f)
        records_to_export = [
            r for r in all_records if r["act_id"].startswith(args.commune + "_")
        ]
        print(
            f"Filtering to commune '{args.commune}': {len(records_to_export)} records"
        )

        tmp_path = NER_JSON.parent / f"ner_llm_{args.commune}_tmp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records_to_export, f, ensure_ascii=False)
        export_all(tmp_path, OUTPUT_DIR / args.commune)
        tmp_path.unlink()
    else:
        export_all(NER_JSON, OUTPUT_DIR)


if __name__ == "__main__":
    main()
