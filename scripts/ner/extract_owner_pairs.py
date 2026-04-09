"""Extract unique (owner_name_raw, habitation_name_raw) pairs per commune.

Reads the three cleaned CSVs (births, deaths, marriages) and writes one JSON file
per commune under NER_datasets/llm/owner_pairs/.

Usage:
    python scripts/ner/extract_owner_pairs.py
    python scripts/ner/extract_owner_pairs.py --commune abymes
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
CLEANED_DIR = BASE / "NER_datasets/llm/cleaned"
OUTPUT_DIR = BASE / "NER_datasets/llm/owner_pairs"
CSV_FILES = ["ner_birth.csv", "ner_death.csv", "ner_marriage.csv"]


def extract_pairs(commune_filter: str | None = None) -> dict[str, Counter]:
    """Return {commune: Counter({(owner, plantation): count})}."""
    by_commune: dict[str, Counter] = {}

    for fname in CSV_FILES:
        path = CLEANED_DIR / fname
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                commune = row.get("commune", "").strip()
                if not commune:
                    continue
                if commune_filter and commune != commune_filter:
                    continue
                owner = (row.get("owner_name_raw") or "").strip() or None
                plantation = (row.get("habitation_name_raw") or "").strip() or None
                if owner:
                    by_commune.setdefault(commune, Counter())
                    by_commune[commune][(owner, plantation)] += 1

    return by_commune


def write_pairs(by_commune: dict[str, Counter]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for commune, counter in sorted(by_commune.items()):
        rows = [
            {"owner_name_raw": owner, "habitation_name_raw": plantation, "count": count}
            for (owner, plantation), count in sorted(
                counter.items(), key=lambda x: -x[1]
            )
        ]
        out_path = OUTPUT_DIR / f"{commune}_owner_pairs.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(
            f"  {commune}: {len(rows)} unique (owner, plantation) pairs → {out_path.name}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commune", type=str, default=None)
    args = parser.parse_args()

    print("Extracting owner/plantation pairs...")
    by_commune = extract_pairs(args.commune)
    write_pairs(by_commune)
    print("Done.")


if __name__ == "__main__":
    main()
