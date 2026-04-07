"""Build acts_dataset.json for all communes and years.

Iterates over every commune/year folder in the OCR_gem31 output
directory, calls build_dataset + save_dataset, and prints a summary.

Prints summary of the number of acts and crops for each commune/year.
Saves the dataset to NER_datasets/raw/acts_dataset.json.

Prints examples of split acts after processing.

Usage:
    python scripts/build_all_datasets.py
    python scripts/build_all_datasets.py --commune abymes
    python scripts/build_all_datasets.py --commune abymes --year 1842
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.ner.dataset import build_dataset
from trocr_handwritten.ner.split_acts import split_merged_acts

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/OCR_gem31"
)


def main():
    parser = argparse.ArgumentParser(
        description="Build act datasets for all communes/years."
    )
    parser.add_argument(
        "--commune", type=str, default=None, help="Specific commune (default: all)"
    )
    parser.add_argument(
        "--year", type=str, default=None, help="Specific year (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    all_records = []
    commune_dirs = (
        sorted(BASE.iterdir()) if args.commune is None else [BASE / args.commune]
    )

    for commune_dir in commune_dirs:
        if not commune_dir.is_dir():
            continue
        commune = commune_dir.name

        year_dirs = (
            sorted(commune_dir.iterdir())
            if args.year is None
            else [commune_dir / args.year]
        )

        for year_dir in year_dirs:
            if not year_dir.is_dir():
                continue
            year = year_dir.name

            records, pt_count = build_dataset(str(year_dir), commune, year)
            records = split_merged_acts(records)
            print(f"{commune}/{year}: {len(records)} acts  {pt_count} crops")
            all_records.extend(records)

    out_dir = Path(
        "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
        "3. OCR/2. TrOCR/5. Data (output)/ECES/NER_datasets/raw"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "acts_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in all_records], f, ensure_ascii=False, indent=2
        )
    print(f"\nTotal: {len(all_records)} acts -> {out_path}")


if __name__ == "__main__":
    main()
