"""Run regex NER extraction on all communes/years (or a specific one).

Loads acts_dataset.json, runs RegexExtractor, saves ner_regex.json and ner_regex.csv.

Usage:
    python scripts/ner/run_regex_extraction.py                          # all
    python scripts/ner/run_regex_extraction.py --commune abymes         # one commune
    python scripts/ner/run_regex_extraction.py --commune abymes --year 1842
    python scripts/ner/run_regex_extraction.py --commune abymes --year 1842 -n 20  # first 20
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.ner.regex_extractor import RegexExtractor
from trocr_handwritten.ner.schemas import ActRecord
from trocr_handwritten.ner.pipeline import save_ner_results

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/Gemini3_transcribed"
)


def main():
    parser = argparse.ArgumentParser(description="Run regex NER extraction.")
    parser.add_argument("--commune", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument("-n", type=int, default=None, help="Limit number of acts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    extractor = RegexExtractor()
    total_acts = 0
    total_deaths = 0
    total_births = 0

    commune_dirs = [BASE / args.commune] if args.commune else sorted(BASE.iterdir())

    for commune_dir in commune_dirs:
        if not commune_dir.is_dir():
            continue
        commune = commune_dir.name
        year_dirs = (
            [commune_dir / args.year] if args.year else sorted(commune_dir.iterdir())
        )

        for year_dir in year_dirs:
            dataset_path = year_dir / "acts_dataset.json"
            if not dataset_path.exists():
                continue
            year = year_dir.name

            with open(dataset_path, encoding="utf-8") as f:
                data = json.load(f)

            records = [ActRecord(**a) for a in data]
            if args.n:
                records = records[: args.n]

            results = extractor.extract_all(records)

            # Save to ner/ subfolder
            ner_dir = str(year_dir / "ner")
            save_ner_results(results, ner_dir, "regex")

            # Stats
            type_counts = Counter(r.act_type for r in results)
            deaths = sum(1 for r in results if r.death_act)
            births = sum(1 for r in results if r.birth_act)
            types_str = ", ".join(f"{t}:{c}" for t, c in type_counts.most_common())
            print(
                f"{commune}/{year}: {len(results)} acts ({types_str}) — extracted {deaths} deaths, {births} births"
            )

            total_acts += len(results)
            total_deaths += deaths
            total_births += births

    print(
        f"\nTotal: {total_acts} acts, {total_deaths} deaths extracted, {total_births} births extracted"
    )


if __name__ == "__main__":
    main()
