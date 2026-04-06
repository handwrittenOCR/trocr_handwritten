"""Run regex NER extraction on all communes/years (or a specific one).

Loads acts_dataset.json, runs RegexExtractor, saves ner_regex.json and ner_regex.csv.

Usage:
    python scripts/ner/run_regex_extraction.py                          # all
    python scripts/ner/run_regex_extraction.py --commune abymes         # one commune
    python scripts/ner/run_regex_extraction.py --commune abymes --year 1842
    python scripts/ner/run_regex_extraction.py --commune abymes --year 1842 -n 20  # first 20
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.ner.regex_extractor import RegexExtractor
from trocr_handwritten.ner.schemas import ActRecord, flatten_ner_result

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/Gemini3_transcribed"
)


def main():
    parser = argparse.ArgumentParser(description="Run regex NER extraction.")
    parser.add_argument("-n", type=int, default=None, help="Limit number of acts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    ner_base = Path(
        "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
        "3. OCR/2. TrOCR/5. Data (output)/ECES/NER_datasets"
    )
    dataset_path = ner_base / "raw" / "acts_dataset.json"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run: python scripts/ocr/build_all_datasets.py")
        return

    with open(dataset_path, encoding="utf-8") as f:
        records = [ActRecord(**a) for a in json.load(f)]
    if args.n:
        records = records[: args.n]

    extractor = RegexExtractor()
    results = extractor.extract_all(records)

    ner_path = ner_base / "regex"
    ner_path.mkdir(parents=True, exist_ok=True)
    with open(ner_path / "ner_regex.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)
    rows = [flatten_ner_result(r) for r in results]
    if rows:
        all_fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with open(ner_path / "ner_regex.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    type_counts = Counter(r.act_type for r in results)
    types_str = ", ".join(f"{t}:{c}" for t, c in type_counts.most_common())
    deaths = sum(1 for r in results if r.death_act)
    births = sum(1 for r in results if r.birth_act)

    print(f"Total: {len(results)} acts ({types_str})")
    print(f"Extracted: {deaths} deaths, {births} births")
    print(f"Output: {ner_path}")


if __name__ == "__main__":
    main()
