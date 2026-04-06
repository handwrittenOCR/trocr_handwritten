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

    print(f"\nTotal: {len(results)} acts ({types_str})")
    print(f"Output: {ner_path}\n")

    death_results = [r for r in results if r.death_act]
    birth_results = [r for r in results if r.birth_act]
    marriage_results = [r for r in results if r.marriage_act]

    def _rate(items, accessor):
        n = sum(1 for r in items if accessor(r))
        pct = 100 * n / len(items) if items else 0
        return f"{n}/{len(items)} ({pct:.1f}%)"

    if death_results:
        print(f"--- Deaths ({len(death_results)}) ---")
        print(
            f"  person name:    {_rate(death_results, lambda r: r.death_act.person.name)}"
        )
        print(
            f"  person age:     {_rate(death_results, lambda r: r.death_act.person.age)}"
        )
        print(
            f"  person sex:     {_rate(death_results, lambda r: r.death_act.person.sex)}"
        )
        print(
            f"  occupation:     {_rate(death_results, lambda r: r.death_act.person.occupation)}"
        )
        print(
            f"  registration:   {_rate(death_results, lambda r: r.death_act.person.registration_number)}"
        )
        print(
            f"  death date:     {_rate(death_results, lambda r: r.death_act.death_date)}"
        )
        print(
            f"  death time:     {_rate(death_results, lambda r: r.death_act.death_time)}"
        )
        print(
            f"  declarant:      {_rate(death_results, lambda r: r.death_act.declarant_name)}"
        )
        print(
            f"  owner:          {_rate(death_results, lambda r: r.death_act.owner_name)}"
        )
        print(
            f"  habitation:     {_rate(death_results, lambda r: r.death_act.habitation_name)}"
        )
        print(
            f"  officer:        {_rate(death_results, lambda r: r.death_act.officer_name)}"
        )
        print(
            f"  commune:        {_rate(death_results, lambda r: r.death_act.commune)}"
        )

    if birth_results:
        print(f"\n--- Births ({len(birth_results)}) ---")
        print(
            f"  child name:     {_rate(birth_results, lambda r: r.birth_act.child.name)}"
        )
        print(
            f"  child sex:      {_rate(birth_results, lambda r: r.birth_act.child.sex)}"
        )
        print(
            f"  child reg:      {_rate(birth_results, lambda r: r.birth_act.child.registration_number)}"
        )
        print(
            f"  mother name:    {_rate(birth_results, lambda r: r.birth_act.mother.name)}"
        )
        print(
            f"  mother age:     {_rate(birth_results, lambda r: r.birth_act.mother.age)}"
        )
        print(
            f"  mother reg:     {_rate(birth_results, lambda r: r.birth_act.mother.registration_number)}"
        )
        print(
            f"  birth date:     {_rate(birth_results, lambda r: r.birth_act.birth_date)}"
        )
        print(
            f"  birth time:     {_rate(birth_results, lambda r: r.birth_act.birth_time)}"
        )
        print(
            f"  declarant:      {_rate(birth_results, lambda r: r.birth_act.declarant_name)}"
        )
        print(
            f"  owner:          {_rate(birth_results, lambda r: r.birth_act.owner_name)}"
        )
        print(
            f"  habitation:     {_rate(birth_results, lambda r: r.birth_act.habitation_name)}"
        )
        print(
            f"  officer:        {_rate(birth_results, lambda r: r.birth_act.officer_name)}"
        )
        print(
            f"  commune:        {_rate(birth_results, lambda r: r.birth_act.commune)}"
        )

    if marriage_results:
        print(f"\n--- Marriages ({len(marriage_results)}) ---")
        print(
            f"  spouse1 name:   {_rate(marriage_results, lambda r: r.marriage_act.spouse1.name)}"
        )
        print(
            f"  spouse1 age:    {_rate(marriage_results, lambda r: r.marriage_act.spouse1.age)}"
        )
        print(
            f"  spouse2 name:   {_rate(marriage_results, lambda r: r.marriage_act.spouse2.name)}"
        )
        print(
            f"  spouse2 age:    {_rate(marriage_results, lambda r: r.marriage_act.spouse2.age)}"
        )
        print(
            f"  owner:          {_rate(marriage_results, lambda r: r.marriage_act.owner_name)}"
        )
        print(
            f"  officer:        {_rate(marriage_results, lambda r: r.marriage_act.officer_name)}"
        )


if __name__ == "__main__":
    main()
