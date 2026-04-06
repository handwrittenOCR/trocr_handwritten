"""Display N random acts with their extracted NER fields side by side.

Shows Marge text, Plein Texte excerpt, and all extracted entities so you can
visually verify extraction quality.

Usage:
    python scripts/ner/show_ner_results.py                     # 5 random
    python scripts/ner/show_ner_results.py -n 10               # 10 random
    python scripts/ner/show_ner_results.py --type deces         # only deaths
    python scripts/ner/show_ner_results.py --type naissance     # only births
    python scripts/ner/show_ner_results.py --type mariage       # only marriages
    python scripts/ner/show_ner_results.py --commune abymes
    python scripts/ner/show_ner_results.py --seed 42            # reproducible
    python scripts/ner/show_ner_results.py --empty              # show acts with missing fields
"""

import argparse
import json
import random
from pathlib import Path

NER_BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/NER_datasets"
)

SEPARATOR = "=" * 80
SUB_SEP = "-" * 40


def load_ner_results():
    """Load NER regex results."""
    path = NER_BASE / "regex" / "ner_regex.json"
    if not path.exists():
        print(f"Not found: {path}")
        print("Run: python scripts/ner/run_regex_extraction.py")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(value):
    """Format a value for display."""
    if value is None:
        return "-"
    return str(value)


def _person_line(label, person):
    """Format a PersonInfo dict as one line."""
    if person is None:
        return f"  {label}: -"
    parts = []
    if person.get("name"):
        parts.append(person["name"])
    if person.get("sex"):
        parts.append(f"({person['sex']})")
    if person.get("age"):
        parts.append(f"âge {person['age']}")
    if person.get("occupation"):
        parts.append(f"[{person['occupation']}]")
    reg = ""
    if person.get("registration_register") or person.get("registration_number"):
        reg = f" mat. {_fmt(person.get('registration_register'))}-{_fmt(person.get('registration_number'))}"
    return f"  {label}: {' '.join(parts) if parts else '-'}{reg}"


def display_result(result, index):
    """Display one act with its NER extraction."""
    act_id = result["act_id"]
    act_type = result["act_type"]
    print(SEPARATOR)
    print(f"  [{index}] {act_id}")
    print(f"  type: {act_type}  |  method: {result.get('extraction_method', '?')}")
    if result.get("marge_act_type") or result.get("marge_act_name"):
        print(
            f"  marge -> type: {_fmt(result.get('marge_act_type'))}  |  name: {_fmt(result.get('marge_act_name'))}  |  owner: {_fmt(result.get('marge_act_owner'))}  |  no.{_fmt(result.get('marge_act_number'))}"
        )
    print(SEPARATOR)

    marge = result.get("raw_marge", "").strip()
    plein = result.get("raw_plein_texte", "").strip()

    print("\n  MARGE:")
    if marge:
        for line in marge.splitlines():
            print(f"    | {line}")
    else:
        print("    | (empty)")

    print("\n  PLEIN TEXTE (first 600 chars):")
    preview = plein[:1300]
    for line in preview.splitlines():
        print(f"    | {line}")
    if len(plein) > 1300:
        print(f"    | ... ({len(plein) - 1300} more chars)")

    print(f"\n  {SUB_SEP}")
    print("  EXTRACTED ENTITIES:")
    print(f"  {SUB_SEP}")

    if result.get("death_act"):
        d = result["death_act"]
        print(_person_line("Deceased", d.get("person")))
        print(
            f"  Death date: {_fmt(d.get('death_date'))}  time: {_fmt(d.get('death_time'))}"
        )
        print(f"  Death place: {_fmt(d.get('death_place'))}")
        print(
            f"  Declaration: {_fmt(d.get('declaration_date'))}  time: {_fmt(d.get('declaration_time'))}"
        )
        print(
            f"  Declarant: {_fmt(d.get('declarant_name'))}  âge {_fmt(d.get('declarant_age'))}  [{_fmt(d.get('declarant_occupation'))}]"
        )
        print(
            f"  Owner: {_fmt(d.get('owner_name'))}  |  Habitation: {_fmt(d.get('habitation_name'))}"
        )
        print(f"  Officer: {_fmt(d.get('officer_name'))}")

    elif result.get("birth_act"):
        b = result["birth_act"]
        print(_person_line("Child", b.get("child")))
        print(_person_line("Mother", b.get("mother")))
        print(_person_line("Father", b.get("father")))
        print(
            f"  Birth date: {_fmt(b.get('birth_date'))}  time: {_fmt(b.get('birth_time'))}"
        )
        print(f"  Birth place: {_fmt(b.get('birth_place'))}")
        print(
            f"  Declaration: {_fmt(b.get('declaration_date'))}  time: {_fmt(b.get('declaration_time'))}"
        )
        print(
            f"  Declarant: {_fmt(b.get('declarant_name'))}  âge {_fmt(b.get('declarant_age'))}  [{_fmt(b.get('declarant_occupation'))}]"
        )
        print(
            f"  Owner: {_fmt(b.get('owner_name'))}  |  Habitation: {_fmt(b.get('habitation_name'))}"
        )
        print(f"  Officer: {_fmt(b.get('officer_name'))}")

    elif result.get("marriage_act"):
        m = result["marriage_act"]
        print(_person_line("Spouse 1", m.get("spouse1")))
        print(_person_line("Spouse 2", m.get("spouse2")))
        print(
            f"  Marriage date: {_fmt(m.get('marriage_date'))}  time: {_fmt(m.get('marriage_time'))}"
        )
        print(
            f"  Declaration: {_fmt(m.get('declaration_date'))}  time: {_fmt(m.get('declaration_time'))}"
        )
        print(
            f"  Owner: {_fmt(m.get('owner_name'))}  |  Habitation: {_fmt(m.get('habitation_name'))}"
        )
        print(f"  Officer: {_fmt(m.get('officer_name'))}")

    else:
        print("  (no entities extracted)")

    print()


def has_missing_fields(result):
    """Check if key fields are missing from extraction."""
    if result.get("death_act"):
        d = result["death_act"]
        p = d.get("person") or {}
        return not p.get("name") or not d.get("death_date")
    if result.get("birth_act"):
        b = result["birth_act"]
        c = b.get("child") or {}
        return not c.get("name") or not b.get("birth_date")
    if result.get("marriage_act"):
        m = result["marriage_act"]
        s1 = m.get("spouse1") or {}
        return not s1.get("name")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Display random NER extraction results."
    )
    parser.add_argument("-n", type=int, default=5, help="Number of acts to display")
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["deces", "naissance", "mariage", "unknown"],
    )
    parser.add_argument("--commune", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--empty", action="store_true", help="Show acts with missing key fields"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    results = load_ner_results()
    if not results:
        return

    if args.type:
        results = [r for r in results if r["act_type"] == args.type]
    if args.commune:
        results = [r for r in results if r["act_id"].startswith(args.commune + "_")]
    if args.empty:
        results = [r for r in results if has_missing_fields(r)]

    if not results:
        print("No matching acts found.")
        return

    n = min(args.n, len(results))
    sample = random.sample(results, n)

    filters = []
    if args.type:
        filters.append(f"type={args.type}")
    if args.commune:
        filters.append(f"commune={args.commune}")
    if args.empty:
        filters.append("missing fields only")
    filter_desc = f" ({', '.join(filters)})" if filters else ""

    print(f"\nShowing {n}/{len(results)} acts{filter_desc}\n")

    for i, result in enumerate(sample, 1):
        display_result(result, i)


if __name__ == "__main__":
    main()
