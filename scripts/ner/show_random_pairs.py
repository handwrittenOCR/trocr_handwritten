"""Display N random Marge / Plein Texte pairs for visual verification.

Picks random acts from the dataset and prints the Marge and Plein Texte
side by side so you can check whether the pairing makes sense.

Usage:
    python scripts/ner/show_random_pairs.py                     # 5 random from all
    python scripts/ner/show_random_pairs.py -n 10               # 10 random
    python scripts/ner/show_random_pairs.py --commune abymes    # from one commune
    python scripts/ner/show_random_pairs.py --commune abymes --year 1842
    python scripts/ner/show_random_pairs.py --has-marge         # only pairs with marge
"""

import argparse
import json
import random
from pathlib import Path

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/Gemini3_transcribed"
)

SEPARATOR = "=" * 80


def load_acts(commune=None, year=None):
    """Load all acts from acts_dataset.json files."""
    acts = []
    commune_dirs = [BASE / commune] if commune else sorted(BASE.iterdir())

    for commune_dir in commune_dirs:
        if not commune_dir.is_dir():
            continue
        year_dirs = [commune_dir / year] if year else sorted(commune_dir.iterdir())

        for year_dir in year_dirs:
            dataset_path = year_dir / "acts_dataset.json"
            if not dataset_path.exists():
                continue
            with open(dataset_path, encoding="utf-8") as f:
                acts.extend(json.load(f))

    return acts


def display_pair(act, index):
    """Display a single Marge / Plein Texte pair."""
    print(SEPARATOR)
    print(f"  [{index}] {act['act_id']}")
    print(
        f"  commune: {act['commune']}  |  year: {act['year']}  |  page: {act['source_page']}  |  order: {act['order_on_page']}"
    )
    print(
        f"  marge file: {act.get('source_marge_file', '-')}  |  plein texte file: {act['source_plein_texte_file']}"
    )
    print(SEPARATOR)

    marge = act.get("marge_text", "").strip()
    plein = act.get("plein_texte_text", "").strip()

    print("\n  MARGE:")
    if marge:
        for line in marge.splitlines():
            print(f"    | {line}")
    else:
        print("    | (empty)")

    print("\n  PLEIN TEXTE:")
    # Show first 500 chars to keep it readable
    preview = plein[:1000]
    for line in preview.splitlines():
        print(f"    | {line}")
    if len(plein) > 1000:
        print(f"    | ... ({len(plein) - 1000} more chars)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Display random Marge/Plein Texte pairs."
    )
    parser.add_argument(
        "-n", type=int, default=5, help="Number of pairs to display (default: 5)"
    )
    parser.add_argument("--commune", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument(
        "--has-marge", action="store_true", help="Only show acts that have Marge text"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    acts = load_acts(args.commune, args.year)

    if args.has_marge:
        acts = [a for a in acts if a.get("marge_text", "").strip()]

    if not acts:
        print("No acts found.")
        return

    n = min(args.n, len(acts))
    sample = random.sample(acts, n)

    filter_desc = ""
    if args.commune:
        filter_desc += f" commune={args.commune}"
    if args.year:
        filter_desc += f" year={args.year}"
    if args.has_marge:
        filter_desc += " (with marge only)"

    print(f"\nShowing {n} random pairs from {len(acts)} acts{filter_desc}\n")

    for i, act in enumerate(sample, 1):
        display_pair(act, i)


if __name__ == "__main__":
    main()
