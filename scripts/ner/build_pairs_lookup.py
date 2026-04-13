"""Build the (owner_name_raw, plantation_name_raw) → (owner_name_clean, plantation_name_clean) lookup.

Reads {commune}_owner_pairs.json + {commune}_clusters.json and writes {commune}_pairs_lookup.json.
Every observed (owner_raw, plantation_raw) pair is present in the output with canonical spellings.
Acts with only an owner and no plantation are included (plantation_name_clean = null).

Usage:
    python scripts/ner/build_pairs_lookup.py
    python scripts/ner/build_pairs_lookup.py --commune abymes
"""

import argparse
import json
from pathlib import Path

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
PAIRS_DIR = BASE / "NER_datasets/llm/owner_pairs"


def _build_lookup_from_clusters(clusters: list[dict]) -> dict[str, str | None]:
    """Return {raw_name: canonical} from a cluster list (owner or plantation)."""
    lookup: dict[str, str | None] = {}
    for cluster in clusters:
        canonical = cluster.get("canonical")
        for raw in cluster.get("variants", []):
            if raw and str(raw).strip():
                lookup[raw] = canonical
    return lookup


def build_pairs_lookup(commune: str) -> list[dict]:
    pairs_path = PAIRS_DIR / f"{commune}_owner_pairs.json"
    clusters_path = PAIRS_DIR / f"{commune}_clusters.json"

    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)

    owner_lookup: dict[str, str | None] = {}
    plantation_lookup: dict[str, str | None] = {}

    if clusters_path.exists():
        with open(clusters_path, encoding="utf-8") as f:
            clusters = json.load(f)
        owner_lookup = _build_lookup_from_clusters(clusters.get("owner_clusters", []))
        plantation_lookup = _build_lookup_from_clusters(
            clusters.get("plantation_clusters", [])
        )

    rows = []
    for pair in pairs:
        owner_raw = pair["owner_name_raw"]
        plantation_raw = pair["habitation_name_raw"]

        owner_clean = owner_lookup.get(owner_raw, owner_raw)
        plantation_clean = (
            plantation_lookup.get(plantation_raw, plantation_raw)
            if plantation_raw
            else None
        )

        rows.append(
            {
                "owner_name_raw": owner_raw,
                "plantation_name_raw": plantation_raw,
                "owner_name_clean": owner_clean,
                "plantation_name_clean": plantation_clean,
                "count": pair["count"],
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commune", type=str, default=None)
    args = parser.parse_args()

    if args.commune:
        communes = [args.commune]
    else:
        communes = [
            p.stem.replace("_owner_pairs", "")
            for p in sorted(PAIRS_DIR.glob("*_owner_pairs.json"))
        ]

    for commune in communes:
        rows = build_pairs_lookup(commune)
        out_path = PAIRS_DIR / f"{commune}_pairs_lookup.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        changed = sum(
            1
            for r in rows
            if r["owner_name_raw"] != r["owner_name_clean"]
            or r["plantation_name_raw"] != r["plantation_name_clean"]
        )
        print(
            f"  {commune}: {len(rows)} pairs, {changed} canonicalized → {out_path.name}"
        )


if __name__ == "__main__":
    main()
