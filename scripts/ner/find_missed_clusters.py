"""Find unrecognized OCR variants by fuzzy-matching unmapped raw owner names against existing cluster canonicals.

Usage:
    python scripts/ner/find_missed_clusters.py <commune> [--threshold 78]

Reads {commune}_owner_pairs.json (raw extractions) and {commune}_clusters.json
(current curated clusters). Prints raw names not yet listed in any cluster's
variants, ranked by similarity to existing canonicals, so the user can spot
OCR variants that should be merged.

Also flags pairs of cluster canonicals with high mutual similarity, and
duplicate plantation cluster canonicals.
"""

import argparse
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
PAIRS_DIR = BASE / "NER_datasets/llm/owner_pairs"


_HONORIFICS = re.compile(
    r"\b(monsieur|mr|m|sieur|sr|sieurs|messieurs|mm|mme|mmes|made|"
    r"madame|mad|mde|dame|dames|demoiselle|delle|dlle|melle|mademoiselle|"
    r"veuve|vve|ve|feu|feue|succession|heritiers|h[eé]ritiers|mineurs?)\b\.?",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """Normalise for fuzzy matching: strip honorifics + non-letters."""
    s = (s or "").lower()
    s = _HONORIFICS.sub(" ", s)
    return re.sub(r"[^a-z]", "", s)


def find_missed(commune: str, threshold: int = 78) -> None:
    """Print suggestions for missed groupings in {commune}_clusters.json."""
    clusters_p = PAIRS_DIR / f"{commune}_clusters.json"
    pairs_p = PAIRS_DIR / f"{commune}_owner_pairs.json"
    with open(clusters_p, encoding="utf-8") as f:
        d = json.load(f)
    with open(pairs_p, encoding="utf-8") as f:
        pairs = json.load(f)

    def _scan(dim_name, key, raws):
        known = {v for c in d[key] for v in c.get("variants", [])}
        unknown = sorted(raws - known)
        canons = [c["canonical"] for c in d[key] if c.get("canonical")]
        print(
            f"\n=== {commune} {dim_name}: unmapped raws with similarity >= {threshold} ==="
        )
        print(f"({len(unknown)} unmapped of {len(raws)} total)\n")
        for u in unknown:
            nu = _norm(u)
            if not nu:
                continue
            best_score = 0
            best = None
            for c in canons:
                s = fuzz.ratio(nu, _norm(c))
                if s > best_score:
                    best_score = s
                    best = c
            if best_score >= threshold:
                print(f"  {best_score:>3.0f}  {u!r:55s} -> {best!r}")
        return canons

    owner_raws = {p["owner_name_raw"] for p in pairs if p["owner_name_raw"]}
    plant_raws = {p["habitation_name_raw"] for p in pairs if p["habitation_name_raw"]}
    canons = _scan("owners", "owner_clusters", owner_raws)
    _scan("plantations", "plantation_clusters", plant_raws)

    print(f"\n=== {commune}: similar owner cluster canonicals (fuzz>=78) ===")
    seen = set()
    for i, a in enumerate(canons):
        for b in canons[i + 1 :]:
            s = fuzz.ratio(_norm(a), _norm(b))
            if s >= 78 and (a, b) not in seen:
                print(f"  {s:>3.0f}  {a!r:55s} <-> {b!r}")
                seen.add((a, b))

    print(f"\n=== {commune}: similar plantation cluster canonicals (fuzz>=78) ===")
    plants = [c["canonical"] for c in d["plantation_clusters"] if c.get("canonical")]
    pseen = set()
    for i, a in enumerate(plants):
        for b in plants[i + 1 :]:
            s = fuzz.ratio(_norm(a), _norm(b))
            if s >= 78 and (a, b) not in pseen:
                print(f"  {s:>3.0f}  {a!r:35s} <-> {b!r}")
                pseen.add((a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("commune", type=str)
    parser.add_argument("--threshold", type=int, default=78)
    args = parser.parse_args()
    find_missed(args.commune, threshold=args.threshold)


if __name__ == "__main__":
    main()
