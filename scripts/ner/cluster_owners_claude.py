"""Use Claude to cluster owner names per commune from (owner, plantation) pairs.

Reads owner_pairs/{commune}_owner_pairs.json, sends to Claude for clustering,
writes the result as owner_pairs/{commune}_owner_clusters.json.

Usage:
    python scripts/ner/cluster_owners_claude.py
    python scripts/ner/cluster_owners_claude.py --commune abymes
    python scripts/ner/cluster_owners_claude.py --commune abymes --model claude-opus-4-6
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
PAIRS_DIR = BASE / "NER_datasets/llm/owner_pairs"

SYSTEM_PROMPT = """You are an expert in 19th-century French colonial civil registries from Guadeloupe.
You will receive a list of (owner_name_raw, habitation_name_raw, count) tuples from a single commune.
These were extracted by OCR from handwritten documents and contain many spelling variants, OCR errors,
abbreviations, and partial names.

Your task: produce two separate cluster lists — one for owners, one for plantations.

OWNER CLUSTERING RULES:
- Use BOTH the owner name AND the plantation name as evidence. Same plantation = strong signal of same owner.
- Consider: OCR errors (P/B/D confusion, T/F, ai/oi, accents missing), abbreviations (Sr/Sieur, Mme/Dame,
  Vve/Veuve, Dlle/Demoiselle, Mlle), name order swaps, partial names (surname only vs full name).
- héritiers X and X → same family/estate, merge.
- père/fils → different generations, keep separate unless plantation confirms same person.
- Dame Veuve X and X → different people (widow ≠ deceased husband), keep separate.
- Multiple family members with the same surname but different first names on the SAME plantation → keep
  them as separate owners (they are different people sharing one estate).
- [illisible], [non précisé] and similar → canonical: null.
- Entries where owner_name_raw is actually a plantation description (e.g. "l'habitation Belleplaine") → canonical: null.
- Do NOT merge people who share only a common surname if their first names differ and the plantation differs.
- Canonical = most complete and most frequent form.

PLANTATION CLUSTERING RULES:
- Cluster spelling variants, OCR errors, and abbreviations of the same plantation name.
- Use the owner as supporting evidence: if two plantation spellings always co-occur with the same owner, they are likely the same plantation.
- A plantation name that is null stays null — do not invent a canonical name.
- Canonical = most complete and most frequent form.

Output a single JSON object with two arrays:
{
  "owner_clusters": [
    {"canonical": "<canonical owner name or null>", "variants": ["<raw1>", "<raw2>", ...]},
    ...
  ],
  "plantation_clusters": [
    {"canonical": "<canonical plantation name or null>", "variants": ["<raw1>", "<raw2>", ...]},
    ...
  ]
}

Only include entries that have variants or need mapping to null. Clean singletons can be omitted."""

USER_TEMPLATE = """Commune: {commune}

Here are all unique (owner_name_raw, habitation_name_raw, count) pairs, sorted by frequency:

{pairs_text}

Produce the owner_clusters and plantation_clusters JSON."""


def build_pairs_text(pairs: list[dict]) -> str:
    lines = []
    for p in pairs:
        owner = p["owner_name_raw"] or "[null]"
        plantation = p["habitation_name_raw"] or "[null]"
        lines.append(
            f"  owner={owner!r:60s} plantation={plantation!r:50s} n={p['count']}"
        )
    return "\n".join(lines)


def cluster_commune(
    commune: str,
    pairs: list[dict],
    client: anthropic.Anthropic,
    model: str,
) -> dict:
    """Return {"owner_clusters": [...], "plantation_clusters": [...]}."""
    pairs_text = build_pairs_text(pairs)
    user_msg = USER_TEMPLATE.format(commune=commune, pairs_text=pairs_text)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    if not isinstance(result, dict) or "owner_clusters" not in result:
        raise ValueError(f"Unexpected response format: {str(result)[:200]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commune", type=str, default=None, help="Single commune to process"
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if args.commune:
        pair_files = [PAIRS_DIR / f"{args.commune}_owner_pairs.json"]
    else:
        pair_files = sorted(PAIRS_DIR.glob("*_owner_pairs.json"))

    if not pair_files:
        print("No pair files found. Run extract_owner_pairs.py first.")
        return

    for pair_file in pair_files:
        commune = pair_file.stem.replace("_owner_pairs", "")
        out_path = PAIRS_DIR / f"{commune}_clusters.json"

        if out_path.exists():
            print(f"  {commune}: clusters already exist, skipping ({out_path.name})")
            continue

        with open(pair_file, encoding="utf-8") as f:
            pairs = json.load(f)

        print(f"  {commune}: {len(pairs)} pairs → sending to Claude ({args.model})...")
        t0 = time.time()

        try:
            clusters = cluster_commune(commune, pairs, client, args.model)
        except Exception as e:
            print(f"  ERROR for {commune}: {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        print(
            f"  {commune}: {len(clusters)} clusters written in {elapsed:.1f}s → {out_path.name}"
        )

        # Polite rate limiting between communes
        if len(pair_files) > 1:
            time.sleep(2)

    print("Done.")


if __name__ == "__main__":
    main()
