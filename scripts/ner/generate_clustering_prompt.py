"""Generate a copy-pasteable Claude prompt for owner clustering, one file per commune.

Usage:
    python scripts/ner/generate_clustering_prompt.py
    python scripts/ner/generate_clustering_prompt.py --commune abymes
"""

import argparse
from pathlib import Path

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
PAIRS_DIR = BASE / "NER_datasets/llm/owner_pairs"
PROMPTS_DIR = BASE / "NER_datasets/llm/owner_pairs/prompts"

PROMPT_TEMPLATE = """\
I am working on 19th-century French colonial civil registries from Guadeloupe (commune: {commune}).
The records were transcribed by OCR from handwritten documents.

Read the file at:
  {pairs_file}

It contains a JSON array of unique (owner_name_raw, habitation_name_raw, count) tuples extracted
from ner_birth.csv, ner_death.csv, ner_marriage.csv. Many entries refer to the same real person or
plantation under different spellings, OCR errors, abbreviations, or partial names.

Your task: produce two separate cluster lists — one for OWNERS, one for PLANTATIONS.

NOTE: A single owner can have several distinct plantations of different types (e.g. Joseph Caillou
owns BOTH a "sucrerie" and a "caféyère" — these are separate physical plantations sharing one owner).
Don't merge such plantation entries — they're different estates even when they share an owner.

OWNER CLUSTERING RULES:
- Use BOTH the owner name AND the plantation as evidence. Same plantation = strong signal of same owner.
- Consider: OCR errors (P/B/D, T/F, ai/oi, accents), abbreviations (Sr/Sieur, Mme/Dame, Vve/Veuve,
  Dlle/Demoiselle), name order swaps, partial names (surname only vs full name).
- héritiers X and X → same family/estate, merge.
- père/fils → keep separate unless plantation confirms same person.
- Dame Veuve X and X → different people, keep separate.
- Multiple family members with the same surname but different first names on the SAME plantation →
  keep as separate owners (different people sharing one estate).
- [illisible], [non précisé] and similar → canonical: null.
- owner_name_raw that is a plantation description (e.g. "l'habitation Belleplaine") → canonical: null.
- Do NOT merge people who share only a surname if first names differ AND plantation also differs. IF first names differ but you suspect that this is an OCR error based on plantation name or the fact that one spelling is only found once in the records, assign them the same owner_id
- Canonical = most complete and most frequent form.

PLANTATION CLUSTERING RULES:
- Cluster spelling variants, OCR errors, abbreviations of the same plantation name.
- Use owner as supporting evidence: same owner + similar plantation spelling → likely same plantation.
- Canonical = most complete and most frequent form.

Output a single JSON object only, no commentary. Write it to:
  {clusters_file}

Each owner cluster MUST also include a "family" field grouping owners who share a surname / family
identity even when they're treated as distinct people (e.g. "Caillou Junior" and "Joseph Caillou" are
separate owners but both belong to family "Caillou"; "Veuve Le Bourg Lacoudrai" and "Bramfeld Le Bourg
Lacoudrai" both belong to family "Le Bourg Lacoudrai"). Set "family" to null if no family grouping applies.

PLANTATION SELF-REFERENCE RULE:
- "habitation de [owner X]" / "propriété de [owner X]" / "habitation du sieur [owner X]" → canonical is
  the owner's name (e.g. "Joseph Caillou" plantation), since the wording proves a habitation exists.
- Vague phrases like "habitation de sa maîtresse", "propriété des sus dits", "habitation délaissée
  par..." STILL imply that a habitation exists. Look up the owner field for the same act and assign
  that owner's name as the plantation canonical (e.g. owner = "veuve Le Bourg Lacoudrai" + habitation
  = "habitation de sa maîtresse" → plantation = "Veuve Le Bourg Lacoudrai"). Only set canonical: null
  when neither the habitation string nor the owner name yields a clear owner identity.

Format:
{{
  "owner_clusters": [
    {{"canonical": "<canonical owner name or null>", "family": "<family name or null>", "variants": ["<raw1>", "<raw2>", ...]}},
    ...
  ],
  "plantation_clusters": [
    {{"canonical": "<canonical plantation name or null>", "variants": ["<raw1>", "<raw2>", ...]}},
    ...
  ]
}}

Only include entries that have variants or need mapping to null.
Clean singletons can be omitted — they keep their raw name as canonical.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commune", type=str, default=None)
    args = parser.parse_args()

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.commune:
        pair_files = [PAIRS_DIR / f"{args.commune}_owner_pairs.json"]
    else:
        pair_files = sorted(PAIRS_DIR.glob("*_owner_pairs.json"))

    for pair_file in pair_files:
        commune = pair_file.stem.replace("_owner_pairs", "")
        clusters_file = PAIRS_DIR / f"{commune}_clusters.json"

        prompt = PROMPT_TEMPLATE.format(
            commune=commune,
            pairs_file=pair_file,
            clusters_file=clusters_file,
            cleaned_dir=PAIRS_DIR.parent / "cleaned",
        )

        out_path = PROMPTS_DIR / f"{commune}_clustering_prompt.txt"
        out_path.write_text(prompt, encoding="utf-8")
        print(f"  {commune} → {out_path}")

    print("Done. Open the .txt file and paste its contents into Claude.")


if __name__ == "__main__":
    main()
