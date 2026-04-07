"""Compare NER extraction: regex vs gemini-3-flash-preview vs gemini-2.5-flash-lite on 20 acts.

Produces a markdown comparison file with:
- Side-by-side regex / flash / flash-lite values for each field
- Embedded crop images (Marge + Plein Texte)
- Full OCR texts (Marge + Plein Texte)
- Agreement stats
"""

import asyncio
import base64
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.ner.llm_extractor import LLMExtractor
from trocr_handwritten.ner.schemas import ActRecord, flatten_ner_result

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/Gemini3_transcribed/"
)
NER_DIR = BASE_DIR / "ner"
LOG_DIR = Path("c:/Users/marie/Github/trocr_handwritten/logs")
COST_LOG_DIR = NER_DIR / "cost_logs"

MODEL_A = "gemini-3-flash-preview"
MODEL_B = "gemini-2.5-flash-lite"
N_ACTS = 20
MAX_CONCURRENT = 5


# ---------------------------------------------------------------------------
# Flatten nested regex/LLM NER results
# ---------------------------------------------------------------------------


def flatten_nested_ner(record: dict) -> dict:
    """Flatten a nested NER result (from ner_regex.json / ner_llm.json) to flat keys."""
    flat = {
        "act_id": record.get("act_id"),
        "act_type": record.get("act_type"),
        "extraction_method": record.get("extraction_method"),
        "raw_marge": record.get("raw_marge"),
        "raw_plein_texte": record.get("raw_plein_texte"),
    }

    death = record.get("death_act")
    if death:
        person = death.get("person", {}) or {}
        flat.update(
            {
                "person_name": person.get("name"),
                "person_sex": person.get("sex"),
                "person_age": person.get("age"),
                "person_qualifier": person.get("qualifier"),
                "person_occupation": person.get("occupation"),
                "person_registration_register": person.get("registration_register"),
                "person_registration_number": person.get("registration_number"),
                "death_date": death.get("death_date"),
                "death_place": death.get("death_place"),
                "declaration_date": death.get("declaration_date"),
                "declarant_name": death.get("declarant_name"),
                "declarant_age": death.get("declarant_age"),
                "declarant_occupation": death.get("declarant_occupation"),
                "owner_name": death.get("owner_name"),
                "owner_commune": death.get("owner_commune"),
                "owner_residence": death.get("owner_residence"),
                "habitation_name": death.get("habitation_name"),
                "officer_name": death.get("officer_name"),
                "commune": death.get("commune"),
            }
        )

    birth = record.get("birth_act")
    if birth:
        child = birth.get("child", {}) or {}
        mother = birth.get("mother", {}) or {}
        father = birth.get("father", {}) or {}
        flat.update(
            {
                "child_name": child.get("name"),
                "child_sex": child.get("sex"),
                "child_registration_register": child.get("registration_register"),
                "child_registration_number": child.get("registration_number"),
                "child_qualifier": child.get("qualifier"),
                "mother_name": mother.get("name"),
                "mother_age": mother.get("age"),
                "mother_registration_register": mother.get("registration_register"),
                "mother_registration_number": mother.get("registration_number"),
                "father_name": father.get("name"),
                "father_age": father.get("age"),
                "birth_date": birth.get("birth_date"),
                "birth_place": birth.get("birth_place"),
                "declaration_date": birth.get("declaration_date"),
                "declarant_name": birth.get("declarant_name"),
                "declarant_age": birth.get("declarant_age"),
                "declarant_occupation": birth.get("declarant_occupation"),
                "owner_name": birth.get("owner_name"),
                "owner_commune": birth.get("owner_commune"),
                "owner_residence": birth.get("owner_residence"),
                "habitation_name": birth.get("habitation_name"),
                "officer_name": birth.get("officer_name"),
                "commune": birth.get("commune"),
            }
        )

    return flat


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------


def load_acts_and_regex():
    """Load acts dataset and existing regex results (flattened)."""
    with open(NER_DIR / "acts_dataset.json", encoding="utf-8") as f:
        all_acts = [ActRecord(**a) for a in json.load(f)]

    with open(NER_DIR / "ner_regex.json", encoding="utf-8") as f:
        all_regex = json.load(f)

    # Filter non-unknown acts, take first N
    acts = [a for a in all_acts if a.act_type in ("deces", "naissance")][:N_ACTS]
    act_ids = {a.act_id for a in acts}

    # Flatten regex results
    regex_by_id = {}
    for r in all_regex:
        if r["act_id"] in act_ids:
            regex_by_id[r["act_id"]] = flatten_nested_ner(r)

    # Load existing LLM results (gemini-3-flash) if available — also flatten
    llm_flash = {}
    llm_path = NER_DIR / "ner_llm.json"
    if llm_path.exists():
        with open(llm_path, encoding="utf-8") as f:
            for r in json.load(f):
                if r["act_id"] in act_ids:
                    llm_flash[r["act_id"]] = flatten_nested_ner(r)

    return acts, regex_by_id, llm_flash


# ---------------------------------------------------------------------------
# Run LLM extraction for a given model
# ---------------------------------------------------------------------------


async def run_extraction(acts: list[ActRecord], model_name: str) -> dict[str, dict]:
    """Run NER extraction on acts with the given model. Returns {act_id: flat_dict}."""
    settings = LLMSettings(
        provider="gemini",
        model_name=model_name,
        temperature=0.0,
        max_tokens=4096,
        request_timeout=60,
    )
    extractor = LLMExtractor(settings)
    results = await extractor.extract_batch(acts, max_concurrent=MAX_CONCURRENT)

    # Log costs
    extractor.cost_tracker.log_summary(log_dir=str(COST_LOG_DIR))
    extractor.cost_tracker.log_summary(log_dir=str(LOG_DIR))

    return {r.act_id: flatten_ner_result(r) for r in results}


# ---------------------------------------------------------------------------
# Find crop images for an act
# ---------------------------------------------------------------------------


def find_crop_images(act: ActRecord) -> dict[str, list[str]]:
    """Return {Marge: [paths], Plein Texte: [paths]} for an act."""
    page_dir = BASE_DIR / act.source_page
    meta_path = page_dir / "metadata.json"
    images = {"Marge": [], "Plein Texte": []}

    if not meta_path.exists():
        return images

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Find the reading order entry for this act
    order = act.order_on_page
    for entry in meta.get("reading_order", []):
        if entry.get("order") == order:
            marge_file = entry.get("marge")
            pt_file = entry.get("plein_texte")
            if marge_file:
                p = page_dir / "Marge" / marge_file
                if p.exists():
                    images["Marge"].append(str(p))
            if pt_file:
                p = page_dir / "Plein Texte" / pt_file
                if p.exists():
                    images["Plein Texte"].append(str(p))
            break

    return images


def img_to_base64(path: str) -> str:
    """Convert an image file to a base64 data URI for embedding in markdown."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    ext = Path(path).suffix.lower()
    mime = {
        "jpg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "image/jpeg")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Build comparison markdown
# ---------------------------------------------------------------------------


def get_fields_for_type(act_type: str) -> list[str]:
    """Return the NER fields relevant for this act type."""
    if act_type == "deces":
        return [
            "person_name",
            "person_sex",
            "person_age",
            "person_occupation",
            "person_registration_register",
            "person_registration_number",
            "death_date",
            "death_time",
            "death_place",
            "declaration_date",
            "declaration_time",
            "declarant_name",
            "declarant_age",
            "declarant_occupation",
            "owner_name",
            "habitation_name",
            "officer_name",
            "commune",
        ]
    else:  # naissance
        return [
            "child_name",
            "child_sex",
            "child_registration_register",
            "child_registration_number",
            "mother_name",
            "mother_age",
            "mother_registration_register",
            "mother_registration_number",
            "father_name",
            "father_age",
            "birth_date",
            "birth_time",
            "birth_place",
            "declaration_date",
            "declaration_time",
            "declarant_name",
            "declarant_age",
            "declarant_occupation",
            "owner_name",
            "habitation_name",
            "officer_name",
            "commune",
        ]


def compare_value(a, b):
    """Compare two values, return status."""
    a_str = str(a).strip().lower() if a else ""
    b_str = str(b).strip().lower() if b else ""
    if not a_str and not b_str:
        return "both_null"
    if not a_str:
        return "b_only"
    if not b_str:
        return "a_only"
    if a_str == b_str or a_str in b_str or b_str in a_str:
        return "match"
    return "mismatch"


def escape_md(val):
    """Escape pipe characters for markdown tables."""
    if val is None:
        return "-"
    s = str(val).strip()
    return s.replace("|", "\\|") if s else "-"


def build_markdown(
    acts: list[ActRecord],
    regex_by_id: dict,
    flash_by_id: dict,
    flash_lite_by_id: dict,
) -> str:
    """Build a markdown comparison document."""
    lines = []
    lines.append(f"# NER Model Comparison: Regex vs {MODEL_A} vs {MODEL_B}")
    lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nSample: {len(acts)} acts from abymes/1842\n")

    # --- Aggregate stats ---
    stats = {"match": 0, "mismatch": 0, "a_only": 0, "b_only": 0, "both_null": 0}
    field_stats = {}
    all_comparisons = []

    for act in acts:
        fields = get_fields_for_type(act.act_type)
        regex = regex_by_id.get(act.act_id, {})
        flash = flash_by_id.get(act.act_id, {})
        flash_lite = flash_lite_by_id.get(act.act_id, {})

        for field in fields:
            r_val = regex.get(field)
            a_val = flash.get(field)
            b_val = flash_lite.get(field)

            status_ab = compare_value(a_val, b_val)
            stats[status_ab] += 1

            if field not in field_stats:
                field_stats[field] = {
                    "match": 0,
                    "mismatch": 0,
                    "a_only": 0,
                    "b_only": 0,
                    "both_null": 0,
                }
            field_stats[field][status_ab] += 1

            all_comparisons.append(
                {
                    "act_id": act.act_id,
                    "field": field,
                    "regex": r_val,
                    "flash": a_val,
                    "flash_lite": b_val,
                    "status": status_ab,
                }
            )

    # Summary table
    total_compared = stats["match"] + stats["mismatch"]
    agreement = (stats["match"] / total_compared * 100) if total_compared else 0

    lines.append("## Summary (Flash vs Flash-Lite)\n")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append(f"| Fields compared | {sum(stats.values())} |")
    lines.append(f"| Match (flash = flash-lite) | {stats['match']} |")
    lines.append(f"| Mismatch | {stats['mismatch']} |")
    lines.append(f"| Flash only | {stats['a_only']} |")
    lines.append(f"| Flash-Lite only | {stats['b_only']} |")
    lines.append(f"| Both null | {stats['both_null']} |")
    lines.append(f"| **Agreement rate** | **{agreement:.1f}%** |")

    # Per-field stats
    lines.append("\n## Per-field Agreement (Flash vs Flash-Lite)\n")
    lines.append("| Field | Match | Mismatch | Flash only | Lite only | Both null |")
    lines.append("|---|---|---|---|---|---|")
    for field, fs in sorted(field_stats.items()):
        lines.append(
            f"| {field} | {fs['match']} | {fs['mismatch']} | {fs['a_only']} | {fs['b_only']} | {fs['both_null']} |"
        )

    # Mismatches detail
    mismatches = [c for c in all_comparisons if c["status"] == "mismatch"]
    if mismatches:
        lines.append("\n## Mismatches Detail\n")
        lines.append("| Act | Field | Regex | Flash | Flash-Lite |")
        lines.append("|---|---|---|---|---|")
        for m in mismatches:
            lines.append(
                f"| {m['act_id'].split('_', 2)[-1]} | {m['field']} "
                f"| {escape_md(m['regex'])} | {escape_md(m['flash'])} | {escape_md(m['flash_lite'])} |"
            )

    # --- Per-act detail with images and full text ---
    lines.append("\n---\n")
    lines.append("## Per-Act Detail\n")

    for act in acts:
        fields = get_fields_for_type(act.act_type)
        regex = regex_by_id.get(act.act_id, {})
        flash = flash_by_id.get(act.act_id, {})
        flash_lite = flash_lite_by_id.get(act.act_id, {})

        short_id = act.act_id.replace("abymes_1842_", "")
        lines.append(f"### {short_id} ({act.act_type})\n")

        # OCR texts
        lines.append(f"**Marge:**\n```\n{act.marge_text}\n```\n")
        lines.append(f"**Plein Texte:**\n```\n{act.plein_texte_text}\n```\n")

        # Comparison table
        lines.append("| Field | Regex | Flash | Flash-Lite | Status |")
        lines.append("|---|---|---|---|---|")

        for field in fields:
            r_val = escape_md(regex.get(field))
            a_val = escape_md(flash.get(field))
            b_val = escape_md(flash_lite.get(field))
            status = compare_value(flash.get(field), flash_lite.get(field))

            status_icon = {
                "match": "=",
                "mismatch": "**!=**",
                "a_only": "flash>",
                "b_only": "lite>",
                "both_null": "-",
            }.get(status, "?")

            lines.append(f"| {field} | {r_val} | {a_val} | {b_val} | {status_icon} |")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    logger.info("Loading acts dataset and existing results...")
    acts, regex_by_id, existing_flash = load_acts_and_regex()
    logger.info(
        f"Loaded {len(acts)} acts, {len(regex_by_id)} regex results, {len(existing_flash)} existing flash results"
    )

    # --- Run gemini-3-flash-preview on acts that don't have results yet ---
    acts_needing_flash = [a for a in acts if a.act_id not in existing_flash]
    if acts_needing_flash:
        logger.info(f"Running {MODEL_A} on {len(acts_needing_flash)} acts...")
        new_flash = await run_extraction(acts_needing_flash, MODEL_A)
        flash_by_id = {**existing_flash, **new_flash}
    else:
        flash_by_id = existing_flash
        logger.info(f"All {len(acts)} acts already have {MODEL_A} results")

    # --- Load existing flash-lite results if they exist (no re-run) ---
    flash_lite_path = NER_DIR / "ner_flash_lite_20.json"
    if flash_lite_path.exists():
        logger.info(f"Loading existing {MODEL_B} results from disk (no re-run)...")
        with open(flash_lite_path, encoding="utf-8") as f:
            flash_lite_raw = json.load(f)
        flash_lite_by_id = {r["act_id"]: r for r in flash_lite_raw}
        logger.info(f"Loaded {len(flash_lite_by_id)} flash-lite results")
    else:
        logger.info(f"Running {MODEL_B} on {len(acts)} acts...")
        flash_lite_by_id = await run_extraction(acts, MODEL_B)
        # Save raw results
        with open(flash_lite_path, "w", encoding="utf-8") as f:
            json.dump(list(flash_lite_by_id.values()), f, ensure_ascii=False, indent=2)

    # --- Build comparison markdown ---
    logger.info("Building comparison markdown...")
    md = build_markdown(acts, regex_by_id, flash_by_id, flash_lite_by_id)

    output_path = (
        LOG_DIR / f"ner_model_comparison_{datetime.now().strftime('%Y-%m-%d')}.md"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Comparison saved to {output_path}")

    # Print quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    total_fields = 0
    matches = 0
    mismatches = 0
    for act in acts:
        fields = get_fields_for_type(act.act_type)
        flash = flash_by_id.get(act.act_id, {})
        fl = flash_lite_by_id.get(act.act_id, {})
        for field in fields:
            s = compare_value(flash.get(field), fl.get(field))
            total_fields += 1
            if s == "match":
                matches += 1
            elif s == "mismatch":
                mismatches += 1

    print(f"Total fields: {total_fields}")
    print(f"Match: {matches} ({matches/total_fields*100:.1f}%)")
    print(f"Mismatch: {mismatches} ({mismatches/total_fields*100:.1f}%)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
