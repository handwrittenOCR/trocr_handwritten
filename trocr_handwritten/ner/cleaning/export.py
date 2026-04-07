"""Export cleaned NER results to per-type CSVs.

Reads ner_llm.json, applies date/age/entity cleaning, writes:
  - ner_llm_birth.csv
  - ner_llm_death.csv
  - ner_llm_marriage.csv
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional

from trocr_handwritten.ner.cleaning.ages import parse_age
from trocr_handwritten.ner.cleaning.dates import parse_act_dates
from trocr_handwritten.ner.cleaning.entities import build_entity_tables

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _year_from_act_id(act_id: str) -> Optional[int]:
    """Extract year_registry from act_id (format: commune_YEAR_...)."""
    m = re.search(r"_(\d{4})_", act_id)
    return int(m.group(1)) if m else None


def _resolve(lookup: dict, raw: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Return (canonical_name, entity_id) or (None, None) if not found."""
    if not raw or str(raw).strip().lower() in ("null", "none", ""):
        return None, None
    result = lookup.get(raw)
    if result:
        return result
    return raw, None


def _clean_str(val: Optional[str]) -> Optional[str]:
    """Replace string 'null' / 'none' with Python None."""
    if val is None:
        return None
    if str(val).strip().lower() in ("null", "none"):
        return None
    return val


# ---------------------------------------------------------------------------
# Per-type export
# ---------------------------------------------------------------------------


def export_deaths(
    records: list[dict],
    owner_lookup: dict,
    plantation_lookup: dict,
    output_path: Path,
) -> int:
    """Write ner_llm_death.csv. Returns number of rows written."""
    rows = []
    for r in records:
        if r.get("act_type") != "deces" or not r.get("death_act"):
            continue
        d = r["death_act"]
        p = d.get("person") or {}
        year_registry = _year_from_act_id(r["act_id"])

        decl_iso, event_iso = parse_act_dates(
            _clean_str(d.get("declaration_date")),
            _clean_str(d.get("death_date")),
            year_registry,
        )
        owner_clean, owner_id = _resolve(owner_lookup, _clean_str(d.get("owner_name")))
        plant_clean, plant_id = _resolve(
            plantation_lookup, _clean_str(d.get("habitation_name"))
        )

        rows.append(
            {
                "act_id": r["act_id"],
                "commune": r["act_id"].split("_")[0],
                "year_registry": year_registry,
                "raw_plein_texte": r.get("raw_plein_texte"),
                "marge_act_type": _clean_str(r.get("marge_act_type")),
                "marge_act_name": _clean_str(r.get("marge_act_name")),
                "marge_act_number": _clean_str(r.get("marge_act_number")),
                "marge_act_owner": _clean_str(r.get("marge_act_owner")),
                "person_name": _clean_str(p.get("name")),
                "person_sex": _clean_str(p.get("sex")),
                "person_qualifier": _clean_str(p.get("qualifier")),
                "person_age_raw": _clean_str(p.get("age")),
                "person_age": parse_age(p.get("age")),
                "person_occupation": _clean_str(p.get("occupation")),
                "person_registration_register": _clean_str(
                    p.get("registration_register")
                ),
                "person_registration_number": _clean_str(p.get("registration_number")),
                "death_date_raw": _clean_str(d.get("death_date")),
                "death_date": event_iso,
                "death_place": _clean_str(d.get("death_place")),
                "declaration_date_raw": _clean_str(d.get("declaration_date")),
                "declaration_date": decl_iso,
                "declarant_name": _clean_str(d.get("declarant_name")),
                "declarant_age_raw": _clean_str(d.get("declarant_age")),
                "declarant_age": parse_age(d.get("declarant_age")),
                "declarant_occupation": _clean_str(d.get("declarant_occupation")),
                "owner_name_raw": _clean_str(d.get("owner_name")),
                "owner_name_clean": owner_clean,
                "owner_id": owner_id,
                "owner_commune": _clean_str(d.get("owner_commune")),
                "owner_residence": _clean_str(d.get("owner_residence")),
                "habitation_name_raw": _clean_str(d.get("habitation_name")),
                "habitation_name_clean": plant_clean,
                "plantation_id": plant_id,
                "officer_name": _clean_str(d.get("officer_name")),
                "commune_act": _clean_str(d.get("commune")),
            }
        )

    _write_csv(rows, output_path)
    return len(rows)


def export_births(
    records: list[dict],
    owner_lookup: dict,
    plantation_lookup: dict,
    output_path: Path,
) -> int:
    """Write ner_llm_birth.csv. Returns number of rows written."""
    rows = []
    for r in records:
        if r.get("act_type") != "naissance" or not r.get("birth_act"):
            continue
        b = r["birth_act"]
        child = b.get("child") or {}
        mother = b.get("mother") or {}
        father = b.get("father") or {}
        year_registry = _year_from_act_id(r["act_id"])

        decl_iso, event_iso = parse_act_dates(
            _clean_str(b.get("declaration_date")),
            _clean_str(b.get("birth_date")),
            year_registry,
        )
        owner_clean, owner_id = _resolve(owner_lookup, _clean_str(b.get("owner_name")))
        plant_clean, plant_id = _resolve(
            plantation_lookup, _clean_str(b.get("habitation_name"))
        )

        rows.append(
            {
                "act_id": r["act_id"],
                "commune": r["act_id"].split("_")[0],
                "year_registry": year_registry,
                "raw_plein_texte": r.get("raw_plein_texte"),
                "marge_act_type": _clean_str(r.get("marge_act_type")),
                "marge_act_name": _clean_str(r.get("marge_act_name")),
                "marge_act_number": _clean_str(r.get("marge_act_number")),
                "marge_act_owner": _clean_str(r.get("marge_act_owner")),
                "child_name": _clean_str(child.get("name")),
                "child_sex": _clean_str(child.get("sex")),
                "child_qualifier": _clean_str(child.get("qualifier")),
                "child_registration_register": _clean_str(
                    child.get("registration_register")
                ),
                "child_registration_number": _clean_str(
                    child.get("registration_number")
                ),
                "mother_name": _clean_str(mother.get("name")),
                "mother_qualifier": _clean_str(mother.get("qualifier")),
                "mother_age_raw": _clean_str(mother.get("age")),
                "mother_age": parse_age(mother.get("age")),
                "mother_occupation": _clean_str(mother.get("occupation")),
                "mother_registration_register": _clean_str(
                    mother.get("registration_register")
                ),
                "mother_registration_number": _clean_str(
                    mother.get("registration_number")
                ),
                "father_name": _clean_str(father.get("name")),
                "father_age_raw": _clean_str(father.get("age")),
                "father_age": parse_age(father.get("age")),
                "birth_date_raw": _clean_str(b.get("birth_date")),
                "birth_date": event_iso,
                "birth_place": _clean_str(b.get("birth_place")),
                "declaration_date_raw": _clean_str(b.get("declaration_date")),
                "declaration_date": decl_iso,
                "declarant_name": _clean_str(b.get("declarant_name")),
                "declarant_age_raw": _clean_str(b.get("declarant_age")),
                "declarant_age": parse_age(b.get("declarant_age")),
                "declarant_occupation": _clean_str(b.get("declarant_occupation")),
                "owner_name_raw": _clean_str(b.get("owner_name")),
                "owner_name_clean": owner_clean,
                "owner_id": owner_id,
                "owner_commune": _clean_str(b.get("owner_commune")),
                "owner_residence": _clean_str(b.get("owner_residence")),
                "habitation_name_raw": _clean_str(b.get("habitation_name")),
                "habitation_name_clean": plant_clean,
                "plantation_id": plant_id,
                "officer_name": _clean_str(b.get("officer_name")),
                "commune_act": _clean_str(b.get("commune")),
            }
        )

    _write_csv(rows, output_path)
    return len(rows)


def export_marriages(
    records: list[dict],
    owner_lookup: dict,
    plantation_lookup: dict,
    output_path: Path,
) -> int:
    """Write ner_llm_marriage.csv. Returns number of rows written."""
    rows = []
    for r in records:
        if r.get("act_type") != "mariage" or not r.get("marriage_act"):
            continue
        m = r["marriage_act"]
        s1 = m.get("spouse1") or {}
        s2 = m.get("spouse2") or {}
        year_registry = _year_from_act_id(r["act_id"])

        decl_iso, event_iso = parse_act_dates(
            _clean_str(m.get("declaration_date")),
            _clean_str(m.get("marriage_date")),
            year_registry,
        )
        owner_clean, owner_id = _resolve(owner_lookup, _clean_str(m.get("owner_name")))
        plant_clean, plant_id = _resolve(
            plantation_lookup, _clean_str(m.get("habitation_name"))
        )

        rows.append(
            {
                "act_id": r["act_id"],
                "commune": r["act_id"].split("_")[0],
                "year_registry": year_registry,
                "raw_plein_texte": r.get("raw_plein_texte"),
                "marge_act_type": _clean_str(r.get("marge_act_type")),
                "marge_act_name": _clean_str(r.get("marge_act_name")),
                "marge_act_number": _clean_str(r.get("marge_act_number")),
                "marge_act_owner": _clean_str(r.get("marge_act_owner")),
                "spouse1_name": _clean_str(s1.get("name")),
                "spouse1_qualifier": _clean_str(s1.get("qualifier")),
                "spouse1_age_raw": _clean_str(s1.get("age")),
                "spouse1_age": parse_age(s1.get("age")),
                "spouse1_occupation": _clean_str(s1.get("occupation")),
                "spouse1_registration_register": _clean_str(
                    s1.get("registration_register")
                ),
                "spouse1_registration_number": _clean_str(
                    s1.get("registration_number")
                ),
                "spouse2_name": _clean_str(s2.get("name")),
                "spouse2_qualifier": _clean_str(s2.get("qualifier")),
                "spouse2_age_raw": _clean_str(s2.get("age")),
                "spouse2_age": parse_age(s2.get("age")),
                "spouse2_occupation": _clean_str(s2.get("occupation")),
                "spouse2_registration_register": _clean_str(
                    s2.get("registration_register")
                ),
                "spouse2_registration_number": _clean_str(
                    s2.get("registration_number")
                ),
                "marriage_date_raw": _clean_str(m.get("marriage_date")),
                "marriage_date": event_iso,
                "declaration_date_raw": _clean_str(m.get("declaration_date")),
                "declaration_date": decl_iso,
                "declarant_name": _clean_str(m.get("declarant_name")),
                "declarant_age_raw": _clean_str(m.get("declarant_age")),
                "declarant_age": parse_age(m.get("declarant_age")),
                "declarant_occupation": _clean_str(m.get("declarant_occupation")),
                "owner_name_raw": _clean_str(m.get("owner_name")),
                "owner_name_clean": owner_clean,
                "owner_id": owner_id,
                "owner_commune": _clean_str(m.get("owner_commune")),
                "owner_residence": _clean_str(m.get("owner_residence")),
                "habitation_name_raw": _clean_str(m.get("habitation_name")),
                "habitation_name_clean": plant_clean,
                "plantation_id": plant_id,
                "officer_name": _clean_str(m.get("officer_name")),
                "commune_act": _clean_str(m.get("commune")),
            }
        )

    _write_csv(rows, output_path)
    return len(rows)


# ---------------------------------------------------------------------------
# Post-processing: fill missing declaration months from neighbouring acts
# ---------------------------------------------------------------------------


def _fill_missing_months(rows: list[dict]) -> list[dict]:
    """Fill declaration_date where month is unknown using the next act's month.

    For acts where declaration_date is None but year_registry is known, look
    forward (then backward) within the same commune × year_registry group to
    borrow a month from the nearest act with a valid ISO declaration_date.
    """
    from datetime import date as _date

    rows_sorted = sorted(rows, key=lambda r: r["act_id"])

    # Index rows by (commune, year_registry) for fast neighbour lookup
    from collections import defaultdict

    group_indices: dict = defaultdict(list)
    for i, r in enumerate(rows_sorted):
        key = (r["commune"], r["year_registry"])
        group_indices[key].append(i)

    for key, indices in group_indices.items():
        year = key[1]
        if not year:
            continue
        for pos, idx in enumerate(indices):
            r = rows_sorted[idx]
            if r["declaration_date"] is not None:
                continue
            # Search forward then backward within group for a valid month
            month = None
            for offset in range(1, len(indices)):
                for direction in (1, -1):
                    neighbour_pos = pos + direction * offset
                    if 0 <= neighbour_pos < len(indices):
                        nd = rows_sorted[indices[neighbour_pos]]["declaration_date"]
                        if nd and len(nd) >= 7:
                            month = int(nd[5:7])
                            break
                if month:
                    break
            if month:
                try:
                    rows_sorted[idx]["declaration_date"] = _date(
                        year, month, 1
                    ).isoformat()
                except ValueError:
                    pass

    # Restore original order
    order = {r["act_id"]: i for i, r in enumerate(rows)}
    return sorted(rows_sorted, key=lambda r: order.get(r["act_id"], 0))


# ---------------------------------------------------------------------------
# Post-process: patch missing declaration months via CSV round-trip
# ---------------------------------------------------------------------------


def _patch_csv_missing_months(path: Path) -> None:
    """Read a cleaned CSV, fill missing declaration months, write back in place."""
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or "declaration_date" not in rows[0]:
        return
    rows = _fill_missing_months(rows)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def export_all(ner_json_path: Path, output_dir: Path) -> None:
    """Read ner_llm.json and write the three cleaned CSVs."""
    with open(ner_json_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {ner_json_path}")

    owner_lookup, plantation_lookup = build_entity_tables(records)
    print(
        f"Entity tables: {len(owner_lookup)} owners, {len(plantation_lookup)} plantations"
    )

    n_deaths = export_deaths(
        records, owner_lookup, plantation_lookup, output_dir / "ner_death.csv"
    )
    n_births = export_births(
        records, owner_lookup, plantation_lookup, output_dir / "ner_birth.csv"
    )
    n_marriages = export_marriages(
        records, owner_lookup, plantation_lookup, output_dir / "ner_marriage.csv"
    )

    # Post-process: fill missing declaration months from neighbouring acts
    for csv_name in ("ner_death.csv", "ner_birth.csv", "ner_marriage.csv"):
        _patch_csv_missing_months(output_dir / csv_name)

    print(f"Exported: {n_deaths} deaths, {n_births} births, {n_marriages} marriages")
    print(f"Output: {output_dir}")
