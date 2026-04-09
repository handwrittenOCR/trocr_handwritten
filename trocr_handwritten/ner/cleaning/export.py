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

from trocr_handwritten.ner.cleaning.ages import parse_age, parse_age_validated
from trocr_handwritten.ner.cleaning.dates import parse_act_dates
from trocr_handwritten.ner.cleaning.entities import (
    build_entity_tables,
    load_clusters_lookup,
)

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
                "declarant_age": parse_age_validated(
                    d.get("declarant_age"), min_age=21
                ),
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
                "mother_age": parse_age_validated(mother.get("age"), min_age=15),
                "mother_occupation": _clean_str(mother.get("occupation")),
                "mother_registration_register": _clean_str(
                    mother.get("registration_register")
                ),
                "mother_registration_number": _clean_str(
                    mother.get("registration_number")
                ),
                "father_name": _clean_str(father.get("name")),
                "father_age_raw": _clean_str(father.get("age")),
                "father_age": parse_age_validated(father.get("age"), min_age=15),
                "birth_date_raw": _clean_str(b.get("birth_date")),
                "birth_date": event_iso,
                "birth_place": _clean_str(b.get("birth_place")),
                "declaration_date_raw": _clean_str(b.get("declaration_date")),
                "declaration_date": decl_iso,
                "declarant_name": _clean_str(b.get("declarant_name")),
                "declarant_age_raw": _clean_str(b.get("declarant_age")),
                "declarant_age": parse_age_validated(
                    b.get("declarant_age"), min_age=21
                ),
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
                "spouse1_age": parse_age_validated(s1.get("age"), min_age=15),
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
                "spouse2_age": parse_age_validated(s2.get("age"), min_age=15),
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
                "declarant_age": parse_age_validated(
                    m.get("declarant_age"), min_age=21
                ),
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


def _is_missing(val) -> bool:
    """True if a date value is absent or not a full ISO date."""
    return not val or str(val).strip() in ("", "null", "none")


def _is_full_iso(val) -> bool:
    """True if val is a complete YYYY-MM-DD ISO date string."""
    v = str(val).strip() if val else ""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", v))


def _fill_declaration_dates(rows: list[dict]) -> list[dict]:
    """Ensure every row has a full declaration_date.

    For rows where declaration_date is missing or not full ISO, borrow the
    month from the nearest neighbour within the same commune × year_registry
    group, then set day=1.
    """
    from collections import defaultdict
    from datetime import date as _date

    rows_sorted = sorted(rows, key=lambda r: r["act_id"])

    group_indices: dict = defaultdict(list)
    for i, r in enumerate(rows_sorted):
        key = (r["commune"], r.get("year_registry"))
        group_indices[key].append(i)

    for key, indices in group_indices.items():
        year = key[1]
        if not year:
            continue
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        for pos, idx in enumerate(indices):
            r = rows_sorted[idx]
            if _is_full_iso(r.get("declaration_date")):
                continue
            month = None
            for offset in range(1, len(indices)):
                for direction in (1, -1):
                    np_ = pos + direction * offset
                    if 0 <= np_ < len(indices):
                        nd = rows_sorted[indices[np_]].get("declaration_date")
                        if _is_full_iso(nd):
                            month = int(str(nd)[5:7])
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

    order = {r["act_id"]: i for i, r in enumerate(rows)}
    return sorted(rows_sorted, key=lambda r: order.get(r["act_id"], 0))


def _normalise_years(rows: list[dict], event_col: str) -> list[dict]:
    """Normalise years in declaration_date and event_date using folder_year.

    - declaration_date year is always replaced by folder_year.
    - event_date year is folder_year, except when month=12 and the act falls
      within the first 10% of acts in the commune × folder_year group
      → folder_year - 1.
    """
    from collections import defaultdict
    import math

    rows_sorted = sorted(rows, key=lambda r: r["act_id"])

    group_indices: dict = defaultdict(list)
    for i, r in enumerate(rows_sorted):
        try:
            folder_year = int(str(r["act_id"]).split("_")[1])
        except (IndexError, ValueError):
            continue
        group_indices[(r["commune"], folder_year)].append(i)

    for (commune, folder_year), indices in group_indices.items():
        cutoff = max(1, math.ceil(len(indices) * 0.10))

        for idx in indices:
            d = rows_sorted[idx].get("declaration_date")
            if _is_full_iso(d):
                rows_sorted[idx][
                    "declaration_date"
                ] = f"{folder_year:04d}-{str(d)[5:7]}-{str(d)[8:10]}"

        for pos, idx in enumerate(indices):
            ev = rows_sorted[idx].get(event_col)
            if not _is_full_iso(ev):
                continue
            month = int(str(ev)[5:7])
            day = int(str(ev)[8:10])
            year = folder_year - 1 if (month == 12 and pos < cutoff) else folder_year
            try:
                from datetime import date as _d

                _d(year, month, day)
                rows_sorted[idx][event_col] = f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass  # keep original year if new year makes date invalid (e.g. Feb 29)

    order = {r["act_id"]: i for i, r in enumerate(rows)}
    return sorted(rows_sorted, key=lambda r: order.get(r["act_id"], 0))


def _fill_event_dates(rows: list[dict], event_col: str) -> list[dict]:
    """Set event_date to declaration_date wherever event_date is missing or year-only."""
    for r in rows:
        if not _is_full_iso(r.get(event_col)):
            r[event_col] = r.get("declaration_date")
    return rows


def _fix_declaration_month_order(rows: list[dict], n_neighbors: int = 5) -> list[dict]:
    """Fix declaration months that break ascending order within commune × year group.

    Acts are sorted by act_id (document order). When a month is strictly less
    than the previous act's month, replace it with the most common month among
    the nearest n_neighbors on each side. The original day is preserved.
    """
    from collections import defaultdict, Counter
    from datetime import date as _date

    rows_sorted = sorted(rows, key=lambda r: r["act_id"])

    group_indices: dict = defaultdict(list)
    for i, r in enumerate(rows_sorted):
        group_indices[(r["commune"], r.get("year_registry"))].append(i)

    for indices in group_indices.values():
        months = [
            (
                int(str(rows_sorted[idx]["declaration_date"])[5:7])
                if _is_full_iso(rows_sorted[idx].get("declaration_date"))
                else None
            )
            for idx in indices
        ]

        prev_month = None
        for pos, idx in enumerate(indices):
            m = months[pos]
            if m is None:
                continue
            if prev_month is not None and m < prev_month:
                neighbor_months = []
                for offset in range(1, n_neighbors + 1):
                    for direction in (1, -1):
                        np_ = pos + direction * offset
                        if 0 <= np_ < len(indices) and months[np_] is not None:
                            neighbor_months.append(months[np_])
                if neighbor_months:
                    valid = [mo for mo in neighbor_months if mo >= prev_month]
                    best_month = Counter(valid or neighbor_months).most_common(1)[0][0]
                    d = str(rows_sorted[idx]["declaration_date"])
                    year, day = d[:4], d[8:10]
                    try:
                        _date(int(year), best_month, int(day))
                        rows_sorted[idx][
                            "declaration_date"
                        ] = f"{year}-{best_month:02d}-{day}"
                    except ValueError:
                        rows_sorted[idx][
                            "declaration_date"
                        ] = f"{year}-{best_month:02d}-01"
                    months[pos] = best_month
            else:
                prev_month = m

    order = {r["act_id"]: i for i, r in enumerate(rows)}
    return sorted(rows_sorted, key=lambda r: order.get(r["act_id"], 0))


def _fix_event_date_anomalies(rows: list[dict], event_col: str) -> list[dict]:
    """Set event_date to declaration_date when event is after declaration or > 60 days before."""
    from datetime import date as _date

    for r in rows:
        decl = r.get("declaration_date")
        ev = r.get(event_col)
        if not _is_full_iso(decl) or not _is_full_iso(ev):
            continue
        try:
            decl_d = _date.fromisoformat(str(decl))
            ev_d = _date.fromisoformat(str(ev))
        except ValueError:
            r[event_col] = str(decl)
            continue
        if ev_d > decl_d or (decl_d - ev_d).days > 60:
            r[event_col] = str(decl)
    return rows


# ---------------------------------------------------------------------------
# Post-process: patch missing declaration months via CSV round-trip
# ---------------------------------------------------------------------------


def _patch_dates(path: Path, event_col: str) -> None:
    """Read a cleaned CSV, fill declaration_date then event_date, write back in place."""
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows:
        return
    rows = _fill_declaration_dates(rows)
    if event_col and event_col in rows[0]:
        rows = _fill_event_dates(rows, event_col)
    rows = _normalise_years(rows, event_col)
    rows = _fix_declaration_month_order(rows)
    if event_col and event_col in rows[0]:
        rows = _fix_event_date_anomalies(rows, event_col)
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


def export_all(
    ner_json_path: Path,
    output_dir: Path,
    clusters_json_path: Optional[Path] = None,
) -> None:
    """Read ner_llm.json and write the three cleaned CSVs.

    Args:
        clusters_json_path: Path to {commune}_clusters.json produced by Claude,
            containing both owner and plantation clusters (distinguished by "type" field).
            When provided, overrides fuzzy auto-clustering for the matching entity types.
            Note: owner and plantation are independent — one owner may appear across multiple
            plantations, and one plantation may have had multiple owners over time.
    """
    with open(ner_json_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {ner_json_path}")

    if clusters_json_path and clusters_json_path.exists():
        owner_lookup = load_clusters_lookup(clusters_json_path, entity_type="owner")
        plantation_lookup = load_clusters_lookup(
            clusters_json_path, entity_type="plantation"
        )
        print(
            f"Entity tables (Claude clusters): {len(owner_lookup)} owners, "
            f"{len(plantation_lookup)} plantations"
        )
    else:
        owner_lookup, plantation_lookup = build_entity_tables(records)
        print(
            f"Entity tables (fuzzy): {len(owner_lookup)} owners, {len(plantation_lookup)} plantations"
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

    _patch_dates(output_dir / "ner_death.csv", "death_date")
    _patch_dates(output_dir / "ner_birth.csv", "birth_date")
    _patch_dates(output_dir / "ner_marriage.csv", "marriage_date")

    print(f"Exported: {n_deaths} deaths, {n_births} births, {n_marriages} marriages")
    print(f"Output: {output_dir}")
