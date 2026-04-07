"""Audit data extraction quality on cleaned NER outputs.

Usage:
    python scripts/ner/audit_quality.py                        # stats + problems for all types
    python scripts/ner/audit_quality.py --mode sample --act-type deces --n 5
    python scripts/ner/audit_quality.py --output logs/audit_2026-04-07.md
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
CSV_DIR = BASE / "NER_datasets/llm/cleaned"
RAW_JSON = BASE / "NER_datasets/llm/ner_llm.json"

ACT_TYPES = {
    "deces": "ner_death.csv",
    "naissance": "ner_birth.csv",
    "mariage": "ner_marriage.csv",
}

# Age columns per act type
AGE_COLS = {
    "deces": ["person_age", "declarant_age"],
    "naissance": ["mother_age", "father_age", "declarant_age"],
    "mariage": ["spouse1_age", "spouse2_age", "declarant_age"],
}

# Key fields that should rarely be null
KEY_FIELDS = {
    "deces": ["person_name", "declaration_date", "death_date", "owner_name_clean"],
    "naissance": ["child_name", "declaration_date", "birth_date", "mother_name"],
    "mariage": ["spouse1_name", "spouse2_name", "declaration_date"],
}

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _fill_rate(series: pd.Series) -> float:
    return series.notna().mean()


def summary_stats(df: pd.DataFrame, act_type: str) -> str:
    """Return markdown string of summary statistics for a cleaned DataFrame."""
    lines = [f"## Summary stats — {act_type} ({len(df)} acts)\n"]

    # Fill rates
    lines.append("### Fill rates\n")
    lines.append("| Column | Non-null | Fill% |")
    lines.append("|--------|----------|-------|")
    for col in df.columns:
        n = df[col].notna().sum()
        pct = 100 * n / len(df) if len(df) else 0
        lines.append(f"| {col} | {n} | {pct:.0f}% |")
    lines.append("")

    # Numeric stats
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("### Numeric columns\n")
        lines.append("| Column | Min | Max | Mean | Median |")
        lines.append("|--------|-----|-----|------|--------|")
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                lines.append(f"| {col} | — | — | — | — |")
            else:
                lines.append(
                    f"| {col} | {s.min():.0f} | {s.max():.0f} | {s.mean():.1f} | {s.median():.0f} |"
                )
        lines.append("")

    # Categorical value counts (low-cardinality columns only)
    cat_cols = [
        c
        for c in df.select_dtypes(include="object").columns
        if df[c].nunique() <= 20 and not c.endswith("_raw") and c != "act_id"
    ]
    if cat_cols:
        lines.append("### Value counts (categorical)\n")
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False).head(10)
            lines.append(f"**{col}**\n")
            for val, cnt in counts.items():
                lines.append(f"- `{val}`: {cnt}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Problem detection
# ---------------------------------------------------------------------------


def detect_problems(df: pd.DataFrame, act_type: str) -> str:
    """Return markdown string listing detected data quality problems."""
    lines = [f"## Problems — {act_type}\n"]
    issues: list[str] = []

    # 1. Key field null rates
    for col in KEY_FIELDS.get(act_type, []):
        if col not in df.columns:
            continue
        null_rate = 1 - _fill_rate(df[col])
        if null_rate > 0.1:
            issues.append(
                f"**High null rate** `{col}`: {null_rate:.0%} null ({int(null_rate * len(df))}/{len(df)})"
            )

    # 2. Date format problems (non-null but not ISO)
    date_cols = [
        c for c in df.columns if c.endswith("_date") and not c.endswith("_raw")
    ]
    for col in date_cols:
        bad = df[col].dropna().apply(lambda v: not bool(ISO_DATE_RE.match(str(v))))
        if bad.sum() > 0:
            examples = df.loc[df[col].notna() & bad, col].head(3).tolist()
            issues.append(
                f"**Non-ISO date** `{col}`: {bad.sum()} values — e.g. {examples}"
            )

    # 3. Age outliers
    for col in AGE_COLS.get(act_type, []):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        too_low = (s < 0).sum()
        too_high = (s > 100).sum()
        if too_low:
            issues.append(f"**Negative age** `{col}`: {too_low} records")
        if too_high:
            vals = s[s > 100].tolist()[:5]
            issues.append(f"**Age > 100** `{col}`: {too_high} records — e.g. {vals}")

    # 4. act_type / marge_act_type mismatch
    if "marge_act_type" in df.columns:
        both = df[df["marge_act_type"].notna()]
        mismatch = both[both["marge_act_type"].astype(str).str.lower() != act_type]
        if len(mismatch) > 0:
            rate = len(mismatch) / len(both)
            issues.append(
                f"**marge/act_type mismatch**: {len(mismatch)} records ({rate:.0%}) — "
                f"marge says something other than '{act_type}'"
            )

    # 5. Duplicate act_ids
    dupes = df["act_id"].duplicated().sum()
    if dupes:
        issues.append(f"**Duplicate act_id**: {dupes} duplicates")

    if not issues:
        lines.append("No problems detected.")
    else:
        lines.extend(f"- {i}" for i in issues)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample observations
# ---------------------------------------------------------------------------


def _load_raw_index(raw_json: Path) -> dict[str, dict]:
    """Return act_id → raw record dict."""
    with open(raw_json, encoding="utf-8") as f:
        records = json.load(f)
    return {r["act_id"]: r for r in records}


def sample_observations(
    df: pd.DataFrame,
    raw_index: dict[str, dict],
    act_type: str,
    n: int = 5,
    commune: str | None = None,
    seed: int | None = None,
) -> str:
    """Return markdown string showing n random acts with raw text + extracted fields."""
    subset = df.copy()
    if commune:
        subset = subset[subset["commune"] == commune]
    if subset.empty:
        return f"## Samples — {act_type}\n\nNo records match the filter.\n"

    if seed is not None:
        random.seed(seed)
    sample = subset.sample(min(n, len(subset)), random_state=seed)

    lines = [f"## Samples — {act_type} (n={len(sample)})\n"]

    for _, row in sample.iterrows():
        act_id = row["act_id"]
        raw = raw_index.get(act_id, {})
        lines.append(f"### `{act_id}`\n")

        marge = (raw.get("raw_marge") or "").strip()
        plein = (raw.get("raw_plein_texte") or "").strip()
        lines.append("**Raw marge:**")
        lines.append(f"```\n{marge}\n```\n")
        lines.append("**Raw plein texte:**")
        lines.append(f"```\n{plein}\n```\n")

        lines.append("**Extracted fields:**\n")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for col in row.index:
            if col in ("act_id", "commune"):
                continue
            val = row[col]
            if pd.isna(val):
                val = ""
            lines.append(f"| {col} | {val} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_df(act_type: str, commune: str | None) -> pd.DataFrame:
    csv_path = CSV_DIR / ACT_TYPES[act_type]
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if commune:
        df = df[df["commune"] == commune]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stats", "problems", "sample", "all"],
        default="all",
        help="Which section(s) to run (default: all)",
    )
    parser.add_argument(
        "--act-type",
        choices=list(ACT_TYPES.keys()),
        default=None,
        help="Restrict to one act type (default: all types)",
    )
    parser.add_argument("--commune", default=None, help="Filter by commune")
    parser.add_argument("--n", type=int, default=5, help="Number of sample acts")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample")
    parser.add_argument("--output", default=None, help="Write report to this path")
    args = parser.parse_args()

    act_types = [args.act_type] if args.act_type else list(ACT_TYPES.keys())
    sections: list[str] = ["# NER Quality Audit\n"]

    raw_index = _load_raw_index(RAW_JSON) if args.mode in ("sample", "all") else {}

    for act_type in act_types:
        df = _load_df(act_type, args.commune)
        if df.empty:
            sections.append(f"## {act_type}: no data\n")
            continue

        if args.mode in ("stats", "all"):
            sections.append(summary_stats(df, act_type))
        if args.mode in ("problems", "all"):
            sections.append(detect_problems(df, act_type))
        if args.mode in ("sample", "all"):
            sections.append(
                sample_observations(
                    df, raw_index, act_type, args.n, args.commune, args.seed
                )
            )

    report = "\n---\n\n".join(sections)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
