"""Display a cost report from logs/api_costs.jsonl, grouped by day and model.

Usage:
    python scripts/cost_report.py                     # all days
    python scripts/cost_report.py 2026-04-03           # specific day
    python scripts/cost_report.py 2026-03-24 2026-04-03  # multiple days
    python scripts/cost_report.py --last 3             # last 3 days
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path("logs/api_costs.jsonl")

# Pricing table (EUR per 1M tokens) — mirrors cost_tracker.py
PRICING = {
    "gemini-2.0-flash": {"input": 0.085, "output": 0.339},
    "gemini-2.0-flash-lite": {"input": 0.064, "output": 0.254},
    "gemini-2.5-pro": {"input": 1.060, "output": 8.482},
    "gemini-2.5-flash-lite": {"input": 0.085, "output": 0.339},
    "gemini-3-pro-preview": {"input": 1.696, "output": 10.178},
    "gemini-3-flash-preview": {"input": 0.424, "output": 2.545},
    "gemini-3.1-pro-preview": {"input": 1.696, "output": 10.178},
    "gemini-3.1-flash-lite-preview": {"input": 0.212, "output": 1.272},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "mistral-large-latest": {"input": 0.50, "output": 1.50},
}


def load_entries(log_path: Path) -> list[dict]:
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Normalize: compute EUR costs if only old USD format
            if "cost_total_eur" not in entry:
                model = entry.get("model", "")
                pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
                inp = entry.get("input_tokens", 0)
                out = entry.get("output_tokens", 0)
                think = entry.get("thinking_tokens", 0)
                entry["cost_input_eur"] = (inp / 1_000_000) * pricing["input"]
                entry["cost_output_eur"] = (out / 1_000_000) * pricing["output"]
                entry["cost_thinking_eur"] = (think / 1_000_000) * pricing["output"]
                entry["cost_total_eur"] = (
                    entry["cost_input_eur"]
                    + entry["cost_output_eur"]
                    + entry["cost_thinking_eur"]
                )
                entry["cost_total_adjusted_eur"] = entry["cost_total_eur"] * 1.30
            if "cost_total_adjusted_eur" not in entry:
                entry["cost_total_adjusted_eur"] = entry["cost_total_eur"] * 1.30
            if "thinking_tokens" not in entry:
                entry["thinking_tokens"] = 0
            entries.append(entry)
    return entries


def aggregate(entries: list[dict]) -> dict:
    """Group by (date, model) and sum."""
    agg = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cost_input_eur": 0.0,
            "cost_output_eur": 0.0,
            "cost_thinking_eur": 0.0,
            "cost_total_eur": 0.0,
            "cost_total_adjusted_eur": 0.0,
        }
    )
    for e in entries:
        day = e["timestamp"][:10]
        model = e.get("model", "unknown")
        key = (day, model)
        a = agg[key]
        a["calls"] += e.get("total_calls", 0)
        a["input_tokens"] += e.get("input_tokens", 0)
        a["output_tokens"] += e.get("output_tokens", 0)
        a["thinking_tokens"] += e.get("thinking_tokens", 0)
        a["cost_input_eur"] += e.get("cost_input_eur", 0.0)
        a["cost_output_eur"] += e.get("cost_output_eur", 0.0)
        a["cost_thinking_eur"] += e.get("cost_thinking_eur", 0.0)
        a["cost_total_eur"] += e.get("cost_total_eur", 0.0)
        a["cost_total_adjusted_eur"] += e.get("cost_total_adjusted_eur", 0.0)
    return agg


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def print_table(agg: dict, days_filter: list[str] | None = None) -> None:
    keys = sorted(agg.keys())
    if days_filter:
        keys = [(d, m) for d, m in keys if d in days_filter]

    if not keys:
        print("No data for the selected days.")
        return

    # Header
    print()
    print(
        f"{'Date':<12} {'Model':<28} {'Calls':>6} {'Input':>9} {'Output':>9} "
        f"{'Think':>9} {'EUR In':>8} {'EUR Out':>8} {'EUR Thk':>8} "
        f"{'Total':>9} {'Adj 1.3x':>9}"
    )
    print("-" * 130)

    current_day = None
    day_total = 0.0
    day_adj = 0.0
    grand_total = 0.0
    grand_adj = 0.0

    for day, model in keys:
        a = agg[(day, model)]

        # Day separator
        if current_day and current_day != day:
            print(f"{'':>99} {'-'*9} {'-'*9}")
            print(f"{'':>89} Day: {day_total:>8.2f}  {day_adj:>8.2f}")
            print()
            day_total = 0.0
            day_adj = 0.0

        current_day = day
        day_total += a["cost_total_eur"]
        day_adj += a["cost_total_adjusted_eur"]
        grand_total += a["cost_total_eur"]
        grand_adj += a["cost_total_adjusted_eur"]

        print(
            f"{day:<12} {model:<28} {a['calls']:>6} "
            f"{fmt_tokens(a['input_tokens']):>9} {fmt_tokens(a['output_tokens']):>9} "
            f"{fmt_tokens(a['thinking_tokens']):>9} "
            f"{a['cost_input_eur']:>8.4f} {a['cost_output_eur']:>8.4f} "
            f"{a['cost_thinking_eur']:>8.4f} "
            f"{a['cost_total_eur']:>9.4f} {a['cost_total_adjusted_eur']:>9.4f}"
        )

    # Last day subtotal
    print(f"{'':>99} {'-'*9} {'-'*9}")
    print(f"{'':>89} Day: {day_total:>8.2f}  {day_adj:>8.2f}")
    print()
    print("=" * 130)
    print(f"{'GRAND TOTAL':>100} {grand_total:>8.2f}  {grand_adj:>8.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Cost report from api_costs.jsonl")
    parser.add_argument("days", nargs="*", help="Specific dates (YYYY-MM-DD)")
    parser.add_argument("--last", type=int, default=0, help="Show last N days")
    parser.add_argument(
        "--log", type=str, default=str(LOG_PATH), help="Path to JSONL log"
    )
    args = parser.parse_args()

    entries = load_entries(Path(args.log))
    agg = aggregate(entries)

    all_days = sorted({k[0] for k in agg})

    if args.last > 0:
        days_filter = all_days[-args.last :]
    elif args.days:
        days_filter = args.days
    else:
        days_filter = None  # show all

    print_table(agg, days_filter)


if __name__ == "__main__":
    main()
