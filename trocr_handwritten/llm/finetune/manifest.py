import json
from pathlib import Path


def write_manifest(path, records):
    """Write records as one JSON object per line (jsonl manifest)."""
    Path(path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


def proportional_take(strata, size):
    """
    Select `size` records across pre-ordered strata, preserving proportions.

    Each stratum contributes round(size * len / total) of its leading items, then
    the selection is topped up to the exact size from remaining items in stratum
    order. Callers shuffle the strata beforehand; reusing the same shuffled strata
    across increasing sizes yields nested subsets.

    Args:
        strata: Mapping of stratum key -> ordered list of records.
        size: Target number of records.

    Returns:
        list: The selected records.
    """
    total = sum(len(v) for v in strata.values())
    if total == 0:
        return []
    size = min(size, total)
    chosen, seen = [], set()
    for items in strata.values():
        k = round(size * len(items) / total)
        for r in items[:k]:
            chosen.append(r)
            seen.add(id(r))
    if len(chosen) >= size:
        return chosen[:size]
    for items in strata.values():
        for r in items:
            if id(r) not in seen:
                chosen.append(r)
                seen.add(id(r))
                if len(chosen) == size:
                    return chosen
    return chosen
