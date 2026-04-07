"""Detect and split merged acts before NER extraction.

A single YOLO crop sometimes captures multiple consecutive civil acts.
This module detects them via the 'L'An mil huit cent' header pattern and
splits the plein_texte into separate ActRecord instances.
"""

import re
from typing import List

from trocr_handwritten.ner.schemas import ActRecord

# Matches the standard act header at the start of a line (within first 10 chars).
# "Aujourd'hui" is intentionally excluded: it appears in marriage body text and is
# already handled by split_registries() in dataset.py.
_ACT_HEADER = re.compile(
    r"(?m)^\s{0,10}l.{0,2}an\s+mil\s+huit",
    re.IGNORECASE,
)

_MIN_CHARS_PER_SEGMENT = 100
_LENGTH_THRESHOLD_FACTOR = 1.5


def _split_plein_texte(text: str) -> List[str]:
    """Split plein texte on act headers. Returns list of non-empty segments."""
    matches = list(_ACT_HEADER.finditer(text))
    if len(matches) <= 1:
        return [text]

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if len(segment) >= _MIN_CHARS_PER_SEGMENT:
            segments.append(segment)

    return segments if len(segments) > 1 else [text]


def split_merged_acts(records: List[ActRecord]) -> List[ActRecord]:
    """Expand any merged ActRecords into individual ones.

    Merged acts keep the original marge on the first segment only.
    Split acts get act_ids suffixed with _split1, _split2, etc.
    Original single-act records are returned unchanged.
    """
    result: List[ActRecord] = []
    n_split = 0

    lengths = [len(r.plein_texte_text) for r in records]
    if lengths:
        lengths_sorted = sorted(lengths)
        median_len = lengths_sorted[len(lengths_sorted) // 2]
    else:
        median_len = 0
    length_cutoff = median_len * _LENGTH_THRESHOLD_FACTOR

    for record in records:
        if len(record.plein_texte_text) < length_cutoff:
            result.append(record)
            continue
        segments = _split_plein_texte(record.plein_texte_text)

        if len(segments) == 1:
            result.append(record)
            continue

        n_split += 1
        print(
            f"  Splitting {record.act_id} ({len(record.plein_texte_text)} chars) → {len(segments)} acts"
        )
        for line in record.plein_texte_text[:1000].splitlines():
            print(f"    {line[:120]}")
        for i, segment in enumerate(segments):
            new_id = f"{record.act_id}_split{i + 1}"
            marge = record.marge_text if i == 0 else ""
            result.append(
                record.model_copy(
                    update={
                        "act_id": new_id,
                        "plein_texte_text": segment,
                        "marge_text": marge,
                    }
                )
            )

    if n_split:
        print(f"Split {n_split} merged acts → {len(result)} total records")

    return result
