"""Detect and split merged acts before NER extraction.

A single YOLO crop sometimes captures multiple consecutive civil acts.
This module detects them via the 'L'An mil huit cent' header pattern and
splits the plein_texte into separate ActRecord instances.
"""

import re
from typing import List

from trocr_handwritten.ner.schemas import ActRecord

# Matches the standard act header: "L'An mil huit cent ..." or "L An mil huit cent ..."
# Handles OCR variants: L An / L'An / l an, uppercase/lowercase
_ACT_HEADER = re.compile(
    r"(?<!\w)l.{0,2}an\s+mil\s+huit\s+cent",
    re.IGNORECASE,
)

# A merged act is one where the header appears more than once
_MIN_CHARS_PER_SEGMENT = 100


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

    for record in records:
        segments = _split_plein_texte(record.plein_texte_text)

        if len(segments) == 1:
            result.append(record)
            continue

        n_split += 1
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
