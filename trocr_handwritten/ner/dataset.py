"""Build a structured dataset of civil acts from YOLO crop transcriptions.

Reads metadata.json + .md transcription files from the OCR output directory,
pairs Marge and Plein Texte crops using reading_order, detects act type,
and produces one ActRecord per act.

Usage:
    python -m trocr_handwritten.ner.dataset \
        --input_dir <transcription_dir> \
        --output_dir <output_dir> \
        --commune abymes --year 1842
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from trocr_handwritten.ner.schemas import ActRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Act type detection from Marge text
# ---------------------------------------------------------------------------

_ACT_TYPE_PATTERNS = [
    (re.compile(r"d[ée]c[eè]s", re.IGNORECASE), "deces"),
    (re.compile(r"naissance", re.IGNORECASE), "naissance"),
    (re.compile(r"mariage", re.IGNORECASE), "mariage"),
]

_ACT_NUMBER_PATTERN = re.compile(r"(?:n[°ᵒo˚]|16°)\s*(\d+)", re.IGNORECASE)

# "Aujourd'hui" pattern: marks the start of each registry entry in gosier-style
# registers where multiple entries appear in a single crop without preamble.
_AUJOURDHUI_PATTERN = re.compile(
    r"(?=(?:^|\n)\s*[Aa]ujourd\s*'?\s*hui\b)", re.IGNORECASE
)

_AUJOURDHUI_PATTERN_NO_H = re.compile(r"(?=(?:^|\n)\s*[Aa]ujourd)", re.IGNORECASE)


# Preamble pattern: detects the start of a new act.
# All acts begin with "L'An mil huit cent ..." with OCR variations:
#   - L'/L'/l' or missing entirely ("An Mil...")
#   - Capitalization varies (Mil/mil, Huit/huit)
#   - Punctuation varies (huit-cent, huit cent)
#   - Optional prefix before L'An (e.g. "V: David\nL'An...")
_PREAMBLE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:.*\n\s*)?L['\u2019]\s*[Aa]n\s+[Mm]il\s+huit[\s\-]cent",
    re.IGNORECASE,
)
_PREAMBLE_PATTERN_NO_L = re.compile(
    r"^[Aa]n\s+[Mm]il\s+huit[\s\-]cent",
    re.IGNORECASE,
)
# Loose preamble: "L'An" anywhere in first 50 chars, with "mil" in first 100 chars
# Catches garbled OCR like "L'An Aujourd'hui mil huit cent quarante mil trois"
_PREAMBLE_PATTERN_LOOSE = re.compile(
    r"[Ll]['\u2019]\s*[Aa]n\s+",
    re.IGNORECASE,
)
# "Mêmes jour et an que dessus" — shorthand preamble (sainte_anne style)
_MEMES_JOUR_PATTERN = re.compile(r"^[Mm][êe]mes?\s+jour", re.IGNORECASE)
# "Pardevant nous" at start — act opening without date preamble
_PARDEVANT_PATTERN = re.compile(r"^[Pp]ardevant\s+[Nn]ous", re.IGNORECASE)
# Registry header page (not an act)
_REGISTRY_HEADER_PATTERN = re.compile(r"r[ée]gistre\s+contenant", re.IGNORECASE)


def is_new_act(plein_texte_text: str) -> bool:
    """Check if a Plein Texte crop is the start of a new act (vs. a continuation).

    A new act starts with the formulaic preamble 'L'An mil huit cent...' or 'aujourd'hui'justt
    A continuation is any text that does not start with this preamble.
    Registry header pages are treated as new (standalone) records.
    """
    # Strip OCR strikethrough markers before checking
    text = re.sub(r"~~[^~]*~~\s*", "", plein_texte_text).strip()
    if _PREAMBLE_PATTERN.match(text) or _AUJOURDHUI_PATTERN.match(text):
        return True
    if _PREAMBLE_PATTERN_NO_L.match(text) or _AUJOURDHUI_PATTERN_NO_H.match(text):
        return True
    # Loose match: "L'An" in first 50 chars + "mil" in first 100 chars
    # Catches garbled OCR from reconstructed copies
    if _PREAMBLE_PATTERN_LOOSE.search(text[:50]) and re.search(
        r"mil", text[:100], re.IGNORECASE
    ):
        return True
    if _REGISTRY_HEADER_PATTERN.search(text[:200]):
        return True
    # "Mêmes jour et an que dessus" — shorthand for same date
    if _MEMES_JOUR_PATTERN.match(text):
        return True
    # "Pardevant nous" at start — act opening without date preamble
    if _PARDEVANT_PATTERN.match(text):
        return True
    # "Mil huit cent" without "L'An" prefix
    if re.match(r"^[Mm]il\s+huit[\s\-]cent", text):
        return True
    return False


def detect_act_type(marge_text: str) -> str:
    """Detect act type from Marge text. Returns 'deces', 'naissance', 'mariage', or 'unknown'."""
    for pattern, act_type in _ACT_TYPE_PATTERNS:
        if pattern.search(marge_text):
            return act_type
    return "unknown"


def extract_act_number(marge_text: str) -> Optional[str]:
    """Extract act number from Marge text (e.g. 'N° 2' -> '2')."""
    match = _ACT_NUMBER_PATTERN.search(marge_text)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Deduplication: remove overlapping YOLO crops, keep the largest
# ---------------------------------------------------------------------------


def _iou(box_a: dict, box_b: dict) -> float:
    """Compute intersection-over-union for two XYWH coordinate dicts."""
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["width"], ay1 + box_a["height"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["width"], by1 + box_b["height"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(small: dict, large: dict) -> float:
    """Fraction of 'small' box area that is inside 'large' box."""
    sx1, sy1 = small["x"], small["y"]
    sx2, sy2 = sx1 + small["width"], sy1 + small["height"]
    lx1, ly1 = large["x"], large["y"]
    lx2, ly2 = lx1 + large["width"], ly1 + large["height"]

    ix1, iy1 = max(sx1, lx1), max(sy1, ly1)
    ix2, iy2 = min(sx2, lx2), min(sy2, ly2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_small = small["width"] * small["height"]
    return inter / area_small if area_small > 0 else 0.0


def deduplicate_crops(metadata: dict, containment_threshold: float = 0.8) -> List[dict]:
    """Remove overlapping Plein Texte crops, keeping the largest.

    When a small crop is largely contained inside a larger one (>80% overlap),
    the small crop is dropped as redundant.

    Returns a deduplicated reading_order list.
    """
    plein_texte_list = metadata.get("Plein Texte", [])
    if not plein_texte_list:
        return metadata.get("reading_order", [])

    # Build a lookup: jpg filename -> coordinates
    coords = {}
    for crop in plein_texte_list:
        coords[crop["cropped_image_name"]] = crop["coordinates"]

    reading_order = metadata.get("reading_order", [])
    if not reading_order:
        return []

    # Sort crops by area (largest first) for containment checks
    crops_by_area = sorted(
        [(e, coords.get(e.get("plein_texte", ""), {})) for e in reading_order],
        key=lambda x: x[1].get("width", 0) * x[1].get("height", 0),
        reverse=True,
    )

    keep = []
    removed_jpgs = set()

    for entry, box in crops_by_area:
        jpg = entry.get("plein_texte", "")
        if not box or jpg in removed_jpgs:
            continue

        # Check if this crop is mostly contained in any already-kept larger crop
        is_redundant = False
        for kept_entry, kept_box in keep:
            if _containment(box, kept_box) > containment_threshold:
                is_redundant = True
                logger.debug(
                    "Dropping redundant crop %s (%.0f%% inside %s)",
                    jpg,
                    _containment(box, kept_box) * 100,
                    kept_entry.get("plein_texte"),
                )
                removed_jpgs.add(jpg)
                break

        if not is_redundant:
            keep.append((entry, box))

    # Return in original reading order (sorted by order field)
    kept_entries = [e for e, _ in keep]
    kept_entries.sort(key=lambda e: e.get("order", 0))

    if removed_jpgs:
        logger.info(
            "Deduplicated: removed %d redundant crops (%s)",
            len(removed_jpgs),
            ", ".join(sorted(removed_jpgs)),
        )

    return kept_entries


# ---------------------------------------------------------------------------
# Split multi-entry text into individual registries
# ---------------------------------------------------------------------------


def split_registries(text: str) -> List[str]:
    """Split a transcription containing multiple registry entries.

    Each entry starts with 'Aujourd'hui'. If the text contains multiple
    occurrences, split into one string per entry. If only one or zero,
    return the full text as a single-element list.
    """
    parts = _AUJOURDHUI_PATTERN.split(text)
    # Filter out empty/whitespace-only parts
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        return [text.strip()]

    return parts


# ---------------------------------------------------------------------------
# Read a .md transcription file
# ---------------------------------------------------------------------------


def _read_md_file(page_dir: Path, subfolder: str, jpg_filename: str) -> Optional[str]:
    """Read the .md transcription corresponding to a .jpg crop filename."""
    md_filename = jpg_filename.replace(".jpg", ".md")
    md_path = page_dir / subfolder / md_filename
    if md_path.exists():
        return md_path.read_text(encoding="utf-8").strip()
    logger.warning("Missing transcription: %s", md_path)
    return None


# ---------------------------------------------------------------------------
# Build dataset from a single transcription directory
# ---------------------------------------------------------------------------


def build_dataset(
    input_dir: str,
    commune: str,
    year: str,
) -> Tuple[List[ActRecord], int]:
    """Walk page folders and reconstruct acts from reading_order.

    Args:
        input_dir: Path to the transcription directory (e.g. .../abymes/1842).
        commune: Commune name for act IDs.
        year: Year for act IDs.

    Returns:
        Tuple of (list of ActRecord, plein_texte_count).
    """
    input_path = Path(input_dir)
    records: List[ActRecord] = []
    continuation_count = 0
    plein_texte_count = 0

    # Collect and sort page folders
    page_dirs = sorted(
        [
            d
            for d in input_path.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
    )

    if not page_dirs:
        logger.warning("No page folders with metadata.json found in %s", input_dir)
        return records, 0

    logger.info("Found %d page folders in %s", len(page_dirs), input_dir)

    split_count = 0

    for page_dir in page_dirs:
        page_name = page_dir.name

        # Load metadata (prefer corrected version if available)
        reading_order_path = page_dir / "metadata_reading_order.json"
        metadata_path = (
            reading_order_path
            if reading_order_path.exists()
            else page_dir / "metadata.json"
        )
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        # Deduplicate overlapping crops
        reading_order = deduplicate_crops(metadata)
        if not reading_order:
            logger.debug("No reading_order in %s (cover/header page)", page_name)
            continue

        plein_texte_count += sum(1 for e in reading_order if e.get("plein_texte"))

        for entry in reading_order:
            order = entry.get("order", 0)
            plein_texte_jpg = entry.get("plein_texte")
            marge_jpg = entry.get("marge")  # can be None

            # Read Plein Texte (required)
            if not plein_texte_jpg:
                logger.warning(
                    "No plein_texte in reading_order entry %s order %d",
                    page_name,
                    order,
                )
                continue

            plein_texte_text = _read_md_file(page_dir, "Plein Texte", plein_texte_jpg)
            if plein_texte_text is None:
                continue

            # Read Marge (optional)
            marge_text = ""
            source_marge_file = None
            if marge_jpg:
                marge_content = _read_md_file(page_dir, "Marge", marge_jpg)
                if marge_content:
                    marge_text = marge_content
                    source_marge_file = marge_jpg.replace(".jpg", ".md")

            # Split multi-entry transcriptions into individual registries
            registry_entries = split_registries(plein_texte_text)

            for sub_idx, entry_text in enumerate(registry_entries):
                # Check if this is a new act or a continuation of the previous one.
                # A new act starts with the preamble "L'An mil huit cent..."
                # A continuation is any crop whose text does NOT start with the preamble.
                if not is_new_act(entry_text) and records:
                    # Only merge as continuation if there's a single entry
                    # (multi-entry splits are always standalone)
                    if len(registry_entries) == 1:
                        prev = records[-1]
                        records[-1] = prev.model_copy(
                            update={
                                "plein_texte_text": prev.plein_texte_text
                                + "\n"
                                + entry_text,
                            }
                        )
                        continuation_count += 1
                        logger.debug(
                            "Continuation merged: %s order %d -> %s",
                            page_name,
                            order,
                            prev.act_id,
                        )
                        continue

                # Build act ID (add sub-index when a crop was split)
                if len(registry_entries) > 1:
                    act_id = f"{commune}_{year}_{page_name}_order{order}_entry{sub_idx}"
                    split_count += 1
                else:
                    act_id = f"{commune}_{year}_{page_name}_order{order}"

                record = ActRecord(
                    act_id=act_id,
                    marge_text=marge_text,
                    plein_texte_text=entry_text,
                    source_page=page_name,
                    source_marge_file=source_marge_file,
                    source_plein_texte_file=plein_texte_jpg.replace(".jpg", ".md"),
                    commune=commune,
                    year=year,
                    order_on_page=order,
                )
                if _REGISTRY_HEADER_PATTERN.search(entry_text[:200]):
                    logger.debug("Skipping preamble act: %s", act_id)
                    continue
                records.append(record)

    if continuation_count:
        logger.info(
            "Merged %d continuation crops into previous acts", continuation_count
        )
    if split_count:
        logger.info(
            "Split multi-entry crops into %d individual registry entries",
            split_count,
        )

    logger.info(
        "Built %d acts from %d plein_texte regions (%d continuations merged, %d splits)",
        len(records),
        plein_texte_count,
        continuation_count,
        split_count,
    )
    return records, plein_texte_count


# ---------------------------------------------------------------------------
# Save dataset
# ---------------------------------------------------------------------------


def save_dataset(records: List[ActRecord], output_dir: str) -> Tuple[Path, Path]:
    """Save act records as JSON and CSV.

    Returns:
        Tuple of (json_path, csv_path).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_path / "acts_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = output_path / "acts_dataset.csv"
    if records:
        fieldnames = list(records[0].model_dump().keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.model_dump())

    logger.info("Saved dataset: %s (%d records)", json_path, len(records))
    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build structured act dataset from OCR transcriptions."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to transcription directory (e.g. .../OCR_gem31/abymes/1842)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to input_dir)",
    )
    parser.add_argument("--commune", type=str, default="abymes")
    parser.add_argument("--year", type=str, default="1842")

    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    records, pt_count = build_dataset(args.input_dir, args.commune, args.year)
    if records:
        json_path, csv_path = save_dataset(records, output_dir)
        print(f"Dataset saved: {json_path} ({len(records)} acts  {pt_count} crops)")

        # Print summary
        from collections import Counter

        type_counts = Counter(r.act_type for r in records)
        for act_type, count in type_counts.most_common():
            print(f"  {act_type}: {count}")
    else:
        print("No acts found.")


if __name__ == "__main__":
    main()
