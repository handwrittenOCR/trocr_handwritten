"""Rebuild reading_order using page-aware matching and save as separate file.

Does NOT modify the original metadata.json. Instead:
1. Restores metadata.json to the original (page-unaware) reading_order
2. Saves the corrected reading_order as reading_order_fixed.json per page
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES/OCR_gem31"
)


def build_reading_order_original(metadata):
    """Original page-unaware reading order (y-overlap only, no page-side filter)."""
    marges = metadata.get("Marge", [])
    textes = metadata.get("Plein Texte", [])

    def y_center(region):
        c = region["coordinates"]
        return c["y"] + c["height"] / 2

    def y_range(region):
        c = region["coordinates"]
        return c["y"], c["y"] + c["height"]

    matched_marges = set()
    entries = []

    for texte in textes:
        t_top, t_bot = y_range(texte)
        best_marge = None
        best_overlap = 0

        for i, marge in enumerate(marges):
            if i in matched_marges:
                continue
            m_top, m_bot = y_range(marge)
            overlap = max(0, min(t_bot, m_bot) - max(t_top, m_top))
            if overlap > best_overlap:
                best_overlap = overlap
                best_marge = i

        entry = {
            "plein_texte": texte["cropped_image_name"],
            "marge": None,
            "y_center": y_center(texte),
        }
        if best_marge is not None and best_overlap > 0:
            entry["marge"] = marges[best_marge]["cropped_image_name"]
            matched_marges.add(best_marge)

        entries.append(entry)

    for i, marge in enumerate(marges):
        if i not in matched_marges:
            entries.append(
                {
                    "plein_texte": None,
                    "marge": marge["cropped_image_name"],
                    "y_center": y_center(marge),
                }
            )

    entries.sort(key=lambda e: e["y_center"])
    for idx, entry in enumerate(entries):
        entry["order"] = idx + 1

    return entries


def main():
    from trocr_handwritten.parse.utils import build_reading_order

    grand_total = 0
    grand_changed = 0

    for commune_dir in sorted(BASE.iterdir()):
        if not commune_dir.is_dir():
            continue
        for year_dir in sorted(commune_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            total = 0
            changed = 0

            for page_dir in sorted(year_dir.iterdir()):
                meta_path = page_dir / "metadata.json"
                if not meta_path.exists():
                    continue

                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

                # Build both versions from the coordinate data
                original_order = build_reading_order_original(meta)
                fixed_order = build_reading_order(meta)  # page-aware version

                total += 1

                # Restore metadata.json to original reading_order
                meta["reading_order"] = original_order
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4, ensure_ascii=False)

                # Save full metadata with corrected reading_order
                fixed_meta = dict(meta)
                fixed_meta["reading_order"] = fixed_order
                fixed_path = page_dir / "metadata_reading_order.json"
                with open(fixed_path, "w", encoding="utf-8") as f:
                    json.dump(fixed_meta, f, indent=4, ensure_ascii=False)

                if original_order != fixed_order:
                    changed += 1

            if total > 0:
                print(
                    f"{commune_dir.name}/{year_dir.name}: {changed}/{total} pages differ"
                )
                grand_total += total
                grand_changed += changed

    print(
        f"\nTotal: {grand_changed}/{grand_total} pages have different fixed reading_order"
    )
    print(
        "Original metadata.json restored, fixed version in metadata_reading_order.json"
    )


if __name__ == "__main__":
    main()
