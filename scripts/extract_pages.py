"""Extract only the 'pages' subfolders from commune zip files into Guadeloupe/<commune>/<year>/pages/."""

import re
import zipfile
from pathlib import Path

BASE_DIR = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\Images"
)
OUTPUT_DIR = BASE_DIR / "Guadeloupe"


def commune_name_from_zip(zip_path: Path) -> str:
    """Extract commune name from zip filename like 'abymes_OCRed_05032024.zip' -> 'abymes'."""
    stem = zip_path.stem  # e.g. abymes_OCRed_05032024
    # Remove the _OCRed_DDMMYYYY or _transk_DDMMYYYY suffix
    match = re.match(r"^(.+?)_(OCRed|transk)_\d+", stem)
    if match:
        return match.group(1)
    return stem


def extract_pages_from_zip(zip_path: Path) -> None:
    commune = commune_name_from_zip(zip_path)
    dest = OUTPUT_DIR / commune
    print(f"\n[ZIP] {zip_path.name} -> {commune}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        pages_entries = [
            n for n in zf.namelist() if "/pages/" in n and not n.endswith("/")
        ]
        print(f"   Found {len(pages_entries)} page files")

        for entry in pages_entries:
            # entry is like "1842/pages/FILE.jpg"
            out_path = dest / entry
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                continue
            with zf.open(entry) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    print(f"   [OK] Extracted to {dest}")


def main():
    zip_files = sorted(BASE_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files in {BASE_DIR}")
    for zf in zip_files:
        extract_pages_from_zip(zf)
    print("\nDone!")


if __name__ == "__main__":
    main()
