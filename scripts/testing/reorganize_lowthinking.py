"""
Reorganize test_lowthinking folder:
- old/ subfolder: old transcriptions (HIGH thinking)
- new/ subfolder: new transcriptions (LOW thinking)
- images/ subfolder: source crop images
"""

import shutil
from pathlib import Path

BASE = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes\1841"
)
OUTPUT_DIR = Path("test_lowthinking")

# Collect all .md files currently in test_lowthinking (these are the NEW transcriptions)
new_mds = list(OUTPUT_DIR.rglob("*.md"))
new_mds = [p for p in new_mds if p.name != "comparison.md"]

for new_md in new_mds:
    # Relative path from OUTPUT_DIR, e.g. FRAD971_.../Plein Texte/000.md
    rel = new_md.relative_to(OUTPUT_DIR)
    parts = rel.parts  # (page, region_type, filename)
    page, region_type, filename = parts[0], parts[1], parts[2]
    stem = Path(filename).stem

    # Source paths
    old_md = BASE / page / region_type / filename
    jpg = BASE / page / region_type / f"{stem}.jpg"

    # Target paths
    new_target = OUTPUT_DIR / "new" / page / region_type / filename
    old_target = OUTPUT_DIR / "old" / page / region_type / filename
    img_target = OUTPUT_DIR / "images" / page / region_type / f"{stem}.jpg"

    # Create dirs and copy
    new_target.parent.mkdir(parents=True, exist_ok=True)
    old_target.parent.mkdir(parents=True, exist_ok=True)
    img_target.parent.mkdir(parents=True, exist_ok=True)

    # Move new transcription
    shutil.copy2(new_md, new_target)

    # Copy old transcription
    if old_md.exists():
        shutil.copy2(old_md, old_target)
    else:
        old_target.write_text("(no old transcription)", encoding="utf-8")

    # Copy image
    if jpg.exists():
        shutil.copy2(jpg, img_target)

    print(f"  {page}/{region_type}/{filename}")

# Clean up old flat structure (remove page dirs at top level, keep comparison.md + new/old/images)
for new_md in new_mds:
    new_md.unlink()

# Remove now-empty page directories at top level
for d in OUTPUT_DIR.iterdir():
    if d.is_dir() and d.name not in ("new", "old", "images"):
        shutil.rmtree(d)

print(f"\nReorganized into: {OUTPUT_DIR}/old/, {OUTPUT_DIR}/new/, {OUTPUT_DIR}/images/")
