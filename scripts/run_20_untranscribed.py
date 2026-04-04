"""
Run the full pipeline (YOLO crop + Gemini 3.1 Pro OCR) on 20 untranscribed pages.

Picks the first 20 pages from gosier that have no transcription output yet.
Steps:
  1. Scan gosier/pages/ for .jpg files without a matching folder in the output dir
  2. Copy the first 20 to a temp input dir (so YOLO runs only on those)
  3. Run YOLO layout parser to crop Marge + Plein Texte regions
  4. Run Gemini 3.1 Pro OCR (LOW thinking) on the crops
  5. Log costs
"""

import asyncio
import shutil
from pathlib import Path

from trocr_handwritten.parse.layout_parser import main as yolo_main
from trocr_handwritten.parse.settings import LayoutParserSettings, CLASS_NAMES
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.llm.factory import get_provider
from trocr_handwritten.llm.ocr import process_all_images, load_prompt
from trocr_handwritten.utils.cost_tracker import CostTracker
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
RAW_BASE = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\0. Brut\Guadeloupe\gosier\pages"
)
OUTPUT_BASE = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\gosier"
)

# Temporary dir for the 20 selected pages (so YOLO only processes these)
TEMP_INPUT = Path("data/tmp_gosier_20")

# ── Config ─────────────────────────────────────────────────────────
N_PAGES = 20
MODEL = "gemini-3.1-pro-preview"
HF_REPO = "MarieBgl/historical-layout-bagnards-EC"
HF_FILE = "20250111_yolov10_bagnards_EC.pt"
MAX_CONCURRENT = 5  # conservative to avoid rate limits


def find_untranscribed_pages(n: int) -> list[Path]:
    """Return the first N raw pages that have no output folder yet."""
    all_pages = sorted(RAW_BASE.glob("*.jpg"))
    untranscribed = []
    for page in all_pages:
        page_stem = page.stem  # e.g. FRAD971_1E35_043_113_002_C
        output_dir = OUTPUT_BASE / page_stem
        if not output_dir.exists():
            untranscribed.append(page)
            if len(untranscribed) >= n:
                break
    return untranscribed


async def run():
    # ── Step 0: Find 20 untranscribed pages ────────────────────────
    pages = find_untranscribed_pages(N_PAGES)
    if not pages:
        logger.info("No untranscribed pages found — nothing to do.")
        return

    logger.info(f"Selected {len(pages)} untranscribed pages from gosier:")
    for p in pages:
        logger.info(f"  {p.name}")

    # ── Step 1: Copy pages to temp dir ─────────────────────────────
    TEMP_INPUT.mkdir(parents=True, exist_ok=True)
    for p in pages:
        dst = TEMP_INPUT / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    # ── Step 2: YOLO layout parsing ────────────────────────────────
    logger.info("Running YOLO layout parser...")
    # Only keep Marge (class 2) and Plein Texte (class 4)
    filtered_classes = {
        k: v for k, v in CLASS_NAMES.items() if v in ("Marge", "Plein Texte")
    }

    yolo_settings = LayoutParserSettings(
        path_folder=str(TEMP_INPUT),
        path_output=str(OUTPUT_BASE),
        hf_repo=HF_REPO,
        hf_filename=HF_FILE,
        device="cpu",
        conf=0.2,
        iou=0.5,
        create_annotation_json=False,
        class_names=filtered_classes,
    )
    yolo_main(yolo_settings, logger)
    logger.info("YOLO done.")

    # ── Step 3: OCR with Gemini 3.1 Pro ────────────────────────────
    logger.info(f"Running OCR with {MODEL} (thinking=LOW)...")
    llm_settings = LLMSettings(
        provider="gemini",
        model_name=MODEL,
        max_tokens=4096,
        request_timeout=180,
    )
    provider = get_provider(llm_settings)
    prompt = load_prompt(llm_settings.prompt_path)
    cost_tracker = CostTracker(model_name=MODEL)

    # Find all crop images under the output dir for our selected pages
    crop_images = []
    for page in pages:
        page_dir = OUTPUT_BASE / page.stem
        if page_dir.exists():
            crop_images.extend(sorted(page_dir.rglob("*.jpg")))

    logger.info(f"Found {len(crop_images)} crops to transcribe")

    if crop_images:
        await process_all_images(
            images=crop_images,
            provider=provider,
            prompt=prompt,
            output_extension=".md",
            cost_tracker=cost_tracker,
            max_concurrent=MAX_CONCURRENT,
        )

    # ── Step 4: Log costs ──────────────────────────────────────────
    cost_tracker.model_name = getattr(provider, "actual_model_name", MODEL)
    cost_tracker.log_summary(log_dir="logs")
    logger.info(f"\n{cost_tracker.summary()}")

    # ── Cleanup temp dir ───────────────────────────────────────────
    shutil.rmtree(TEMP_INPUT, ignore_errors=True)
    logger.info("Temp input dir cleaned up. Done!")


if __name__ == "__main__":
    asyncio.run(run())
