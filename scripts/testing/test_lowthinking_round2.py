"""
Round 2: Re-run OCR on all crops from 2 additional pages (004_C and 005_C) from abymes/1841
with thinking_level=LOW, save outputs, and append to comparison.md.
"""

import asyncio
from pathlib import Path

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.llm.factory import get_provider
from trocr_handwritten.utils.cost_tracker import CostTracker

BASE = Path(
    r"C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR"
    r"\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes\1841"
)

CROPS = [
    # Page 004_C
    ("Plein Texte", "FRAD971_1E35_001_101_004_C", "000"),
    ("Plein Texte", "FRAD971_1E35_001_101_004_C", "004"),
    ("Marge", "FRAD971_1E35_001_101_004_C", "001"),
    ("Marge", "FRAD971_1E35_001_101_004_C", "003"),
    # Page 005_C
    ("Plein Texte", "FRAD971_1E35_001_101_005_C", "001"),
    ("Plein Texte", "FRAD971_1E35_001_101_005_C", "002"),
    ("Plein Texte", "FRAD971_1E35_001_101_005_C", "003"),
    ("Plein Texte", "FRAD971_1E35_001_101_005_C", "007"),
    ("Marge", "FRAD971_1E35_001_101_005_C", "004"),
    ("Marge", "FRAD971_1E35_001_101_005_C", "005"),
    ("Marge", "FRAD971_1E35_001_101_005_C", "008"),
    ("Marge", "FRAD971_1E35_001_101_005_C", "009"),
]

OUTPUT_DIR = Path("test_lowthinking")


async def main():
    settings = LLMSettings()
    provider = get_provider(settings)
    prompt = Path("config/ocr.prompt").read_text(encoding="utf-8")
    cost_tracker = CostTracker(model_name=settings.model_name)

    results = []

    for region_type, page, crop_num in CROPS:
        jpg = BASE / page / region_type / f"{crop_num}.jpg"
        old_md = BASE / page / region_type / f"{crop_num}.md"

        if not jpg.exists():
            print(f"  SKIP (no jpg): {region_type} | {page} | {crop_num}")
            continue

        old_text = (
            old_md.read_text(encoding="utf-8").strip()
            if old_md.exists()
            else "(no old transcription)"
        )

        print(f"\nProcessing {region_type} | {page} | crop {crop_num} ...")
        new_text, inp, out, think = await provider.ocr_image_async(jpg, prompt)
        cost_tracker.add_usage(inp, out, think)
        print(f"  tokens: input={inp}, output={out}, thinking={think}")
        print(f"  result: {new_text[:80]}...")

        # Save new output
        out_dir = OUTPUT_DIR / page / region_type
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{crop_num}.md").write_text(new_text, encoding="utf-8")

        results.append(
            {
                "region_type": region_type,
                "page": page,
                "crop": crop_num,
                "old": old_text,
                "new": new_text.strip(),
                "input_tokens": inp,
                "output_tokens": out,
                "thinking_tokens": think,
            }
        )

    # Append to existing comparison.md
    comparison_path = OUTPUT_DIR / "comparison.md"
    existing = (
        comparison_path.read_text(encoding="utf-8") if comparison_path.exists() else ""
    )

    lines = [
        "",
        "# Round 2 — Pages 004_C and 005_C",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['region_type']} -- `{r['page']}` -- crop `{r['crop']}`",
            "",
            f"**Tokens:** input={r['input_tokens']}, output={r['output_tokens']}, thinking={r['thinking_tokens']}",
            "",
            "### OLD (HIGH thinking)",
            "```",
            r["old"],
            "```",
            "",
            "### NEW (LOW thinking)",
            "```",
            r["new"],
            "```",
            "",
            "---",
            "",
        ]

    lines.append(f"\n## Round 2 Cost Summary\n\n```\n{cost_tracker.summary()}\n```\n")

    comparison_path.write_text(existing + "\n".join(lines), encoding="utf-8")
    print(f"\nComparison appended to {comparison_path}")

    cost_tracker.model_name = getattr(
        provider, "actual_model_name", settings.model_name
    )
    cost_tracker.log_summary(log_dir="logs")
    print(f"\n{cost_tracker.summary()}")


asyncio.run(main())
