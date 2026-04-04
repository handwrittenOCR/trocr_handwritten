"""
Re-run OCR on 6 already-transcribed crops (3 Marge + 3 Plein Texte) from abymes/1841
with thinking_level=LOW, save outputs, and generate a comparison markdown.
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
    # (region_type, page, crop_number)
    ("Plein Texte", "FRAD971_1E35_001_101_002_C", "000"),
    ("Plein Texte", "FRAD971_1E35_001_101_003_C", "003"),
    ("Plein Texte", "FRAD971_1E35_001_101_003_C", "000"),
    ("Marge", "FRAD971_1E35_001_101_003_C", "001"),
    ("Marge", "FRAD971_1E35_001_101_003_C", "004"),
    ("Marge", "FRAD971_1E35_001_101_003_C", "005"),
]

OUTPUT_DIR = Path("test_lowthinking")


async def main():
    settings = LLMSettings()  # gemini-3-pro-preview, LOW thinking via native SDK
    provider = get_provider(settings)
    prompt = Path("config/ocr.prompt").read_text(encoding="utf-8")
    cost_tracker = CostTracker(model_name=settings.model_name)

    results = []

    for region_type, page, crop_num in CROPS:
        jpg = BASE / page / region_type / f"{crop_num}.jpg"
        old_md = BASE / page / region_type / f"{crop_num}.md"

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

    # Generate comparison markdown
    lines = [
        "# LOW vs HIGH Thinking — OCR Comparison",
        "",
        f"**Model:** {settings.model_name} (redirects to gemini-3.1-pro-preview)",
        "**Thinking mode (new):** LOW  |  **Thinking mode (old):** HIGH (default, uncapped)",
        "",
        "---",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['region_type']} — `{r['page']}` — crop `{r['crop']}`",
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

    lines.append(f"\n## Cost Summary\n\n```\n{cost_tracker.summary()}\n```\n")

    comparison_path = OUTPUT_DIR / "comparison.md"
    comparison_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison saved to {comparison_path}")
    cost_tracker.model_name = getattr(
        provider, "actual_model_name", settings.model_name
    )
    cost_tracker.log_summary(log_dir="logs")
    print(f"\n{cost_tracker.summary()}")


asyncio.run(main())
