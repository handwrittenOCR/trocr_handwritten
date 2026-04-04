"""
Generate HIGH thinking (default/uncapped) OCR for FRAD971_1E35_014_102_024_C crops.
The LOW thinking results already exist in test_lowthinking/new/.
This script saves HIGH thinking results to test_lowthinking/old/ and appends
a comparison to test_lowthinking/comparison.md.
"""

import asyncio
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

from trocr_handwritten.utils.cost_tracker import CostTracker

load_dotenv()

BASE = Path("test_lowthinking")
PAGE = "FRAD971_1E35_014_102_024_C"

CROPS = [
    ("Plein Texte", "000"),
    ("Plein Texte", "014"),
    ("Marge", "005"),
    ("Marge", "006"),
]

MODEL = "gemini-3.1-pro-preview"


async def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = Path("config/ocr.prompt").read_text(encoding="utf-8")
    cost_tracker = CostTracker(model_name=MODEL)

    # HIGH thinking = no thinking_config (default, uncapped)
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=4096,
        # No thinking_config → default HIGH thinking
    )

    results = []

    for region_type, crop_num in CROPS:
        img_path = BASE / "new" / PAGE / region_type / f"{crop_num}.jpg"
        low_md_path = BASE / "new" / PAGE / region_type / f"{crop_num}.md"

        if not img_path.exists():
            print(f"  SKIP (no jpg): {region_type}/{crop_num}")
            continue

        low_text = (
            low_md_path.read_text(encoding="utf-8").strip()
            if low_md_path.exists()
            else "(no LOW transcription)"
        )

        print(f"\nProcessing {region_type} | {crop_num} (HIGH thinking) ...")

        image_bytes = img_path.read_bytes()
        mime = "image/jpeg"
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda p=parts: client.models.generate_content(
                model=MODEL,
                contents=p,
                config=config,
            ),
        )

        um = response.usage_metadata
        inp = um.prompt_token_count or 0
        out = um.candidates_token_count or 0
        think = um.thoughts_token_count or 0
        high_text = response.text

        actual_model = getattr(response, "model_version", None)
        if actual_model:
            cost_tracker.model_name = actual_model

        cost_tracker.add_usage(inp, out, think)
        print(f"  tokens: input={inp}, output={out}, thinking={think}")
        print(f"  result: {high_text[:80]}...")

        # Save to old/
        out_dir = BASE / "old" / PAGE / region_type
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{crop_num}.md").write_text(high_text, encoding="utf-8")

        results.append(
            {
                "region_type": region_type,
                "crop": crop_num,
                "high": high_text.strip(),
                "low": low_text,
                "input_tokens": inp,
                "output_tokens": out,
                "thinking_tokens": think,
            }
        )

    # Append comparison to comparison.md
    comparison_path = BASE / "comparison.md"
    existing = (
        comparison_path.read_text(encoding="utf-8") if comparison_path.exists() else ""
    )

    lines = [
        "",
        f"# Round 3 — Page {PAGE} (Anse-Bertrand 1842)",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['region_type']} — `{PAGE}` — crop `{r['crop']}`",
            "",
            f"**Tokens (HIGH):** input={r['input_tokens']}, output={r['output_tokens']}, thinking={r['thinking_tokens']}",
            "",
            "### OLD (HIGH thinking)",
            "```",
            r["high"],
            "```",
            "",
            "### NEW (LOW thinking)",
            "```",
            r["low"],
            "```",
            "",
            "---",
            "",
        ]

    lines.append(f"\n## Round 3 Cost Summary\n\n```\n{cost_tracker.summary()}\n```\n")

    comparison_path.write_text(existing + "\n".join(lines), encoding="utf-8")
    print(f"\nComparison appended to {comparison_path}")

    cost_tracker.log_summary(log_dir="logs")
    print(f"\n{cost_tracker.summary()}")


asyncio.run(main())
