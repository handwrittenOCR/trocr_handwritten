"""NER extraction pipeline: dataset -> regex + LLM -> compare -> output.

Usage:
    python -m trocr_handwritten.ner.pipeline \
        --input_dir <transcription_dir> \
        --output_dir <output_dir> \
        --methods regex llm \
        --model gemini-3-flash-preview \
        --max_concurrent 10
"""

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import List

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.ner.compare import (
    compare_results,
    compute_agreement,
    save_comparison,
)
from trocr_handwritten.ner.dataset import build_dataset, save_dataset
from trocr_handwritten.ner.llm_extractor import LLMExtractor
from trocr_handwritten.ner.regex_extractor import RegexExtractor
from trocr_handwritten.ner.schemas import NERResult, flatten_ner_result

logger = logging.getLogger(__name__)


def save_ner_results(results: List[NERResult], output_dir: str, method: str) -> None:
    """Save NER results as JSON and flat CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON (nested)
    json_path = output_path / f"ner_{method}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)

    # CSV (flat) — collect all field names across all rows for a consistent header
    csv_path = output_path / f"ner_{method}.csv"
    rows = [flatten_ner_result(r) for r in results]
    if rows:
        all_fields = dict.fromkeys(k for row in rows for k in row.keys())
        fieldnames = list(all_fields.keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Saved %s results: %s (%d records)", method, json_path, len(results))


async def run_pipeline(
    input_dir: str,
    output_dir: str,
    commune: str,
    year: str,
    methods: List[str],
    provider: str,
    model: str,
    max_concurrent: int,
    timeout: int,
    limit: int = None,
):
    """Run the full NER pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Build dataset
    print("=== Step 1: Building dataset ===")
    records, _ = build_dataset(input_dir, commune, year)
    if not records:
        print("No acts found. Exiting.")
        return
    save_dataset(records, output_dir)

    if limit:
        records = records[:limit]
        print(f"Limited to {limit} records for testing")

    # Step 2: Regex extraction
    regex_results = None
    if "regex" in methods:
        print(f"\n=== Step 2: Regex extraction ({len(records)} acts) ===")
        extractor = RegexExtractor()
        regex_results = extractor.extract_all(records)
        save_ner_results(regex_results, output_dir, "regex")

        deaths = sum(1 for r in regex_results if r.death_act)
        births = sum(1 for r in regex_results if r.birth_act)
        print(f"  Extracted: {deaths} deaths, {births} births")

    # Step 3: LLM extraction
    llm_results = None
    if "llm" in methods:
        print(f"\n=== Step 3: LLM extraction ({len(records)} acts) ===")
        settings = LLMSettings(
            provider=provider,
            model_name=model,
            request_timeout=timeout,
        )
        llm_extractor = LLMExtractor(settings)
        llm_results = await llm_extractor.extract_batch(
            records, max_concurrent=max_concurrent
        )
        save_ner_results(llm_results, output_dir, "llm")

        deaths = sum(1 for r in llm_results if r.death_act)
        births = sum(1 for r in llm_results if r.birth_act)
        print(f"  Extracted: {deaths} deaths, {births} births")
        print(f"  {llm_extractor.cost_tracker.summary()}")
        llm_extractor.cost_tracker.log_summary()

        if llm_extractor.failed:
            failed_path = output_path / "failed_ner.json"
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(llm_extractor.failed, f, ensure_ascii=False, indent=2)
            print(f"  {len(llm_extractor.failed)} failed. See {failed_path}")

    # Step 4: Compare
    if regex_results and llm_results:
        print("\n=== Step 4: Comparison ===")
        comparison_rows = compare_results(regex_results, llm_results)
        agreement = compute_agreement(comparison_rows)
        save_comparison(comparison_rows, agreement, output_dir)

        # Print summary
        print(
            f"  {'Field':<35} {'Match':>6} {'Mismatch':>9} {'Regex only':>11} {'LLM only':>9} {'Agree%':>7}"
        )
        print(f"  {'-'*35} {'-'*6} {'-'*9} {'-'*11} {'-'*9} {'-'*7}")
        for field, s in sorted(agreement.items()):
            agree_str = (
                f"{s['agreement_pct']}%" if s["agreement_pct"] is not None else "n/a"
            )
            print(
                f"  {field:<35} {s['match']:>6} {s['mismatch']:>9} "
                f"{s['regex_only']:>11} {s['llm_only']:>9} {agree_str:>7}"
            )

    print(f"\n=== Done === Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="NER extraction pipeline for civil registry acts."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Transcription directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to input_dir/ner)",
    )
    parser.add_argument("--commune", type=str, default="abymes")
    parser.add_argument("--year", type=str, default="1842")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["regex", "llm"],
        choices=["regex", "llm"],
        help="Extraction methods to run",
    )
    parser.add_argument("--provider", type=str, default="gemini")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "-n", type=int, default=None, help="Limit number of acts (for testing)"
    )

    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.input_dir) / "ner")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    asyncio.run(
        run_pipeline(
            input_dir=args.input_dir,
            output_dir=output_dir,
            commune=args.commune,
            year=args.year,
            methods=args.methods,
            provider=args.provider,
            model=args.model,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            limit=args.n,
        )
    )


if __name__ == "__main__":
    main()
