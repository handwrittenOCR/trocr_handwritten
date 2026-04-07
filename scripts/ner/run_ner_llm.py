"""Run LLM NER extraction on all acts (or a filtered subset).

Reads from NER_datasets/raw/acts_dataset.json, skips already-processed acts,
and saves results to NER_datasets/ner_llm.json.

Usage:
    python scripts/ner/run_ner_llm.py
    python scripts/ner/run_ner_llm.py --commune abymes
    python scripts/ner/run_ner_llm.py --commune abymes --year 1842
    python scripts/ner/run_ner_llm.py --budget 5.0
    python scripts/ner/run_ner_llm.py -n 50
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.ner.llm_extractor import LLMExtractor
from trocr_handwritten.ner.schemas import ActRecord, NERResult

logger = logging.getLogger(__name__)

BASE = Path(
    "C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/"
    "3. OCR/2. TrOCR/5. Data (output)/ECES"
)
ACTS_DATASET = BASE / "NER_datasets/raw/acts_dataset.json"
NER_OUTPUT = BASE / "NER_datasets/ner_llm.json"
CHUNK_SIZE = 500


def load_acts(
    commune: str | None, year: str | None, limit: int | None
) -> list[ActRecord]:
    """Load and filter acts from acts_dataset.json."""
    with open(ACTS_DATASET, encoding="utf-8") as f:
        raw = json.load(f)

    records = [ActRecord(**r) for r in raw]

    if commune:
        records = [r for r in records if r.commune == commune]
    if year:
        records = [r for r in records if r.year == year]
    if limit:
        records = records[:limit]

    return records


def load_existing_results() -> dict[str, NERResult]:
    """Load already-processed results keyed by act_id."""
    if not NER_OUTPUT.exists():
        return {}
    with open(NER_OUTPUT, encoding="utf-8") as f:
        raw = json.load(f)
    return {r["act_id"]: NERResult(**r) for r in raw}


def save_merged_results(
    existing: dict[str, NERResult], new_results: list[NERResult]
) -> None:
    """Merge new results into existing and save."""
    for r in new_results:
        existing[r.act_id] = r
    NER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(NER_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in existing.values()], f, ensure_ascii=False, indent=2
        )
    logger.info("Saved %d total results to %s", len(existing), NER_OUTPUT)


async def run(
    commune: str | None,
    year: str | None,
    limit: int | None,
    model: str,
    max_concurrent: int,
    budget_eur: float,
) -> None:
    """Main async runner."""
    all_records = load_acts(commune, year, limit)
    existing = load_existing_results()

    pending = [r for r in all_records if r.act_id not in existing]
    print(
        f"Acts: {len(all_records)} total, {len(existing)} done, {len(pending)} pending"
    )

    if not pending:
        print("Nothing to process.")
        return

    settings = LLMSettings(provider="gemini", model_name=model, request_timeout=90)
    extractor = LLMExtractor(settings)

    all_new: list[NERResult] = []
    chunks = [pending[i : i + CHUNK_SIZE] for i in range(0, len(pending), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        print(f"\nChunk {idx + 1}/{len(chunks)}: {len(chunk)} acts")
        results = await extractor.extract_batch(chunk, max_concurrent=max_concurrent)
        all_new.extend(results)
        save_merged_results(existing, results)

        cost = extractor.cost_tracker.get_total_cost()
        adjusted = cost * 1.30
        print(f"  Cost so far: EUR {cost:.4f} (adjusted: EUR {adjusted:.4f})")

        if budget_eur > 0 and adjusted >= budget_eur:
            print(f"  Budget EUR {budget_eur:.2f} reached. Stopping.")
            break

    print(f"\nProcessed {len(all_new)} new acts.")
    print(extractor.cost_tracker.summary())
    extractor.cost_tracker.log_summary(log_dir="logs")

    if extractor.failed:
        failed_path = NER_OUTPUT.parent / "failed_ner.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(extractor.failed, f, ensure_ascii=False, indent=2)
        print(f"  {len(extractor.failed)} failed. See {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM NER on all acts.")
    parser.add_argument("--commune", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        metavar="EUR",
        help="Stop when adjusted cost exceeds this amount in EUR (0 = no limit)",
    )
    parser.add_argument("-n", type=int, default=None, help="Limit number of acts")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    asyncio.run(
        run(
            commune=args.commune,
            year=args.year,
            limit=args.n,
            model=args.model,
            max_concurrent=args.max_concurrent,
            budget_eur=args.budget,
        )
    )


if __name__ == "__main__":
    main()
