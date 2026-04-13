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
NER_OUTPUT = BASE / "NER_datasets/llm/ner_llm_all.json"
CHUNK_SIZE = 500


def merge_all_runs() -> None:
    """Merge ner_llm.json and all ner_llm_N.json into ner_llm_all.json."""
    merged: dict[str, dict] = {}
    candidates = sorted(NER_OUTPUT.parent.glob("ner_llm*.json"))
    all_path = NER_OUTPUT.with_stem("ner_llm_all")
    sources = [p for p in candidates if p != all_path]
    for path in sources:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                merged[r["act_id"]] = r
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} acts from {len(sources)} file(s) → {all_path.name}")


def resolve_output_path() -> Path:
    """Return NER_OUTPUT if it doesn't exist, else ner_llm_1.json, ner_llm_2.json, ..."""
    if not NER_OUTPUT.exists():
        return NER_OUTPUT
    i = 1
    while True:
        candidate = NER_OUTPUT.with_stem(f"ner_llm_{i}")
        if not candidate.exists():
            return candidate
        i += 1


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
    """Load already-processed results from all ner_llm*.json files, keyed by act_id."""
    all_path = NER_OUTPUT.with_stem("ner_llm_all")
    candidates = sorted(NER_OUTPUT.parent.glob("ner_llm*.json"))
    sources = [p for p in candidates if p != all_path]
    if all_path.exists():
        sources = [all_path]
    merged: dict[str, NERResult] = {}
    for path in sources:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                merged[r["act_id"]] = NERResult(**r)
    return merged


def save_merged_results(
    existing: dict[str, NERResult], new_results: list[NERResult], output_path: Path
) -> None:
    """Merge new results into existing and save."""
    for r in new_results:
        existing[r.act_id] = r
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in existing.values()], f, ensure_ascii=False, indent=2
        )
    logger.info("Saved %d total results to %s", len(existing), output_path)


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
    output_path = resolve_output_path()
    if output_path != NER_OUTPUT:
        print(f"ner_llm.json already exists — saving new results to {output_path.name}")

    failed_path = NER_OUTPUT.parent / "failed_ner.json"
    failed_ids: set[str] = set()
    if failed_path.exists():
        with open(failed_path, encoding="utf-8") as f:
            failed_ids = set(json.load(f).keys())
        for fid in failed_ids:
            existing.pop(fid, None)

    pending = [r for r in all_records if r.act_id not in existing]
    print(
        f"Acts: {len(all_records)} total, {len(existing)} done, "
        f"{len(failed_ids)} failed, {len(pending)} pending"
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
        save_merged_results(existing, results, output_path)

        cost = extractor.cost_tracker.get_total_cost()
        adjusted = cost * 1.30
        print(f"  Cost so far: EUR {cost:.4f} (adjusted: EUR {adjusted:.4f})")

        if budget_eur > 0 and adjusted >= budget_eur:
            print(f"  Budget EUR {budget_eur:.2f} reached. Stopping.")
            break

    print(f"\nProcessed {len(all_new)} new acts.")
    print(extractor.cost_tracker.summary())
    extractor.cost_tracker.log_summary(log_dir="logs")

    merge_all_runs()

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
