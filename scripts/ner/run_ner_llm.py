"""Run LLM NER extraction on a dataset of acts.

Reads from a JSON file of ActRecords, skips already-processed acts,
and saves results to an output JSON file.

Usage:
    python scripts/ner/run_ner_llm.py --input acts.json --output ner_results.json
    python scripts/ner/run_ner_llm.py --input acts.json --output ner_results.json --budget 5.0
    python scripts/ner/run_ner_llm.py --input acts.json --output ner_results.json -n 50
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
from trocr_handwritten.ner.schemas import ActRecord

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500


def load_acts(path: Path, limit: int | None) -> list[ActRecord]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = [ActRecord(**r) for r in raw]
    return records[:limit] if limit else records


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["act_id"]: r for r in json.load(f)}


def save_results(results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


async def run(
    input_path: Path,
    output_path: Path,
    prompt_path: Path,
    tool_path: Path,
    model: str,
    max_concurrent: int,
    budget_eur: float,
    limit: int | None,
) -> None:
    all_records = load_acts(input_path, limit)
    existing = load_existing(output_path)
    pending = [r for r in all_records if r.act_id not in existing]
    print(
        f"Acts: {len(all_records)} total, {len(existing)} done, {len(pending)} pending"
    )

    if not pending:
        print("Nothing to process.")
        return

    with open(prompt_path, encoding="utf-8") as f:
        prompt = f.read().strip()
    with open(tool_path, encoding="utf-8") as f:
        tool = json.load(f)

    settings = LLMSettings(provider="gemini", model_name=model, request_timeout=90)
    extractor = LLMExtractor(
        settings, prompt=prompt, tool=tool, max_concurrent=max_concurrent
    )

    chunks = [pending[i : i + CHUNK_SIZE] for i in range(0, len(pending), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        print(f"\nChunk {idx + 1}/{len(chunks)}: {len(chunk)} acts")
        results = await extractor.extract_batch(
            chunk,
            text_fn=lambda r: f"MARGE:\n{r.marge_text}\n\nTEXTE COMPLET:\n{r.plein_texte_text}",
            id_fn=lambda r: r.act_id,
        )

        for record, result in zip(chunk, results):
            if result is not None:
                result["act_id"] = record.act_id
                existing[record.act_id] = result
        save_results(existing, output_path)

        cost = extractor.cost_tracker.get_total_cost()
        adjusted = cost * 1.30
        print(f"  Cost so far: EUR {cost:.4f} (adjusted: EUR {adjusted:.4f})")

        if budget_eur > 0 and adjusted >= budget_eur:
            print(f"  Budget EUR {budget_eur:.2f} reached. Stopping.")
            break

    print(f"\nDone. {len(existing)} total results saved to {output_path}")
    print(extractor.cost_tracker.summary())
    extractor.cost_tracker.log_summary(log_dir="logs")

    if extractor.failed:
        failed_path = output_path.with_name("failed_ner.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(extractor.failed, f, ensure_ascii=False, indent=2)
        print(f"  {len(extractor.failed)} failed. See {failed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to acts_dataset.json"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to output ner_results.json"
    )
    parser.add_argument("--prompt", type=Path, default=Path("config/ner.prompt"))
    parser.add_argument(
        "--tool", type=Path, required=True, help="Path to JSON tool definition"
    )
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--budget", type=float, default=0.0, metavar="EUR")
    parser.add_argument("-n", type=int, default=None, help="Limit number of acts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    asyncio.run(
        run(
            input_path=args.input,
            output_path=args.output,
            prompt_path=args.prompt,
            tool_path=args.tool,
            model=args.model,
            max_concurrent=args.max_concurrent,
            budget_eur=args.budget,
            limit=args.n,
        )
    )


if __name__ == "__main__":
    main()
