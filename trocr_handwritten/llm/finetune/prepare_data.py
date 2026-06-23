import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from trocr_handwritten.llm.finetune.manifest import proportional_take, write_manifest
from trocr_handwritten.llm.finetune.settings import DataSettings


def collect_split(base: Path, split: str):
    """
    Collect (relative_image_path, region, text) for every labelled crop in a split.

    Args:
        base: Root OCR directory (e.g. data/ocr).
        split: One of train/dev/test.

    Returns:
        list: Records with rel_image, region, text.
    """
    records = []
    labels_root = base / split / "labels"
    for lbl in sorted(labels_root.rglob("*.txt")):
        region = lbl.parent.name
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = base / split / "images" / region / f"{lbl.stem}{ext}"
            if cand.exists():
                img = f"{split}/images/{region}/{lbl.stem}{ext}"
                break
        if img is None:
            continue
        records.append({"image": img, "region": region, "text": text})
    return records


def nested_stratified_subsets(records, paliers, seed=42):
    """
    Build nested, region-stratified subsets of increasing size.

    Args:
        records: Full train records.
        paliers: Sorted target sizes.
        seed: Shuffle seed.

    Returns:
        dict: size -> list of records.
    """
    rng = random.Random(seed)
    by_region = defaultdict(list)
    for r in records:
        by_region[r["region"]].append(r)
    for reg in by_region:
        rng.shuffle(by_region[reg])
    return {size: proportional_take(by_region, size) for size in paliers}


def main():
    settings = DataSettings()
    parser = argparse.ArgumentParser(description="Prepare VLM fine-tuning manifests.")
    parser.add_argument("--ocr-dir", default=settings.ocr_dir)
    parser.add_argument("--out-dir", default=settings.out_dir)
    parser.add_argument("--seed", type=int, default=settings.seed)
    args = parser.parse_args()

    base = Path(args.ocr_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in settings.eval_splits:
        recs = collect_split(base, split)
        write_manifest(out / f"{split}.jsonl", recs)
        print(f"{split}: {len(recs)} records")

    train = collect_split(base, settings.train_split)
    print(f"train full: {len(train)} records")
    subsets = nested_stratified_subsets(train, settings.paliers, seed=args.seed)

    prev = set()
    for size in settings.paliers:
        recs = subsets[size]
        write_manifest(out / f"train_{size}.jsonl", recs)
        keys = {r["image"] for r in recs}
        nested = prev.issubset(keys)
        reg = Counter(r["region"] for r in recs)
        print(f"train_{size}: {len(recs)} | nested={nested} | {dict(reg)}")
        prev = keys


if __name__ == "__main__":
    main()
