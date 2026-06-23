import argparse
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset

from trocr_handwritten.llm.finetune.manifest import proportional_take, write_manifest
from trocr_handwritten.llm.finetune.settings import TekliaSettings


def century(value):
    """Map a start_date (int year, possibly 0/unknown) to a century bucket key."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return "unk"
    if year <= 0:
        return "unk"
    return str(year // 100 + 1)


def collect_rows(settings):
    """
    Load RecordGold rows that carry a public IIIF image URL and non-empty text.

    Args:
        settings: TekliaSettings.

    Returns:
        list: Records with record_id, url, text, parish, bucket.
    """
    ds = load_dataset(settings.dataset, split=settings.split)
    rows = []
    for r in ds:
        url = r.get("record_url") or ""
        text = (r.get("text") or "").strip()
        if settings.iiif_host not in url or not text:
            continue
        rows.append(
            {
                "record_id": str(r.get("record_id")),
                "url": url,
                "text": text,
                "parish": r.get("parish") or "unk",
                "bucket": century(r.get("start_date")),
            }
        )
    return rows


def stratified_sample(rows, max_samples, seed=42):
    """
    Take a stratified subsample by parish x century, preserving proportions.

    Args:
        rows: Full row list.
        max_samples: Target size, or None to keep all.
        seed: Shuffle seed.

    Returns:
        list: Subsampled rows.
    """
    if not max_samples or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["parish"], r["bucket"])].append(r)
    for key in by_key:
        rng.shuffle(by_key[key])
    return proportional_take(by_key, max_samples)


def fetch_image(row, img_dir, session, settings):
    """Download one IIIF region image to disk, returning the path or None."""
    dest = img_dir / f"{row['record_id']}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for _ in range(settings.retries):
        try:
            resp = session.get(row["url"], timeout=settings.timeout)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return dest
        except requests.RequestException:
            pass
    return None


def download_all(rows, img_dir, settings):
    """Download every row's image concurrently, returning rows that succeeded."""
    img_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    with ThreadPoolExecutor(max_workers=settings.workers) as pool:
        futures = {
            pool.submit(fetch_image, r, img_dir, session, settings): r for r in rows
        }
        done = 0
        for fut in as_completed(futures):
            if fut.result() is not None:
                ok.append(futures[fut])
            done += 1
            if done % 500 == 0:
                print(f"  downloaded {done}/{len(rows)} ({len(ok)} ok)")
    return ok


def main():
    settings = TekliaSettings()
    parser = argparse.ArgumentParser(
        description="Download RecordGold IIIF images and write a fine-tuning manifest."
    )
    parser.add_argument("--out-dir", default=settings.out_dir)
    parser.add_argument("--split", default=settings.split)
    parser.add_argument("--max-samples", type=int, default=settings.max_samples)
    parser.add_argument("--workers", type=int, default=settings.workers)
    parser.add_argument("--seed", type=int, default=settings.seed)
    args = parser.parse_args()

    settings = settings.model_copy(
        update={
            "out_dir": args.out_dir,
            "split": args.split,
            "max_samples": args.max_samples,
            "workers": args.workers,
            "seed": args.seed,
        }
    )
    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(settings)
    print(f"{settings.split}: {len(rows)} rows with IIIF image + text")
    rows = stratified_sample(rows, settings.max_samples, seed=settings.seed)
    print(f"selected: {len(rows)} | buckets={dict(Counter(r['bucket'] for r in rows))}")

    rows = download_all(rows, out / "images", settings)
    print(f"downloaded ok: {len(rows)}")

    records = [
        {
            "image": f"{settings.region_label}/images/{r['record_id']}.jpg",
            "region": settings.region_label,
            "text": r["text"],
        }
        for r in rows
    ]
    manifest = out / f"{settings.split}.jsonl"
    write_manifest(manifest, records)
    print(f"wrote {manifest} ({len(records)} records)")


if __name__ == "__main__":
    main()
