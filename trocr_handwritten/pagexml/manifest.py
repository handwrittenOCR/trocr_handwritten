"""
Rebuild ``pages_manifest.csv`` from the corpus PAGE-XML and check the corpus against the
HTR dataset.

The PAGE-XML files are the primary data (boxes, classes, handwriting status, transcriptions);
the manifest is a table derived from them, one row per region. The page split comes from the
detection dataset (``<splits-dir>/<split>/annotations.json``); ``ls_key`` (internal Label Studio
key) is carried over from an existing manifest when given.

Usage:
    python -m trocr_handwritten.pagexml.manifest --out data/processed/pages_manifest.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from trocr_handwritten.pagexml.page import read_corpus
from trocr_handwritten.pagexml.settings import COLS, PageXmlSettings


def page_splits(splits_dir):
    """
    Split of every page of the detection dataset.

    Args:
        splits_dir: Folder with ``<split>/annotations.json`` (``filename`` per page).

    Returns:
        dict: page filename -> split.
    """
    out = {}
    for s in PageXmlSettings().splits:
        for e in json.load(
            open(Path(splits_dir) / s / "annotations.json", encoding="utf-8")
        ):
            out[e["filename"]] = s
    return out


def build_rows(corpus, splits, ls_keys=None):
    """
    Manifest rows of a corpus, one per region, in page then document order.

    Args:
        corpus: Output of ``read_corpus``.
        splits: page filename -> split.
        ls_keys: Optional region_id -> Label Studio key.

    Returns:
        list: Row dicts with the ``COLS`` columns.
    """
    rows = []
    for path, page, regions in corpus:
        for r in regions:
            x, y, w, h = r["box"]
            rows.append(
                {
                    "page": page,
                    "page_xml": f"page_xml/{path.name}",
                    "split": splits[page],
                    "region_id": r["region_id"],
                    "class": r["class"],
                    "handwriting": r["handwriting"],
                    "transcription": r["transcription"],
                    "htr_category": r["htr_category"] or "",
                    "htr_crop": r["htr_crop"] or "",
                    "ls_key": (ls_keys or {}).get(r["region_id"], ""),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "box_provenance": r["box_provenance"] or "",
                }
            )
    return rows


def check(corpus, splits, htr_dir):
    """
    Consistency problems between the PAGE-XML corpus and the HTR dataset.

    Checks: every page has a split; every box is marked validated; a region has a text if and
    only if it is transcribed; every transcribed region has its HTR crop (image, label, entry of
    ``annotations.json``) in the split of its page, with the same text; every HTR crop belongs to
    a transcribed region.

    Args:
        corpus: Output of ``read_corpus``.
        splits: page filename -> split.
        htr_dir: HTR dataset root (``<split>/annotations.json``, ``images/``, ``labels/``).

    Returns:
        list: Problem descriptions, empty when the corpus is consistent.
    """
    htr_dir = Path(htr_dir)
    htr = {}
    for s in PageXmlSettings().splits:
        for e in json.load(open(htr_dir / s / "annotations.json", encoding="utf-8")):
            htr[(s, e["subfolder"], Path(e["filename"]).stem)] = e
    problems, seen = [], set()
    for _, page, regions in corpus:
        if page not in splits:
            problems.append(f"{page}: page without split")
            continue
        for r in regions:
            rid = f"{page} {r['region_id']}"
            if not r["validated"]:
                problems.append(f"{rid}: box not marked validated")
            if (r["text"] is not None) != (r["transcription"] == "transcrit"):
                problems.append(f"{rid}: text and transcription status disagree")
            if r["transcription"] != "transcrit":
                continue
            k = (r["htr_split"], r["htr_category"], r["htr_crop"])
            seen.add(k)
            if r["htr_split"] != splits[page]:
                problems.append(
                    f"{rid}: crop split {r['htr_split']} != page split {splits[page]}"
                )
            if k not in htr:
                problems.append(
                    f"{rid}: crop {'/'.join(k)} missing from annotations.json"
                )
                continue
            lbl = htr_dir / k[0] / "labels" / k[1] / f"{k[2]}.txt"
            img = htr_dir / k[0] / "images" / k[1] / htr[k]["filename"]
            if not img.exists() or not lbl.exists():
                problems.append(f"{rid}: crop image or label file missing")
            elif (
                not r["text"].strip()
                == htr[k]["text"].strip()
                == lbl.read_text(encoding="utf-8").strip()
            ):
                problems.append(f"{rid}: text differs between PAGE-XML and HTR")
    problems += [
        f"HTR crop {'/'.join(k)} without transcribed region"
        for k in sorted(set(htr) - seen)
    ]
    return problems


def main():
    """CLI: write the manifest and print the consistency check; exit 1 on problems."""
    st = PageXmlSettings()
    ap = argparse.ArgumentParser(
        description="Rebuild pages_manifest.csv from the PAGE-XML and check the corpus."
    )
    ap.add_argument("--page-xml-dir", default=st.page_xml_dir)
    ap.add_argument("--splits-dir", default=st.splits_dir)
    ap.add_argument("--htr-dir", default=st.htr_dir)
    ap.add_argument(
        "--ls-keys",
        default=st.manifest,
        help="Existing manifest to carry ls_key from (optional).",
    )
    ap.add_argument("--out", required=True, help="Manifest CSV to write.")
    args = ap.parse_args()
    corpus = read_corpus(args.page_xml_dir)
    splits = page_splits(args.splits_dir)
    ls_keys = {}
    if args.ls_keys and Path(args.ls_keys).exists():
        ls_keys = {
            r["region_id"]: r["ls_key"]
            for r in csv.DictReader(open(args.ls_keys, encoding="utf-8"))
            if r["ls_key"]
        }
    rows = build_rows(corpus, splits, ls_keys)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    problems = check(corpus, splits, args.htr_dir)
    n_tr = sum(r["transcription"] == "transcrit" for r in rows)
    print(f"{len(corpus)} pages, {len(rows)} regions, {n_tr} transcribed -> {args.out}")
    print(f"{len(problems)} problems")
    for p in problems[:50]:
        print(" ", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
