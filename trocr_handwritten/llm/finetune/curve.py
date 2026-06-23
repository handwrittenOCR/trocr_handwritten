import argparse
import json
from pathlib import Path

from trocr_handwritten.llm.finetune.settings import CurveSettings


def stats(metrics_path):
    """Return (cer_mean, cer_median, cer_std, n) from a metrics.json, or None."""
    p = Path(metrics_path)
    if not p.exists():
        return None
    g = json.loads(p.read_text())["global"]
    return g["cer_mean"], g["cer_median"], g["cer_std"], g["n"]


def cell(key, palier, split, settings):
    """Resolve the metrics.json path for one (key, palier, split) cell."""
    root = settings.ocr_dir
    if palier == 0:
        if split == "test":
            return stats(
                f"{root}/test/predictions/{settings.basepred[key]}/metrics.json"
            )
        return stats(f"{root}/dev/predictions/zs-{key}/metrics.json")
    return stats(f"{root}/{split}/predictions/ft-{key}-{palier}/metrics.json")


def main():
    settings = CurveSettings()
    parser = argparse.ArgumentParser(description="Print LoRA learning curves.")
    parser.add_argument("--ocr-dir", default=settings.ocr_dir)
    args = parser.parse_args()
    settings = settings.model_copy(update={"ocr_dir": args.ocr_dir})

    for key in settings.keys:
        print(f"\n=== {key} (CER clipped@1) ===")
        print(
            f"{'palier':>6} | {'dev mean':>8} {'median':>7} {'std':>6} "
            f"| {'test mean':>9} {'median':>7} {'std':>6}"
        )
        for p in settings.paliers:
            d = cell(key, p, "dev", settings)
            t = cell(key, p, "test", settings)

            def fmt(s):
                return (
                    f"{s[0]:>8.4f} {s[1]:>7.4f} {s[2]:>6.3f}"
                    if s
                    else f"{'-':>8} {'-':>7} {'-':>6}"
                )

            print(f"{p:>6} | {fmt(d)} | {fmt(t)}")


if __name__ == "__main__":
    main()
