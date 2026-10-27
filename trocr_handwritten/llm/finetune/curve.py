import argparse
import json
from pathlib import Path

from trocr_handwritten.llm.finetune.settings import CurveSettings


def stats(metrics_path, level="N2"):
    """Return (micro_cer, macro_cer, micro_cer_N0, n) from a metrics.json, or None."""
    p = Path(metrics_path)
    if not p.exists():
        return None
    g = json.loads(p.read_text())["global"]
    return (
        g[level]["micro_cer"],
        g[level]["macro_cer"],
        g["N0"]["micro_cer"],
        g[level]["n"],
    )


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
        print(f"\n=== {key} (micro-CER, N2 primary / N0 secondary) ===")
        print(
            f"{'palier':>6} | {'dev N2':>8} {'dev N0':>7} "
            f"| {'test N2':>8} {'test macro':>10} {'test N0':>7}"
        )
        for p in settings.paliers:
            d = cell(key, p, "dev", settings)
            t = cell(key, p, "test", settings)

            def fmt(s, cols):
                if not s:
                    return " ".join(f"{'-':>{w}}" for _, w in cols)
                return " ".join(f"{s[i]:>{w}.4f}" for i, w in cols)

            dev = fmt(d, [(0, 8), (2, 7)])
            test = fmt(t, [(0, 8), (1, 10), (2, 7)])
            print(f"{p:>6} | {dev} | {test}")


if __name__ == "__main__":
    main()
