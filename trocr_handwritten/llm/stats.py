import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from trocr_handwritten.llm.settings import StatsSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_per_item(metrics_path):
    """Load the per_item list from a metrics.json produced by llm.metrics."""
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return data["per_item"]


def _register_stats(per_item, level, subfolder=None):
    """
    Pre-aggregate edits and reference characters per register for one level.

    Args:
        per_item: per_item list from a metrics record.
        level: normalization code (N0..N3).
        subfolder: if given, keep only items of that category.

    Returns:
        dict: register -> {"edits": int, "chars": int}.
    """
    reg = {}
    for item in per_item:
        if subfolder is not None and item["subfolder"] != subfolder:
            continue
        lv = item["levels"][level]
        acc = reg.setdefault(item["register"], {"edits": 0, "chars": 0})
        acc["edits"] += lv["char_edits"]
        acc["chars"] += lv["ref_chars"]
    return reg


def _micro(selection):
    """Micro-CER over a list of per-register {edits, chars} dicts."""
    edits = sum(s["edits"] for s in selection)
    chars = sum(s["chars"] for s in selection)
    return edits / chars if chars else 0.0


def _summary(samples, point):
    """Percentile 95% CI, SE and the observed point estimate."""
    return {
        "point": point,
        "se": float(np.std(samples, ddof=1)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


class RegisterBootstrap(StatsSettings):
    """
    Register-cluster bootstrap CIs and paired contrasts.

    Inherits its parameters (B, seed, default level, MDE factor) from
    `StatsSettings`, so callers tune behaviour by constructing the class with
    overrides, e.g. `RegisterBootstrap(bootstrap_samples=2000)`. The register is
    the resampling unit: errors correlate within a hand and the split isolates
    hands at the register level, so resampling crops would understate variance.
    """

    def _bootstrap(self, reg_stats, stat_fn=_micro):
        """Resample registers with replacement and recompute stat_fn B times."""
        registers = list(reg_stats.values())
        R = len(registers)
        rng = np.random.default_rng(self.seed)
        out = np.empty(self.bootstrap_samples)
        for b in range(self.bootstrap_samples):
            idx = rng.integers(0, R, size=R)
            out[b] = stat_fn([registers[i] for i in idx])
        return out

    def ci(self, metrics_path, level=None, subfolder=None):
        """Register-cluster CI for one system's micro-CER at one level."""
        level = level or self.default_level
        reg = _register_stats(_load_per_item(metrics_path), level, subfolder)
        result = _summary(self._bootstrap(reg), _micro(list(reg.values())))
        result["n_registers"] = len(reg)
        return result

    def paired(self, a_path, b_path, level=None, subfolder=None):
        """
        Paired register-cluster contrast on micro-CER(A) - micro-CER(B).

        The same registers are drawn for both systems every replicate, absorbing
        between-system correlation. Reports the CI on the difference, a bootstrap
        p-value, and the minimum detectable effect (MDE = mde_factor * SE).
        """
        level = level or self.default_level
        a_reg = _register_stats(_load_per_item(a_path), level, subfolder)
        b_reg = _register_stats(_load_per_item(b_path), level, subfolder)
        registers = sorted(set(a_reg) & set(b_reg))
        if not registers:
            raise ValueError("No shared registers between the two systems")

        a = [a_reg[r] for r in registers]
        b = [b_reg[r] for r in registers]
        point = _micro(a) - _micro(b)

        R = len(registers)
        B = self.bootstrap_samples
        rng = np.random.default_rng(self.seed)
        diffs = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, R, size=R)
            diffs[i] = _micro([a[j] for j in idx]) - _micro([b[j] for j in idx])

        result = _summary(diffs, point)
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        result["p_value"] = float(min(1.0, max(p, 2 / B)))
        result["mde"] = float(self.mde_factor * result["se"])
        result["n_registers"] = R
        result["significant"] = result["ci_low"] > 0 or result["ci_high"] < 0
        return result

    def replicates(self, metrics_paths, level=None, subfolder=None):
        """
        Seed-replication summary: point micro-CER per run, then mean/std across runs.

        Measures optimization variance (training/decoding seeds), orthogonal to the
        register-cluster CI which measures test-sampling variance. Report both.
        """
        level = level or self.default_level
        points = []
        for path in metrics_paths:
            reg = _register_stats(_load_per_item(path), level, subfolder)
            points.append(_micro(list(reg.values())))
        return {
            "runs": len(points),
            "points": points,
            "mean": mean(points),
            "std": stdev(points) if len(points) > 1 else 0.0,
        }

    @staticmethod
    def holm(pvalues):
        """Holm-Bonferroni adjusted p-values, preserving input order."""
        order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
        m = len(pvalues)
        adjusted = [0.0] * m
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min((m - rank) * pvalues[i], 1.0))
            adjusted[i] = running
        return adjusted


def main():
    defaults = StatsSettings()
    parser = argparse.ArgumentParser(
        description="Register-cluster bootstrap CIs and paired contrasts."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ci = sub.add_parser("ci", help="CI for one system")
    p_ci.add_argument("--metrics", required=True)
    p_ci.add_argument("--level", default=defaults.default_level)
    p_ci.add_argument("--subfolder", default=None)
    p_ci.add_argument("-B", type=int, default=defaults.bootstrap_samples)

    p_pair = sub.add_parser("paired", help="Paired A - B contrast")
    p_pair.add_argument("--a", required=True)
    p_pair.add_argument("--b", required=True)
    p_pair.add_argument("--level", default=defaults.default_level)
    p_pair.add_argument("--subfolder", default=None)
    p_pair.add_argument("-B", type=int, default=defaults.bootstrap_samples)

    p_rep = sub.add_parser("replicates", help="Seed-replication mean/std")
    p_rep.add_argument("--metrics", nargs="+", required=True)
    p_rep.add_argument("--level", default=defaults.default_level)
    p_rep.add_argument("--subfolder", default=None)

    args = parser.parse_args()
    if args.cmd == "ci":
        out = RegisterBootstrap(bootstrap_samples=args.B).ci(
            args.metrics, args.level, args.subfolder
        )
    elif args.cmd == "paired":
        out = RegisterBootstrap(bootstrap_samples=args.B).paired(
            args.a, args.b, args.level, args.subfolder
        )
    else:
        out = RegisterBootstrap().replicates(args.metrics, args.level, args.subfolder)
    logger.info(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
