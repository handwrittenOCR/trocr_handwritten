import argparse
import os
import subprocess
import sys
from pathlib import Path

from trocr_handwritten.parse.settings import BenchmarkSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

DONE_MARKER = "TRAINING_DONE"


def run_logged(cmd, log_path, env=None):
    """
    Run a command, streaming its output to the console and to a log file.

    Args:
        cmd: Command as a list of strings
        log_path: Log file, overwritten
        env: Environment for the child process

    Returns:
        int: Return code
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def model_base(settings, name):
    """
    Model passed to parse.train for a run; DocLayout-YOLO weights are downloaded once.

    Args:
        settings: BenchmarkSettings instance
        name: Run name

    Returns:
        str: --model-base value
    """
    base = settings.runs[name]
    if base != "doclayout":
        return base
    path = Path(settings.weights_dir) / settings.doclayout_filename
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            settings.doclayout_repo,
            settings.doclayout_filename,
            local_dir=settings.weights_dir,
        )
    return str(path)


def child_env(name, settings):
    """
    Environment of the training and scoring processes of a run. DocLayout-YOLO 0.0.4
    reloads its checkpoints with torch.load without weights_only, which torch >= 2.6
    refuses unless TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set.

    Args:
        name: Run name
        settings: BenchmarkSettings instance

    Returns:
        dict: Environment variables
    """
    env = dict(os.environ)
    if settings.runs[name] == "doclayout":
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def train_run(name, settings):
    """
    Train one run unless its TRAINING_DONE marker exists. The marker is written only
    when training exits cleanly with a best.pt, so an interrupted run (which may
    already hold a best.pt) is trained again.

    Args:
        name: Run name
        settings: BenchmarkSettings instance

    Returns:
        bool: True when the run has a finished training
    """
    run_dir = Path(settings.path_data) / "models" / name
    best = run_dir / "weights" / "best.pt"
    if (run_dir / DONE_MARKER).exists() and best.exists():
        logger.info(f"skip training {name} (finished run found)")
        return True
    if best.exists():
        logger.warning(
            f"{name}: best.pt without {DONE_MARKER} (interrupted run), training again"
        )
    cmd = [
        sys.executable,
        "-m",
        "trocr_handwritten.parse.train",
        "--path-data",
        settings.path_data,
        "--images-dir",
        settings.images_dir,
        "--model-base",
        model_base(settings, name),
        "--name",
        name,
        *settings.extra_args.get(name, []),
    ]
    logger.info(f"===== training {name} =====")
    code = run_logged(
        cmd,
        Path(settings.path_data) / "logs" / f"train-{name}.log",
        child_env(name, settings),
    )
    if code != 0 or not best.exists():
        logger.error(f"{name}: training failed (exit code {code}), not scored")
        return False
    (run_dir / DONE_MARKER).write_text("ok\n")
    return True


def score_run(name, settings):
    """
    Score a run's best.pt on each split with parse.metrics, one JSON per split.

    Args:
        name: Run name
        settings: BenchmarkSettings instance

    Returns:
        list: Paths of the JSON reports written
    """
    best = Path(settings.path_data) / "models" / name / "weights" / "best.pt"
    written = []
    for split in settings.splits:
        out = (
            Path(settings.path_data) / "eval" / f"{name}_{split}_{settings.device}.json"
        )
        cmd = [
            sys.executable,
            "-m",
            "trocr_handwritten.parse.metrics",
            "--path-data",
            settings.path_data,
            "--split",
            split,
            "--model",
            str(best),
            "--device",
            settings.device,
            "--out-json",
            str(out),
        ]
        logger.info(f"===== scoring {name} on {split} =====")
        code = run_logged(
            cmd,
            Path(settings.path_data)
            / "logs"
            / f"eval-{name}-{split}-{settings.device}.log",
            child_env(name, settings),
        )
        if code == 0 and out.exists():
            written.append(out)
        else:
            logger.error(f"{name}: scoring on {split} failed (exit code {code})")
    return written


def main():
    """
    CLI: train and score the layout detectors of BenchmarkSettings.runs. Pass run
    names to restrict; --score-only re-scores finished runs without training (e.g.
    to time pages on a free GPU; any run with a best.pt is scored). Relaunching resumes:
    finished runs are skipped.
    """
    defaults = BenchmarkSettings()
    parser = argparse.ArgumentParser(description="Benchmark layout detectors")
    parser.add_argument(
        "runs",
        nargs="*",
        help=f"Runs to process (default: all of {list(defaults.runs)})",
    )
    parser.add_argument("--path-data", default=defaults.path_data)
    parser.add_argument("--images-dir", default=defaults.images_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    settings = BenchmarkSettings(
        path_data=args.path_data, images_dir=args.images_dir, device=args.device
    )
    names = args.runs or list(settings.runs)
    unknown = [n for n in names if n not in settings.runs]
    if unknown:
        parser.error(f"unknown runs {unknown}; known: {list(settings.runs)}")
    if settings.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            sys.exit(
                "CUDA not available to torch: see RUN_LAYOUT_BENCHMARK.md, section 2"
            )

    summary = {}
    for name in names:
        best = Path(settings.path_data) / "models" / name / "weights" / "best.pt"
        trained = best.exists() if args.score_only else train_run(name, settings)
        summary[name] = (
            [str(p) for p in score_run(name, settings)] if trained else "failed"
        )
    for name, result in summary.items():
        logger.info(f"{name}: {result}")
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
