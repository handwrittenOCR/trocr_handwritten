import argparse
import json
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from trocr_handwritten.labelstudio import LabelStudio, _env
from trocr_handwritten.utils.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


def parse_task(task: Dict):
    """
    Extract (filename, text, status) from an annotated task.

    Returns None if the task has no human annotation yet. Falls back to the
    pre-filled transcription when the annotation carries no text.
    """
    anns = task.get("annotations") or []
    if not anns:
        return None
    text, status = "", "ok"
    for res in anns[0].get("result", []):
        if res.get("from_name") == "transcription":
            vals = res.get("value", {}).get("text", [])
            text = vals[0] if vals else ""
        elif res.get("from_name") == "status":
            choices = res.get("value", {}).get("choices", [])
            status = choices[0] if choices else "ok"
    data = task.get("data", {})
    if not text:
        text = data.get("transcription", "")
    return {
        "filename": data.get("filename"),
        "text": (text or "").strip(),
        "status": status,
    }


def main():
    """CLI: pull transcriptions from Label Studio into a folder of .txt files."""
    parser = argparse.ArgumentParser(description="Export Label Studio transcriptions.")
    parser.add_argument(
        "--projects",
        type=int,
        nargs="+",
        required=True,
        help="Label Studio project ids to export.",
    )
    parser.add_argument(
        "--out",
        default="transcriptions",
        help="Output folder for <name>.txt files and transcriptions.json.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ls = LabelStudio(_env("LS_URL"), _env("LS_TOKEN"))

    results = []
    stats = {"written": 0, "rejected": 0, "skipped": 0}
    for pid in args.projects:
        for task in ls.export_tasks(pid):
            rec = parse_task(task)
            if rec is None or not rec["filename"]:
                stats["skipped"] += 1
                continue
            if rec["status"].startswith("reject") or not rec["text"]:
                stats["rejected"] += 1
                continue
            (out_dir / (Path(rec["filename"]).stem + ".txt")).write_text(
                rec["text"], encoding="utf-8"
            )
            results.append({"filename": rec["filename"], "text": rec["text"]})
            stats["written"] += 1

    (out_dir / "transcriptions.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"{stats} -> {out_dir}")


if __name__ == "__main__":
    main()
