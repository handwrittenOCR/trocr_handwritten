"""OCR of layout crops through the Gemini Batch API (half the standard price, results within 24 h).

Same request as the synchronous ocr.py: the OCR prompt and the full-resolution crop as inline
JPEG, same generation config (temperature, max tokens, thinking level). Only Marge / Plein Texte
crops without a transcription (.md) are sent. Crops are large (~450 KB), so the requests are split
into parts below the 2 GB batch-file limit; each part is one batch job. Work files go to
--work-dir (default D:/ECES_work/ocr_batch/<tag>), outside Dropbox.

    prepare   write the request parts + manifest, print the cost estimate; refuse above --budget.
              No API call.
    submit    upload each part not yet submitted and create its batch job (resumable)
    status    print the state of every part
    fetch     download finished parts, write <crop>.md next to each crop, log the cost once per
              part (batch rate), list the crops without a transcription

Usage (from the trocr_handwritten root):
    python -m trocr_handwritten.llm.ocr_batch prepare --tag tranche1 --budget 70 \\
        --input-dir "<ECES>/OCR_gem31/moule" "<ECES>/OCR_gem31/petit_canal"
    python -m trocr_handwritten.llm.ocr_batch submit --tag tranche1
    python -m trocr_handwritten.llm.ocr_batch status --tag tranche1
    python -m trocr_handwritten.llm.ocr_batch fetch --tag tranche1
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

from google.genai import types

from trocr_handwritten.llm.ocr import find_images, load_prompt
from trocr_handwritten.llm.providers.gemini import GeminiProvider
from trocr_handwritten.llm.settings import LLMSettings, OCRSettings
from trocr_handwritten.utils.cost_tracker import PRICING, CostTracker

WORK_DIR = Path("D:/ECES_work/ocr_batch")
PROMPT = Path(__file__).resolve().parents[2] / "config" / "ocr.prompt"
EST_IN, EST_OUT = 1220, 224
MAX_PART_BYTES = 1_800_000_000


def provider(model: str) -> GeminiProvider:
    """Gemini provider holding the client and the generation config of `model`."""
    return GeminiProvider(LLMSettings(provider="gemini", model_name=model))


def estimate_eur(model: str, n: int) -> float:
    """Batch cost estimate, EUR adjusted (x1.3), from tokens per crop of the gem31 pilot
    (input ~1,220; output + thinking ~224)."""
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return n * (EST_IN * p["input"] + EST_OUT * p["output"]) / 1e6 * 0.5 * 1.3


def save(d: Path, m: dict) -> None:
    """Write the manifest of a tag."""
    (d / "manifest.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load(tag: str, work_dir: Path) -> tuple[Path, dict]:
    """Folder and manifest of a prepared tag."""
    d = work_dir / tag
    if not (d / "manifest.json").exists():
        raise SystemExit(f"{d} not prepared")
    return d, json.loads((d / "manifest.json").read_text(encoding="utf-8"))


def cmd_prepare(a) -> None:
    """Write the request parts and the manifest. No API call."""
    d = a.work_dir / a.tag
    if (d / "manifest.json").exists():
        raise SystemExit(f"{d} already prepared; use another --tag")
    classes = OCRSettings().ocr_classes
    images = sorted(
        {
            p
            for root in a.input_dir
            for p in find_images(Path(root), "*.jpg", classes=classes)
        }
    )
    if not images:
        raise SystemExit("No crop without transcription.")
    est = estimate_eur(a.model, len(images))
    if a.budget and est > a.budget:
        raise SystemExit(
            f"{len(images)} crops, estimate EUR {est:.2f} adjusted > budget EUR {a.budget:.2f}: "
            "nothing prepared."
        )
    prompt = load_prompt(str(PROMPT))
    gen = (
        provider(a.model)
        ._get_config(a.model)
        .model_dump(mode="json", exclude_none=True, by_alias=True)
    )
    d.mkdir(parents=True, exist_ok=True)
    parts, f, size, n = [], None, 0, 0
    for img in images:
        line = (
            json.dumps(
                {
                    "key": str(img),
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": base64.b64encode(
                                                img.read_bytes()
                                            ).decode(),
                                        }
                                    },
                                ],
                            }
                        ],
                        "generationConfig": gen,
                    },
                }
            )
            + "\n"
        )
        if f is None or size + len(line) > MAX_PART_BYTES:
            if f:
                f.close()
                parts[-1]["n"] = n
            name = f"requests_{len(parts):03d}.jsonl"
            parts.append({"file": name, "n": 0, "batch_name": None, "fetched": False})
            f, size, n = open(d / name, "w", encoding="utf-8"), 0, 0
        f.write(line)
        size += len(line)
        n += 1
    f.close()
    parts[-1]["n"] = n
    m = {
        "tag": a.tag,
        "model": a.model,
        "prepared": datetime.now().isoformat(timespec="seconds"),
        "input_dirs": [str(x) for x in a.input_dir],
        "n_images": len(images),
        "estimate_eur_adjusted": round(est, 2),
        "parts": parts,
    }
    save(d, m)
    print(f"{len(images)} crops in {len(parts)} part(s) -> {d}")
    print(f"Estimate (batch, adjusted): EUR {est:.2f}")


def cmd_submit(a) -> None:
    """Upload every part not yet submitted and create its batch job; resumable after a failure."""
    d, m = load(a.tag, a.work_dir)
    prov = provider(m["model"])
    client = prov.native_client
    for p in m["parts"]:
        if p["batch_name"]:
            continue
        up = client.files.upload(
            file=str(d / p["file"]),
            config=types.UploadFileConfig(
                display_name=f"ocr_{a.tag}_{p['file']}", mime_type="jsonl"
            ),
        )
        job = client.batches.create(
            model=m["model"],
            src=up.name,
            config={"display_name": f"ocr_{a.tag}_{p['file']}"},
        )
        p["batch_name"] = job.name
        p["submitted"] = datetime.now().isoformat(timespec="seconds")
        save(d, m)
        print(f"{p['file']}: {p['n']} crops -> {job.name}")


def cmd_status(a) -> None:
    """Print the state of every part."""
    _, m = load(a.tag, a.work_dir)
    prov = provider(m["model"])
    client = prov.native_client
    for p in m["parts"]:
        state = (
            client.batches.get(name=p["batch_name"]).state.name
            if p["batch_name"]
            else "NOT SUBMITTED"
        )
        print(
            f"{p['file']}: {p['n']} crops, {state}{', fetched' if p['fetched'] else ''}"
        )


def _text(resp: dict) -> str:
    """The transcription of one batch response: text parts, thoughts left out."""
    out = []
    for c in resp.get("candidates") or []:
        for part in (c.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought"):
                out.append(part["text"])
    return "".join(out).strip()


def cmd_fetch(a) -> None:
    """Write the transcriptions of every finished part and log its cost once."""
    d, m = load(a.tag, a.work_dir)
    prov = provider(m["model"])
    client = prov.native_client
    failed = {}
    for p in m["parts"]:
        if not p["batch_name"] or p["fetched"]:
            continue
        job = client.batches.get(name=p["batch_name"])
        if job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"{p['file']}: {job.state.name}, not fetched")
            continue
        raw = client.files.download(file=job.dest.file_name).decode("utf-8")
        (d / p["file"].replace("requests_", "results_")).write_text(
            raw, encoding="utf-8"
        )
        tracker = CostTracker(model_name=m["model"], batch=True)
        written = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if row.get("error"):
                failed[key] = str(row["error"])[:300]
                continue
            resp = row.get("response") or {}
            u = resp.get("usageMetadata") or {}
            tracker.input_tokens += u.get("promptTokenCount", 0) or 0
            tracker.output_tokens += u.get("candidatesTokenCount", 0) or 0
            tracker.thinking_tokens += u.get("thoughtsTokenCount", 0) or 0
            tracker.total_calls += 1
            text = _text(resp)
            if not text:
                failed[key] = "empty_response"
                continue
            md = Path(key).with_suffix(".md")
            if not md.exists():
                md.write_text(text, encoding="utf-8")
                written += 1
        print(tracker.summary())
        tracker.log_summary()
        p["fetched"] = datetime.now().isoformat(timespec="seconds")
        p["written"] = written
        save(d, m)
        print(f"{p['file']}: {written} transcriptions written")
    if failed:
        (d / "failed.json").write_text(
            json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{len(failed)} crops without transcription, see {d / 'failed.json'}; "
            "a new prepare (another --tag) picks them up, since they still have no .md"
        )


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=WORK_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tag", required=True)
    p.add_argument("--input-dir", nargs="+", required=True)
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="refuse above this estimate (EUR adjusted)",
    )
    for name in ("submit", "status", "fetch"):
        sub.add_parser(name).add_argument("--tag", required=True)
    a = ap.parse_args()
    {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "fetch": cmd_fetch,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
