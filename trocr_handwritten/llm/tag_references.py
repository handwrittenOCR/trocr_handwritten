import argparse
import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from trocr_handwritten.llm.metrics import NORMALIZERS
from trocr_handwritten.llm.providers.gemini import GeminiProvider
from trocr_handwritten.llm.settings import LLMSettings
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

POS_TAGS = "NOUN PROPN VERB AUX ADJ ADV NUM DET ADP PRON CCONJ SCONJ INTJ PART SYM X"
NER_TAGS = "PER LOC ORG DATE NUM PROF MISC O"

PROMPT = """Tu annotes des transcriptions de registres d'état civil et de bagne \
des Antilles françaises du XIXe siècle.

On te donne une liste de tokens numérotés. Pour CHAQUE token, donne une étiquette \
morphosyntaxique et une étiquette d'entité nommée.

Étiquettes morphosyntaxiques (POS) : {pos}
Étiquettes d'entités (NER) : {ner}
  PER  = nom ou prénom de personne
  LOC  = lieu (commune, pays, quartier, île)
  ORG  = institution, tribunal, administration
  DATE = tout composant d'une date (jour, mois, année)
  NUM  = nombre qui n'est pas une date (taille, numéro d'écrou, âge)
  PROF = métier ou profession
  MISC = autre entité nommée
  O    = token ordinaire

Réponds UNIQUEMENT par un tableau JSON, un objet par token, dans l'ordre, sans \
texte autour :
[{{"i": 0, "pos": "DET", "ner": "O"}}, {{"i": 1, "pos": "PROPN", "ner": "PER"}}]

Tokens :
{tokens}"""


def tokenize(text, level="N2"):
    """Split a reference into the same whitespace tokens the WER alignment uses."""
    return NORMALIZERS[level](text).split()


def collect_labels(labels_dir, label_ext=".txt"):
    """Return {key: token list} for every non-empty label file, key = subfolder/stem."""
    root = Path(labels_dir)
    out = {}
    for f in sorted(root.rglob(f"*{label_ext}")):
        rel = f.relative_to(root)
        subfolder = str(rel.parent) if rel.parent != Path(".") else ""
        tokens = tokenize(f.read_text(encoding="utf-8"))
        if tokens:
            out[f"{subfolder}/{f.stem}"] = tokens
    return out


def parse_tags(raw, n_tokens):
    """
    Read the model's JSON answer into per-index (pos, ner) pairs.

    Indices absent from the answer fall back to ("X", "O") so the returned list
    always aligns positionally with the reference tokens.
    """
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return None
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tags = [("X", "O")] * n_tokens
    for row in rows:
        try:
            i = int(row["i"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n_tokens:
            tags[i] = (str(row.get("pos", "X")), str(row.get("ner", "O")))
    return tags


async def tag_one(client, model, key, tokens, semaphore, max_retries=3):
    """Tag a single reference, retrying on transport or parse failure."""
    listing = "\n".join(f"{i}\t{t}" for i, t in enumerate(tokens))
    prompt = PROMPT.format(pos=POS_TAGS, ner=NER_TAGS, tokens=listing)
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                tags = parse_tags(resp.choices[0].message.content, len(tokens))
                if tags is not None:
                    return key, tags
                logger.warning(f"{key}: unparsable answer (attempt {attempt + 1})")
            except Exception as exc:
                logger.warning(f"{key}: {type(exc).__name__} (attempt {attempt + 1})")
            await asyncio.sleep(2**attempt)
    logger.error(f"{key}: giving up, tagged as X/O")
    return key, [("X", "O")] * len(tokens)


async def run(labels, model, api_key, max_concurrent):
    """Tag every reference concurrently and return {key: [[token, pos, ner], ...]}."""
    client = AsyncOpenAI(api_key=api_key, base_url=GeminiProvider.GEMINI_BASE_URL)
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        tag_one(client, model, key, tokens, semaphore) for key, tokens in labels.items()
    ]
    out = {}
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="tagging"):
        key, tags = await coro
        out[key] = [[tok, pos, ner] for tok, (pos, ner) in zip(labels[key], tags)]
    return out


def main():
    """
    Annotate reference transcriptions with POS and NER labels using an LLM.

    Written once to a cache file and reused by `error_analysis --tags`, so the
    API is paid for a single time per split. Tokenisation is the N2 whitespace
    split, identical to the one the WER alignment uses, which lets the analysis
    join errors to labels by token index without any realignment.
    """
    settings = LLMSettings()
    ap = argparse.ArgumentParser(description="Tag reference texts with POS and NER.")
    ap.add_argument("--labels", required=True, help="Directory with label files")
    ap.add_argument("--output", required=True, help="JSON cache to write")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--label-ext", default=".txt")
    ap.add_argument("--max-concurrent", type=int, default=16)
    args = ap.parse_args()

    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is not set")

    labels = collect_labels(args.labels, args.label_ext)
    logger.info(f"Tagging {len(labels)} references with {args.model}")
    tagged = asyncio.run(
        run(labels, args.model, settings.google_api_key, args.max_concurrent)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")

    n_tokens = sum(len(v) for v in tagged.values())
    unknown = sum(1 for v in tagged.values() for _, pos, _ in v if pos == "X")
    logger.info(
        f"Wrote {out} -- {len(tagged)} documents, {n_tokens} tokens, "
        f"{unknown} untagged ({100 * unknown / max(n_tokens, 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
