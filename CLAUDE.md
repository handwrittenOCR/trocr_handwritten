# CLAUDE.md

## Project Overview

**trocr_handwritten** is an end-to-end pipeline for transcribing 19th-century French handwritten civil registries (slaves' civil registries from Guadeloupe). The pipeline has three stages: layout parsing (YOLO), optional image preprocessing, and OCR transcription (LLM-based).

## Current Objective

Transcribe thousands of civil registry pages stored at:
```
C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\0. Brut\Guadeloupe\
```
Each commune folder (abymes, anse_bertrand, etc.) contains year subfolders (1841-1848), each with a `pages/` subfolder of `.jpg` scans.

### Pipeline Steps

1. **Layout parsing (YOLO)** — Crop pages into regions. We only care about **Marge** (class 2) and **Plein Texte** (class 4); other classes can be ignored.
2. **Preprocessing (optional)** — Greyscale + CLAHE contrast + adaptive binarization via `image_process.py`.
3. **OCR (Gemini 3 Flash)** — Transcribe cropped regions using `gemini-3-flash` through the Google API (provider: `gemini`).

### Test Data

First test: 2 pages from `abymes/1842/pages/` (files: `FRAD971_1E35_002_101_002_C.jpg`, `FRAD971_1E35_002_101_003_C.jpg`).

## Architecture

```
trocr_handwritten/
├── parse/              # YOLO layout parsing
│   ├── layout_parser.py  # Main entry point
│   ├── settings.py       # LayoutParserSettings, CLASS_NAMES (8 classes)
│   └── utils.py          # YOLOModel, crop creation
├── llm/                # LLM-based OCR
│   ├── ocr.py            # Async OCR pipeline (main entry point)
│   ├── settings.py       # LLMSettings, OCRSettings
│   ├── base.py           # Abstract LLMProvider with retry/fallback
│   ├── factory.py        # get_provider()
│   └── providers/        # openai.py, gemini.py, mistral.py
├── image_process.py    # Preprocessing (greyscale, CLAHE, binarization)
├── utils/              # Logging, cost tracking, S3, annotation helpers
└── trocr/              # TrOCR fine-tuned model (not used in current pipeline)
```

## Key Commands

```bash
# Layout parsing
python -m trocr_handwritten.parse.layout_parser <input_dir>

# LLM OCR
python -m trocr_handwritten.llm.ocr --provider gemini --model gemini-3-flash --input_dir <dir>

# Full pipeline
make run          # runs yolo + llm
make preprocess   # image preprocessing only
```

## Configuration

- **Layout parser settings**: `trocr_handwritten/parse/settings.py` — `LayoutParserSettings` dataclass
- **LLM/OCR settings**: `trocr_handwritten/llm/settings.py` — `LLMSettings` + `OCRSettings` (Pydantic)
- **API keys**: `.env` file with `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `HUGGINGFACE_API_KEY`
- **OCR prompt**: `config/ocr.prompt` (French transcription instructions)
- **Default YOLO model**: `MarieBgl/historical-layout-bagnards-EC` on HuggingFace (file: `20250111_yolov10_bagnards_EC.pt`)

## Rules

- Use `uv` for dependency management (not Poetry — migration already done).
- Format code with `black` (target: py311). Pre-commit hooks are configured.
- Run tests with `pytest`.
- Keep settings in their respective `settings.py` files (Pydantic for llm/, dataclass for parse/).
- LLM provider for this project: **Gemini** (via OpenAI-compatible SDK).
- Target model: **gemini-3.1-pro-preview** (best OCR quality for handwritten documents).
- Only crop **Marge** and **Plein Texte** classes from YOLO output; ignore Title, En-tete, Nom, Signature, Table, Section.
- All paths on this Windows machine use forward slashes in shell commands.
- Do not delete any file
- Never re-run LLM API calls when results already exist on disk. Always check for and load saved JSON/CSV outputs first (e.g. `ner_llm.json`, `ner_regex.json`, `acts_dataset.json`).
- Start a log at the beginning of each session and save the log at the end of the session in logs/
- **ALWAYS call `cost_tracker.log_summary(log_dir="logs")` at the end of every script that makes LLM API calls.** Never write a script with API calls without this line — it appends to `logs/api_costs.jsonl` and is the only way to track costs.
