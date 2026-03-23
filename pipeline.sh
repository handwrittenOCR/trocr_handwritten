#!/bin/bash
set -e  # stop if either step fails

python trocr_handwritten/parse/layout_parser.py "$@"
python trocr_handwritten/llm/ocr.py "$@"
