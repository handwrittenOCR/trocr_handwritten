#!/bin/bash
# Full pipeline: YOLO crop + Gemini 3 Flash OCR for Abymes 1842
# Usage: bash scripts/run_abymes_1842.sh

set -e

# Use .venv python; fall back to system python
if [ -f ".venv/Scripts/python.exe" ]; then
    VENV=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    VENV=".venv/bin/python"
else
    VENV="python"
fi
INPUT_DIR="C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/3. OCR/2. TrOCR/5. Data (output)/ECES/0. Brut/Guadeloupe/abymes_OCRed_05032024/1842/pages"
OUTPUT_DIR="C:/Users/marie/Dropbox/Personnelle/2. Travail/1. Recherche/3. JMP/3. OCR/2. TrOCR/5. Data (output)/ECES/Gemini3_transcribed/abymes/1842"

HF_REPO="MarieBgl/historical-layout-bagnards-EC"
HF_FILE="20250111_yolov10_bagnards_EC.pt"
MODEL="gemini-3-flash-preview"
MAX_CONCURRENT=15
TIMEOUT=60

echo "=== Abymes 1842 Pipeline ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: YOLO layout parsing (Marge + Plein Texte only)
echo "=== Step 1: YOLO Layout Parsing ==="
date
$VENV -m trocr_handwritten.parse.layout_parser \
    "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --hf-repo "$HF_REPO" \
    --hf-filename "$HF_FILE" \
    --classes Marge "Plein Texte" \
    --device cpu
echo "=== YOLO Done ==="
date
echo ""

# Step 2: Gemini 3 Flash OCR
echo "=== Step 2: Gemini OCR ==="
$VENV -m trocr_handwritten.llm.ocr \
    --provider gemini \
    --model "$MODEL" \
    --input_dir "$OUTPUT_DIR" \
    --pattern "*.jpg" \
    --max_concurrent "$MAX_CONCURRENT" \
    --timeout "$TIMEOUT"
echo "=== OCR Done ==="
date
echo ""

# Summary
TOTAL_CROPS=$(find "/c${OUTPUT_DIR#C:}" -name "*.jpg" | wc -l)
TOTAL_MD=$(find "/c${OUTPUT_DIR#C:}" -name "*.md" | wc -l)
echo "=== Summary ==="
echo "Total crops: $TOTAL_CROPS"
echo "Transcribed: $TOTAL_MD"
echo "Cost log: logs/api_costs.jsonl"
