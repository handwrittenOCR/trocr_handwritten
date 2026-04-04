# Full pipeline: YOLO crop + Gemini 3 Flash OCR for Abymes 1842
# Usage: .\.venv\Scripts\Activate.ps1; .\scripts\run_abymes_1842.ps1

$ErrorActionPreference = "Stop"

$VENV = ".venv\Scripts\python.exe"
$INPUT_DIR = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\0. Brut\Guadeloupe\abymes\1842\pages"
$OUTPUT_DIR = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes\1842"

$HF_REPO = "MarieBgl/historical-layout-bagnards-EC"
$HF_FILE = "20250111_yolov10_bagnards_EC.pt"
$MODEL = "gemini-3-pro-preview"
$MAX_CONCURRENT = 15
$TIMEOUT = 60

Write-Host "=== Abymes 1842 Pipeline ==="
Write-Host "Input:  $INPUT_DIR"
Write-Host "Output: $OUTPUT_DIR"
Write-Host ""

# Step 1: YOLO layout parsing (Marge + Plein Texte only)
Write-Host "=== Step 1: YOLO Layout Parsing ==="
Get-Date
& $VENV -m trocr_handwritten.parse.layout_parser `
    "$INPUT_DIR" `
    --output "$OUTPUT_DIR" `
    --hf-repo "$HF_REPO" `
    --hf-filename "$HF_FILE" `
    --classes Marge "Plein Texte" `
    --device cpu
Write-Host "=== YOLO Done ==="
Get-Date
Write-Host ""

# Step 2: Gemini 3 Flash OCR
Write-Host "=== Step 2: Gemini OCR ==="
& $VENV -m trocr_handwritten.llm.ocr `
    --provider gemini `
    --model "$MODEL" `
    --input_dir "$OUTPUT_DIR" `
    --pattern "*.jpg" `
    --max_concurrent "$MAX_CONCURRENT" `
    --timeout "$TIMEOUT"
Write-Host "=== OCR Done ==="
Get-Date
Write-Host ""

# Summary
$TOTAL_CROPS = (Get-ChildItem -Path "$OUTPUT_DIR" -Filter "*.jpg" -Recurse).Count
$TOTAL_MD = (Get-ChildItem -Path "$OUTPUT_DIR" -Filter "*.md" -Recurse).Count
Write-Host "=== Summary ==="
Write-Host "Total crops: $TOTAL_CROPS"
Write-Host "Transcribed: $TOTAL_MD"
Write-Host "Cost log: logs\api_costs.jsonl"
