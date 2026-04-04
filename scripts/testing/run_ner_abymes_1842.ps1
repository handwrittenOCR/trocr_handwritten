# NER extraction pipeline for Abymes 1842
# Usage: .\.venv\Scripts\Activate.ps1; .\scripts\run_ner_abymes_1842.ps1
#
# Steps:
#   1. Build dataset from OCR transcriptions (free, instant)
#   2. Regex extraction (free, instant)
#   3. LLM extraction with Gemini 3 Flash (228 acts, ~$0.25, ~10 min)
#   4. Compare regex vs LLM
#   5. Merge into consolidated CSV

$ErrorActionPreference = "Stop"

$VENV = ".venv\Scripts\python.exe"
$INPUT_DIR = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes\1842"
$OUTPUT_DIR = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\NER_datasets"
$NER_DIR = "$INPUT_DIR\ner"

$MODEL = "gemini-3-flash-preview"
$MAX_CONCURRENT = 5
$TIMEOUT = 180

Write-Host "=== NER Pipeline: Abymes 1842 ==="
Write-Host "Input:  $INPUT_DIR"
Write-Host "Model:  $MODEL (concurrency: $MAX_CONCURRENT, timeout: ${TIMEOUT}s)"
Write-Host ""

# Step 1-4: Pipeline (dataset + regex + LLM + compare)
Write-Host "=== Running NER Pipeline ==="
Get-Date
& $VENV -m trocr_handwritten.ner.pipeline `
    --input_dir "$INPUT_DIR" `
    --commune abymes `
    --year 1842 `
    --methods regex llm `
    --model "$MODEL" `
    --max_concurrent $MAX_CONCURRENT `
    --timeout $TIMEOUT
Write-Host ""

# Step 5: Merge into consolidated CSV
Write-Host "=== Merging Results ==="
& $VENV -m trocr_handwritten.ner.merge `
    --ner_dir "$NER_DIR" `
    --output_dir "$OUTPUT_DIR" `
    --commune abymes `
    --year 1842
Write-Host ""

Get-Date
Write-Host "=== Done ==="
Write-Host "Final dataset: $OUTPUT_DIR\abymes_1842.csv"
Write-Host "Cost log: logs\api_costs.jsonl"
