# Full pipeline: YOLO crop + Gemini 3 Flash OCR for all Abymes years
# Usage: .\.venv\Scripts\Activate.ps1; .\scripts\run_abymes_all.ps1
# To run a single year: .\scripts\run_abymes_all.ps1 -Years 1843

param(

    [int[]]$Years = @(1841, 1843, 1844, 1845, 1846, 1847, 1848),
    [int]$MaxConcurrent = 10,
    [int]$Timeout = 180
)

$ErrorActionPreference = "Stop"

$VENV = ".venv\Scripts\python.exe"
$BASE_INPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\0. Brut\Guadeloupe\abymes"
$BASE_OUTPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes"

$HF_REPO = "MarieBgl/historical-layout-bagnards-EC"
$HF_FILE = "20250111_yolov10_bagnards_EC.pt"
$MODEL = "gemini-3-pro-preview"

$totalStart = Get-Date
Write-Host "=== Abymes Full Pipeline ==="
Write-Host "Years: $($Years -join ', ')"
Write-Host "Timeout: ${Timeout}s | Workers: $MaxConcurrent"
Write-Host ""

foreach ($year in $Years) {
    $INPUT_DIR = Join-Path $BASE_INPUT "$year\pages"
    $OUTPUT_DIR = Join-Path $BASE_OUTPUT "$year"

    if (-not (Test-Path $INPUT_DIR)) {
        Write-Host "SKIP $year - no pages folder found"
        continue
    }

    $pageCount = (Get-ChildItem -Path $INPUT_DIR -Filter "*.jpg").Count
    Write-Host "=========================================="
    Write-Host "=== $year ($pageCount pages) ==="
    Write-Host "=========================================="
    $yearStart = Get-Date

    # Step 1: YOLO (skip if already done)
    $existingCrops = 0
    if (Test-Path $OUTPUT_DIR) {
        $existingCrops = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.jpg" -Recurse).Count
    }

    if ($existingCrops -gt 0) {
        Write-Host "--- YOLO: skipping, $existingCrops crops already exist ---"
    } else {
        Write-Host "--- YOLO Layout Parsing ---"
        & $VENV -m trocr_handwritten.parse.layout_parser `
            "$INPUT_DIR" `
            --output "$OUTPUT_DIR" `
            --hf-repo "$HF_REPO" `
            --hf-filename "$HF_FILE" `
            --classes Marge "Plein Texte" `
            --device cpu
    }

    # Step 2: OCR
    Write-Host "--- Gemini OCR ---"
    & $VENV -m trocr_handwritten.llm.ocr `
        --provider gemini `
        --model "$MODEL" `
        --input_dir "$OUTPUT_DIR" `
        --pattern "*.jpg" `
        --max_concurrent $MaxConcurrent `
        --timeout $Timeout

    # Year summary
    $crops = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.jpg" -Recurse).Count
    $transcribed = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.md" -Recurse).Count
    $failedJson = Join-Path $OUTPUT_DIR "failed_ocr.json"
    $failedCount = 0
    if (Test-Path $failedJson) {
        $failedCount = (Get-Content $failedJson | ConvertFrom-Json).failed_count
    }
    $elapsed = (Get-Date) - $yearStart

    Write-Host ""
    Write-Host "=== $year Done === ($($elapsed.ToString('hh\:mm\:ss')))"
    Write-Host "  Crops: $crops | Transcribed: $transcribed | Failed: $failedCount"
    Write-Host ""
}

$totalElapsed = (Get-Date) - $totalStart
Write-Host "=========================================="
Write-Host "=== All Done === ($($totalElapsed.ToString('hh\:mm\:ss')))"
Write-Host "Cost log: logs\api_costs.jsonl"
