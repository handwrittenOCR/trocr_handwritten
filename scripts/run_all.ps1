# Full pipeline: YOLO crop + Gemini 3 Pro OCR for all communes and years
# Usage:
#   .\scripts\run_all.ps1                                  # all communes, all years
#   .\scripts\run_all.ps1 -Communes abymes,anse_bertrand   # specific communes
#   .\scripts\run_all.ps1 -Communes abymes -Years 1841,1842
#   .\scripts\run_all.ps1 -MaxConcurrent 5 -Timeout 240

param(
    [string[]]$Communes = @(),
    [int[]]$Years = @(),
    [int]$MaxConcurrent = 10,
    [int]$Timeout = 180,
    [string]$Model = "gemini-3-pro-preview"
)

$ErrorActionPreference = "Stop"

$VENV = ".venv\Scripts\python.exe"
$BASE_INPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\0. Brut\Guadeloupe"
$BASE_OUTPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed"

$HF_REPO = "MarieBgl/historical-layout-bagnards-EC"
$HF_FILE = "20250111_yolov10_bagnards_EC.pt"

# Auto-detect communes if not specified
if ($Communes.Count -eq 0) {
    $Communes = Get-ChildItem -Path $BASE_INPUT -Directory | Select-Object -ExpandProperty Name
}

$totalStart = Get-Date
$totalPages = 0
$totalCrops = 0
$totalTranscribed = 0
$totalFailed = 0

Write-Host "=========================================="
Write-Host "=== Full Pipeline ==="
Write-Host "=========================================="
Write-Host "Communes: $($Communes -join ', ')"
Write-Host "Model:    $Model"
Write-Host "Timeout:  ${Timeout}s | Workers: $MaxConcurrent"
Write-Host ""

foreach ($commune in $Communes) {
    $communeInput = Join-Path $BASE_INPUT $commune
    $communeOutput = Join-Path $BASE_OUTPUT $commune

    if (-not (Test-Path $communeInput)) {
        Write-Host "SKIP $commune - folder not found"
        continue
    }

    # Auto-detect years if not specified
    if ($Years.Count -eq 0) {
        $yearDirs = Get-ChildItem -Path $communeInput -Directory |
            Where-Object { $_.Name -match '^\d{4}' } |
            Select-Object -ExpandProperty Name |
            Sort-Object
    } else {
        $yearDirs = $Years | ForEach-Object { $_.ToString() }
    }

    Write-Host "=========================================="
    Write-Host "=== $commune ($($yearDirs.Count) years) ==="
    Write-Host "=========================================="
    $communeStart = Get-Date

    foreach ($year in $yearDirs) {
        $INPUT_DIR = Join-Path $communeInput "$year\pages"
        $OUTPUT_DIR = Join-Path $communeOutput "$year"

        if (-not (Test-Path $INPUT_DIR)) {
            Write-Host "  SKIP $year - no pages folder"
            continue
        }

        $pageCount = (Get-ChildItem -Path $INPUT_DIR -Filter "*.jpg").Count
        if ($pageCount -eq 0) {
            Write-Host "  SKIP $year - no jpg files"
            continue
        }

        $totalPages += $pageCount
        Write-Host ""
        Write-Host "--- $commune/$year ($pageCount pages) ---"
        $yearStart = Get-Date

        # Step 1: YOLO (skip if already done)
        $existingCrops = 0
        if (Test-Path $OUTPUT_DIR) {
            $existingCrops = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.jpg" -Recurse).Count
        }

        if ($existingCrops -gt 0) {
            Write-Host "  YOLO: skipping, $existingCrops crops already exist"
        } else {
            Write-Host "  YOLO: parsing..."
            & $VENV -m trocr_handwritten.parse.layout_parser `
                "$INPUT_DIR" `
                --output "$OUTPUT_DIR" `
                --hf-repo "$HF_REPO" `
                --hf-filename "$HF_FILE" `
                --classes Marge "Plein Texte" `
                --device cpu
        }

        # Step 2: OCR
        Write-Host "  OCR: transcribing..."
        & $VENV -m trocr_handwritten.llm.ocr `
            --provider gemini `
            --model "$Model" `
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

        $totalCrops += $crops
        $totalTranscribed += $transcribed
        $totalFailed += $failedCount

        Write-Host "  Done ($($elapsed.ToString('hh\:mm\:ss'))) - Crops: $crops | Transcribed: $transcribed | Failed: $failedCount"
    }

    $communeElapsed = (Get-Date) - $communeStart
    Write-Host ""
    Write-Host "=== $commune Done === ($($communeElapsed.ToString('hh\:mm\:ss')))"
    Write-Host ""
}

$totalElapsed = (Get-Date) - $totalStart
Write-Host "=========================================="
Write-Host "=== All Done === ($($totalElapsed.ToString('hh\:mm\:ss')))"
Write-Host "  Pages: $totalPages | Crops: $totalCrops | Transcribed: $totalTranscribed | Failed: $totalFailed"
Write-Host "  Cost log: logs\api_costs.jsonl"
