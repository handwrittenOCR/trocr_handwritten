# Full pipeline: YOLO crop + Gemini 3 Pro OCR for all communes and years
# Budget is GLOBAL across the entire run (not per folder).
# Usage:
#   .\scripts\run_all.ps1                                  # all communes, all years
#   .\scripts\run_all.ps1 -Communes abymes,anse_bertrand   # specific communes
#   .\scripts\run_all.ps1 -Communes abymes -Years 1841,1842
#   .\scripts\run_all.ps1 -MaxConcurrent 5 -Timeout 240
#   .\scripts\run_all.ps1 -Budget 20                       # stop after EUR 20 total

param(
    [string[]]$Communes = @(),
    [int[]]$Years = @(),
    [int]$MaxConcurrent = 10,
    [int]$Timeout = 180,
    [string]$Model = "gemini-3.1-pro-preview",
    [int]$N = 0,
    [double]$Budget = 0.0
)

$ErrorActionPreference = "Stop"

$VENV = ".venv\Scripts\python.exe"
$BASE_INPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Images\Guadeloupe"
$BASE_OUTPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\OCR_gem31"

$HF_REPO = "MarieBgl/historical-layout-bagnards-EC"
$HF_FILE = "20250111_yolov10_bagnards_EC.pt"

$COST_LOG = "logs\api_costs.jsonl"

# Auto-detect communes if not specified
if ($Communes.Count -eq 0) {
    $Communes = Get-ChildItem -Path $BASE_INPUT -Directory | Select-Object -ExpandProperty Name
}

$totalStart = Get-Date
$totalPages = 0
$totalCrops = 0
$totalTranscribed = 0
$totalFailed = 0
$sessionCostEur = 0.0
$budgetExhausted = $false

# Count existing lines in cost log (to read only new entries after each OCR call)
$costLogLinesBefore = 0
if (Test-Path $COST_LOG) {
    $costLogLinesBefore = (Get-Content $COST_LOG).Count
}

Write-Host "=========================================="
Write-Host "=== Full Pipeline ==="
Write-Host "=========================================="
Write-Host "Communes: $($Communes -join ', ')"
Write-Host "Model:    $Model"
Write-Host "Timeout:  ${Timeout}s | Workers: $MaxConcurrent"
if ($N -gt 0) { Write-Host "Limit:    $N images" }
if ($Budget -gt 0) { Write-Host "Budget:   EUR $Budget (global for entire run)" }
Write-Host ""

$remaining = $N
foreach ($commune in $Communes) {
    if ($budgetExhausted) { break }

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
        if ($budgetExhausted) {
            Write-Host "  SKIP $year - global budget exhausted (EUR $([math]::Round($sessionCostEur, 2)) spent)"
            continue
        }
        if ($N -gt 0 -and $remaining -le 0) {
            Write-Host "  SKIP $year - image limit exhausted"
            continue
        }

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

        # Step 2: OCR - compute remaining budget for this folder
        $mdBefore = 0
        if (Test-Path $OUTPUT_DIR) {
            $mdBefore = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.md" -Recurse).Count
        }

        Write-Host "  OCR: transcribing..."
        $ocrArgs = @(
            "-m", "trocr_handwritten.llm.ocr",
            "--provider", "gemini",
            "--model", $Model,
            "--input_dir", $OUTPUT_DIR,
            "--pattern", "*.jpg",
            "--max_concurrent", $MaxConcurrent,
            "--timeout", $Timeout
        )
        if ($N -gt 0) {
            $ocrArgs += @("-n", $remaining)
        }
        # Pass remaining budget (global minus already spent) to each ocr.py call
        if ($Budget -gt 0) {
            $remainingBudget = [math]::Round($Budget - $sessionCostEur, 4)
            if ($remainingBudget -le 0) {
                Write-Host "  SKIP OCR - global budget exhausted"
                $budgetExhausted = $true
                continue
            }
            $ocrArgs += @("--budget", $remainingBudget)
            Write-Host "  Budget remaining: EUR $remainingBudget"
        }
        & $VENV @ocrArgs

        # Read new cost log entries to update session total
        if ($Budget -gt 0 -and (Test-Path $COST_LOG)) {
            $allLines = Get-Content $COST_LOG
            $newLines = $allLines[$costLogLinesBefore..($allLines.Count - 1)]
            $costLogLinesBefore = $allLines.Count
            foreach ($line in $newLines) {
                if (-not $line) { continue }
                $entry = $line | ConvertFrom-Json
                if ($entry.cost_total_adjusted_eur) {
                    $sessionCostEur += $entry.cost_total_adjusted_eur
                } elseif ($entry.cost_total_eur) {
                    $sessionCostEur += $entry.cost_total_eur * 1.30
                }
            }
            Write-Host "  Session cost so far: EUR $([math]::Round($sessionCostEur, 2)) / $Budget"
            if ($sessionCostEur -ge $Budget) {
                Write-Host "  *** GLOBAL BUDGET REACHED - stopping pipeline ***"
                $budgetExhausted = $true
            }
        }

        # Year summary
        $crops = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.jpg" -Recurse).Count
        $transcribed = (Get-ChildItem -Path $OUTPUT_DIR -Filter "*.md" -Recurse).Count
        $newlyDone = $transcribed - $mdBefore
        $failedJson = Join-Path $OUTPUT_DIR "failed_ocr.json"
        $failedCount = 0
        if (Test-Path $failedJson) {
            $failedCount = (Get-Content $failedJson | ConvertFrom-Json).failed_count
        }
        $elapsed = (Get-Date) - $yearStart

        $totalCrops += $crops
        $totalTranscribed += $transcribed
        $totalFailed += $failedCount
        if ($N -gt 0) { $remaining -= $newlyDone }

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
if ($Budget -gt 0) {
    Write-Host "  Session cost: EUR $([math]::Round($sessionCostEur, 2)) / $Budget"
}
Write-Host "  Cost log: logs\api_costs.jsonl"
