# Retry all failed OCR across all communes and years
# Budget is GLOBAL across the entire run (not per folder).
# Usage:
#   .\scripts\retry_all_failed.ps1                              # scan all communes
#   .\scripts\retry_all_failed.ps1 -Communes abymes             # specific commune
#   .\scripts\retry_all_failed.ps1 -Communes abymes -Years 1842 # specific year
#   .\scripts\retry_all_failed.ps1 -Timeout 240 -MaxConcurrent 5
#   .\scripts\retry_all_failed.ps1 -Budget 5                    # stop after EUR 5 total

param(
    [string[]]$Communes = @(),
    [int[]]$Years = @(),
    [int]$Timeout = 180,
    [int]$MaxConcurrent = 10,
    [string]$Model = "gemini-3.1-pro-preview",
    [int]$N = 0,
    [double]$Budget = 0.0
)

$VENV = ".venv\Scripts\python.exe"
$BASE_OUTPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed"
$COST_LOG = "logs\api_costs.jsonl"

# Auto-detect communes if not specified
if ($Communes.Count -eq 0) {
    if (Test-Path $BASE_OUTPUT) {
        $Communes = Get-ChildItem -Path $BASE_OUTPUT -Directory | Select-Object -ExpandProperty Name
    } else {
        Write-Host "No output directory found at $BASE_OUTPUT"
        exit 0
    }
}

$totalNeedsRetry = 0
$totalResolved = 0
$totalStillFailed = 0
$sessionCostEur = 0.0
$budgetExhausted = $false

# Count existing lines in cost log
$costLogLinesBefore = 0
if (Test-Path $COST_LOG) {
    $costLogLinesBefore = (Get-Content $COST_LOG).Count
}

Write-Host "=========================================="
Write-Host "=== Retry All Failed ==="
Write-Host "=========================================="
Write-Host "Model:   $Model"
Write-Host "Timeout: ${Timeout}s | Workers: $MaxConcurrent"
if ($Budget -gt 0) { Write-Host "Budget:  EUR $Budget (global)" }
Write-Host ""

# First pass: scan for all failed_ocr.json files
$retryTargets = @()

foreach ($commune in $Communes) {
    $communeDir = Join-Path $BASE_OUTPUT $commune
    if (-not (Test-Path $communeDir)) { continue }

    # Auto-detect years if not specified
    if ($Years.Count -eq 0) {
        $yearDirs = Get-ChildItem -Path $communeDir -Directory |
            Where-Object { $_.Name -match '^\d{4}' } |
            Select-Object -ExpandProperty Name |
            Sort-Object
    } else {
        $yearDirs = $Years | ForEach-Object { $_.ToString() }
    }

    foreach ($year in $yearDirs) {
        $yearDir = Join-Path $communeDir $year
        $failedJson = Join-Path $yearDir "failed_ocr.json"

        if (-not (Test-Path $failedJson)) { continue }

        $failed = Get-Content $failedJson | ConvertFrom-Json

        # Check which images still need processing (jpg must exist, md must not)
        $needsRetry = @{}
        foreach ($imgPath in $failed.images.PSObject.Properties) {
            $mdPath = $imgPath.Name -replace '\.jpg$', '.md'
            if ((Test-Path $imgPath.Name) -and -not (Test-Path $mdPath)) {
                $needsRetry[$imgPath.Name] = $imgPath.Value
            }
        }

        if ($needsRetry.Count -eq 0) {
            Write-Host "  $commune/$year - all resolved, cleaning up"
            Remove-Item $failedJson
            continue
        }

        # Update failed_ocr.json
        $updatedPre = @{
            timestamp = (Get-Date).ToString("o")
            model = $failed.model
            provider = $failed.provider
            failed_count = $needsRetry.Count
            images = $needsRetry
        }
        $updatedPre | ConvertTo-Json -Depth 3 | Set-Content $failedJson -Encoding UTF8

        $totalNeedsRetry += $needsRetry.Count
        $retryTargets += @{ commune = $commune; year = $year; dir = $yearDir; count = $needsRetry.Count }
        Write-Host "  $commune/$year - $($needsRetry.Count) to retry"
    }
}

if ($retryTargets.Count -eq 0) {
    Write-Host ""
    Write-Host "No failed images found. Everything is transcribed!"
    exit 0
}

Write-Host ""
Write-Host "Total to retry: $totalNeedsRetry across $($retryTargets.Count) year(s)"
Write-Host ""

# Second pass: retry each target
$remaining = $N
foreach ($target in $retryTargets) {
    if ($budgetExhausted) {
        Write-Host "--- $($target.commune)/$($target.year) - skipped (global budget exhausted) ---"
        continue
    }
    if ($N -gt 0 -and $remaining -le 0) {
        Write-Host "--- $($target.commune)/$($target.year) - skipped (image limit exhausted) ---"
        continue
    }

    Write-Host "--- $($target.commune)/$($target.year) ($($target.count) images) ---"

    $ocrArgs = @(
        "-m", "trocr_handwritten.llm.ocr",
        "--provider", "gemini",
        "--model", $Model,
        "--input_dir", $target.dir,
        "--pattern", "*.jpg",
        "--max_concurrent", $MaxConcurrent,
        "--timeout", $Timeout
    )
    if ($N -gt 0) {
        $ocrArgs += @("-n", $remaining)
    }
    # Pass remaining budget to each ocr.py call
    if ($Budget -gt 0) {
        $remainingBudget = [math]::Round($Budget - $sessionCostEur, 4)
        if ($remainingBudget -le 0) {
            Write-Host "  SKIP - global budget exhausted"
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

    # Check results
    $failedJson = Join-Path $target.dir "failed_ocr.json"
    if (-not (Test-Path $failedJson)) {
        $totalResolved += $target.count
        if ($N -gt 0) { $remaining -= $target.count }
        Write-Host "  All resolved!"
        continue
    }

    $failedData = Get-Content $failedJson | ConvertFrom-Json
    $stillFailed = @{}
    $resolved = 0

    foreach ($imgPath in $failedData.images.PSObject.Properties) {
        $mdPath = $imgPath.Name -replace '\.jpg$', '.md'
        if (Test-Path $mdPath) {
            $resolved++
        } else {
            $stillFailed[$imgPath.Name] = $imgPath.Value
        }
    }

    $totalResolved += $resolved
    $totalStillFailed += $stillFailed.Count
    if ($N -gt 0) { $remaining -= $resolved }

    if ($stillFailed.Count -eq 0) {
        Remove-Item $failedJson
        Write-Host "  All resolved!"
    } else {
        $updatedData = @{
            timestamp = (Get-Date).ToString("o")
            model = $failedData.model
            provider = $failedData.provider
            failed_count = $stillFailed.Count
            images = $stillFailed
        }
        $updatedData | ConvertTo-Json -Depth 3 | Set-Content $failedJson -Encoding UTF8
        Write-Host "  Resolved: $resolved | Still failed: $($stillFailed.Count)"
    }
    Write-Host ""
}

Write-Host "=========================================="
Write-Host "=== Retry Summary ==="
Write-Host "  Resolved: $totalResolved | Still failed: $totalStillFailed"
if ($Budget -gt 0) {
    Write-Host "  Session cost: EUR $([math]::Round($sessionCostEur, 2)) / $Budget"
}
Write-Host "  Cost log: logs\api_costs.jsonl"
