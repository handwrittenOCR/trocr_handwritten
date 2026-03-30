# Retry all failed OCR across all communes and years
# Usage:
#   .\scripts\retry_all_failed.ps1                              # scan all communes
#   .\scripts\retry_all_failed.ps1 -Communes abymes             # specific commune
#   .\scripts\retry_all_failed.ps1 -Communes abymes -Years 1842 # specific year
#   .\scripts\retry_all_failed.ps1 -Timeout 240 -MaxConcurrent 5

param(
    [string[]]$Communes = @(),
    [int[]]$Years = @(),
    [int]$Timeout = 180,
    [int]$MaxConcurrent = 10,
    [string]$Model = "gemini-3-pro-preview"
)

$VENV = ".venv\Scripts\python.exe"
$BASE_OUTPUT = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed"

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

Write-Host "=========================================="
Write-Host "=== Retry All Failed ==="
Write-Host "=========================================="
Write-Host "Model:   $Model"
Write-Host "Timeout: ${Timeout}s | Workers: $MaxConcurrent"
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

        # Check which images still need processing
        $needsRetry = @{}
        foreach ($imgPath in $failed.images.PSObject.Properties) {
            $mdPath = $imgPath.Name -replace '\.jpg$', '.md'
            if (-not (Test-Path $mdPath)) {
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
foreach ($target in $retryTargets) {
    Write-Host "--- $($target.commune)/$($target.year) ($($target.count) images) ---"

    & $VENV -m trocr_handwritten.llm.ocr `
        --provider gemini `
        --model "$Model" `
        --input_dir "$($target.dir)" `
        --pattern "*.jpg" `
        --max_concurrent $MaxConcurrent `
        --timeout $Timeout

    # Check results
    $failedJson = Join-Path $target.dir "failed_ocr.json"
    if (-not (Test-Path $failedJson)) {
        $totalResolved += $target.count
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
Write-Host "  Cost log: logs\api_costs.jsonl"
