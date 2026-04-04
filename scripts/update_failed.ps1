# Update failed_ocr.json for all communes/years based on disk state
# Scans each year folder: any .jpg without a .md is marked as failed
# Usage:
#   .\scripts\update_failed.ps1                              # scan all communes
#   .\scripts\update_failed.ps1 -Communes abymes             # specific commune
#   .\scripts\update_failed.ps1 -Communes abymes -Years 1842 # specific year

param(
    [string[]]$Communes = @(),
    [int[]]$Years = @()
)

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

$totalJpg = 0
$totalMd = 0
$totalFailed = 0
$totalCleaned = 0

Write-Host "=========================================="
Write-Host "=== Update Failed OCR ==="
Write-Host "=========================================="
Write-Host ""

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
        if (-not (Test-Path $yearDir)) { continue }

        $jpgs = Get-ChildItem -Path $yearDir -Filter "*.jpg" -Recurse
        $jpgCount = $jpgs.Count
        if ($jpgCount -eq 0) { continue }

        # Find jpgs without corresponding md
        $missing = @{}
        foreach ($jpg in $jpgs) {
            $mdPath = $jpg.FullName -replace '\.jpg$', '.md'
            if (-not (Test-Path $mdPath)) {
                $missing[$jpg.FullName] = "no corresponding output file"
            }
        }

        $mdCount = $jpgCount - $missing.Count
        $totalJpg += $jpgCount
        $totalMd += $mdCount

        $failedJson = Join-Path $yearDir "failed_ocr.json"

        if ($missing.Count -gt 0) {
            $failedData = @{
                timestamp = (Get-Date).ToString("o")
                failed_count = $missing.Count
                images = $missing
            }
            $failedData | ConvertTo-Json -Depth 3 | Set-Content $failedJson -Encoding UTF8
            $totalFailed += $missing.Count
            Write-Host "  $commune/$year - $mdCount/$jpgCount transcribed, $($missing.Count) failed"
        } else {
            if (Test-Path $failedJson) {
                Remove-Item $failedJson
                $totalCleaned++
                Write-Host "  $commune/$year - $jpgCount/$jpgCount transcribed (cleaned stale failed_ocr.json)"
            } else {
                Write-Host "  $commune/$year - $jpgCount/$jpgCount transcribed"
            }
        }
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "=== Summary ==="
Write-Host "  Total crops: $totalJpg"
Write-Host "  Transcribed: $totalMd"
Write-Host "  Still failed: $totalFailed"
if ($totalCleaned -gt 0) {
    Write-Host "  Stale failed_ocr.json cleaned: $totalCleaned"
}
