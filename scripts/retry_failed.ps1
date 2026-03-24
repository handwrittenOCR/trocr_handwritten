# Retry OCR on failed images (skips already transcribed .md files)
# Usage: .\scripts\retry_failed.ps1 -OutputDir <path> [-Timeout 120] [-MaxConcurrent 15] [-MaxRetries 3]

param(
    [string]$OutputDir = "C:\Users\marie\Dropbox\Personnelle\2. Travail\1. Recherche\3. JMP\3. OCR\2. TrOCR\5. Data (output)\ECES\Gemini3_transcribed\abymes\1842",
    [int]$Timeout = 120,
    [int]$MaxConcurrent = 10,
    [string]$Model = "gemini-3-flash-preview"
)

$VENV = ".venv\Scripts\python.exe"

$failedJson = Join-Path $OutputDir "failed_ocr.json"
if (-not (Test-Path $failedJson)) {
    Write-Host "No failed_ocr.json found in $OutputDir"
    exit 0
}

$failed = Get-Content $failedJson | ConvertFrom-Json

# Check which failed images still need processing
$needsRetry = @{}
foreach ($imgPath in $failed.images.PSObject.Properties) {
    $mdPath = $imgPath.Name -replace '\.jpg$', '.md'
    if (-not (Test-Path $mdPath)) {
        $needsRetry[$imgPath.Name] = $imgPath.Value
    }
}

if ($needsRetry.Count -eq 0) {
    Write-Host "All previously failed images now have transcriptions. Nothing to retry."
    Remove-Item $failedJson
    Write-Host "Removed failed_ocr.json"
    exit 0
}

# Update failed_ocr.json before retrying
$failed.failed_count = $needsRetry.Count
$updatedPre = @{
    timestamp = (Get-Date).ToString("o")
    model = $failed.model
    provider = $failed.provider
    failed_count = $needsRetry.Count
    images = $needsRetry
}
$updatedPre | ConvertTo-Json -Depth 3 | Set-Content $failedJson -Encoding UTF8

Write-Host "=== Retrying $($needsRetry.Count) failed images (of $($failed.images.PSObject.Properties.Count) original) ==="
Write-Host "Model:   $Model"
Write-Host "Timeout: ${Timeout}s"
Write-Host "Dir:     $OutputDir"
Write-Host ""

# The OCR script skips images that already have .md files
& $VENV -m trocr_handwritten.llm.ocr `
    --provider gemini `
    --model "$Model" `
    --input_dir "$OutputDir" `
    --pattern "*.jpg" `
    --max_concurrent $MaxConcurrent `
    --timeout $Timeout

Write-Host ""
Write-Host "=== Checking results ==="

# Re-read failed_ocr.json and check which images now have .md files
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

Write-Host "Resolved: $resolved / $($failedData.failed_count)"

if ($stillFailed.Count -eq 0) {
    Remove-Item $failedJson
    Write-Host "All images transcribed! Removed failed_ocr.json"
} else {
    # Update failed_ocr.json with only the remaining failures
    $updatedData = @{
        timestamp = (Get-Date).ToString("o")
        model = $failedData.model
        provider = $failedData.provider
        failed_count = $stillFailed.Count
        images = $stillFailed
    }
    $updatedData | ConvertTo-Json -Depth 3 | Set-Content $failedJson -Encoding UTF8
    Write-Host "Still $($stillFailed.Count) failed. Updated failed_ocr.json"
}
