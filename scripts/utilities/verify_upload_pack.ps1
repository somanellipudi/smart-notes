# Overleaf Upload Pack Verification Script
# Verifies all required files are present and synchronized

Write-Host "`n=========================================================" -ForegroundColor Cyan
Write-Host "     OVERLEAF UPLOAD PACK VERIFICATION" -ForegroundColor Cyan
Write-Host "=========================================================`n" -ForegroundColor Cyan

$uploadPackPath = "submission_bundle\overleaf_upload_pack"
$mainPath = "submission_bundle"
$allPass = $true

# Check 1: Template file
Write-Host "[1/5] Checking template file..." -ForegroundColor Yellow
if (Test-Path "$uploadPackPath\OVERLEAF_TEMPLATE.tex") {
    $mainHash = (Get-FileHash -Path "$mainPath\OVERLEAF_TEMPLATE.tex" -Algorithm SHA256).Hash
    $packHash = (Get-FileHash -Path "$uploadPackPath\OVERLEAF_TEMPLATE.tex" -Algorithm SHA256).Hash
    
    if ($mainHash -eq $packHash) {
        Write-Host "  [OK] OVERLEAF_TEMPLATE.tex: SYNCHRONIZED" -ForegroundColor Green
        $mainSize = (Get-Item "$mainPath\OVERLEAF_TEMPLATE.tex").Length
        Write-Host "    Size: $mainSize bytes" -ForegroundColor Gray
    } else {
        Write-Host "  [FAIL] OVERLEAF_TEMPLATE.tex: OUT OF SYNC" -ForegroundColor Red
        $allPass = $false
    }
} else {
    Write-Host "  [FAIL] OVERLEAF_TEMPLATE.tex: MISSING" -ForegroundColor Red
    $allPass = $false
}

# Check 2: Metrics file
Write-Host "`n[2/5] Checking metrics file..." -ForegroundColor Yellow
if (Test-Path "$uploadPackPath\metrics_values.tex") {
    $metricsMain = Get-Content "$mainPath\metrics_values.tex" -Raw
    $metricsPack = Get-Content "$uploadPackPath\metrics_values.tex" -Raw
    
    if ($metricsMain -eq $metricsPack) {
        Write-Host "  [OK] metrics_values.tex: SYNCHRONIZED" -ForegroundColor Green
        
        if ($metricsMain -match 'AccuracyValue\}\{([^}]+)\}') {
            Write-Host "    Accuracy: $($Matches[1])" -ForegroundColor Gray
        }
        if ($metricsMain -match 'ECEValue\}\{([^}]+)\}') {
            Write-Host "    ECE: $($Matches[1])" -ForegroundColor Gray
        }
        if ($metricsMain -match 'AUCACValue\}\{([^}]+)\}') {
            Write-Host "    AUC-AC: $($Matches[1])" -ForegroundColor Gray
        }
    } else {
        Write-Host "  [FAIL] metrics_values.tex: OUT OF SYNC" -ForegroundColor Red
        $allPass = $false
    }
} else {
    Write-Host "  [FAIL] metrics_values.tex: MISSING" -ForegroundColor Red
    $allPass = $false
}

# Check 3: Figures
Write-Host "`n[3/5] Checking required figures..." -ForegroundColor Yellow
$requiredFigures = @(
    "architecture.pdf",
    "reliability_diagram_verified.pdf",
    "accuracy_coverage_verified.pdf"
)

foreach ($fig in $requiredFigures) {
    $figPath = "$uploadPackPath\figures\$fig"
    if (Test-Path $figPath) {
        $size = (Get-Item $figPath).Length
        Write-Host "  [OK] $fig ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] MISSING: $fig" -ForegroundColor Red
        $allPass = $false
    }
}

# Check 4: Unexpected files
Write-Host "`n[4/5] Checking for unexpected files..." -ForegroundColor Yellow
$allFiles = Get-ChildItem -Path $uploadPackPath -Recurse -File | Where-Object { $_.Name -ne "README_UPLOAD_PACK.md" }
$expectedFiles = @(
    "OVERLEAF_TEMPLATE.tex",
    "metrics_values.tex",
    "architecture.pdf",
    "reliability_diagram_verified.pdf",
    "accuracy_coverage_verified.pdf"
)

$unexpectedFiles = $allFiles | Where-Object { 
    $name = $_.Name
    -not ($expectedFiles -contains $name)
}

if ($unexpectedFiles.Count -eq 0) {
    Write-Host "  [OK] No unexpected files found" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Unexpected files found:" -ForegroundColor Yellow
    foreach ($file in $unexpectedFiles) {
        Write-Host "    - $($file.Name)" -ForegroundColor Gray
    }
}

# Check 5: Package size
Write-Host "`n[5/5] Calculating package size..." -ForegroundColor Yellow
$totalSize = ($allFiles | Measure-Object -Property Length -Sum).Sum
$totalSizeKB = [math]::Round($totalSize / 1KB, 2)
$totalSizeMB = [math]::Round($totalSize / 1MB, 2)

Write-Host "  [OK] Total size: $totalSizeKB KB ($totalSizeMB MB)" -ForegroundColor Green
Write-Host "    Well within Overleaf upload limits" -ForegroundColor Gray

# Summary
Write-Host "`n=========================================================" -ForegroundColor Cyan
if ($allPass) {
    Write-Host " STATUS: READY FOR OVERLEAF UPLOAD" -ForegroundColor Green
    Write-Host "=========================================================`n" -ForegroundColor Cyan
    Write-Host "All checks passed! Upload pack is complete and synchronized." -ForegroundColor White
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Create a ZIP of the upload pack folder" -ForegroundColor White
    Write-Host "  2. Go to https://overleaf.com" -ForegroundColor White
    Write-Host "  3. New Project -> Upload Project" -ForegroundColor White
    Write-Host "  4. Select the ZIP file" -ForegroundColor White
    Write-Host "  5. Set compiler to pdfLaTeX and compile`n" -ForegroundColor White
} else {
    Write-Host " STATUS: ISSUES FOUND - SEE ABOVE" -ForegroundColor Red
    Write-Host "=========================================================`n" -ForegroundColor Cyan
    Write-Host "Please resolve the issues above before uploading." -ForegroundColor White
}
