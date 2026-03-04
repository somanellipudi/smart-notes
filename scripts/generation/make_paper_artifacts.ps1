# Make Paper Artifacts Script for Windows
# 
# Generates all research paper artifacts and updates research_paper.md
#
# Usage:
#   .\scripts\make_paper_artifacts.ps1
#   .\scripts\make_paper_artifacts.ps1 -Quick

param(
    [switch]$Quick = $false
)

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  CalibraTeach Research Paper Artifact Generator" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Build command
$cmd = "python scripts/make_paper_artifacts.py"
if ($Quick) {
    $cmd += " --quick"
    Write-Host "Running in QUICK mode (reduced iterations for testing)" -ForegroundColor Yellow
} else {
    Write-Host "Running in FULL mode (2000 bootstrap samples, 5 seeds)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""

# Run the Python script
Invoke-Expression $cmd

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host "  ✓ SUCCESS: All artifacts generated and paper updated!" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Review artifacts in artifacts/latest/" -ForegroundColor White
    Write-Host "  2. Check research_paper.md for auto-generated sections" -ForegroundColor White
    Write-Host "  3. Run pytest to validate code" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "  ✗ ERROR: Artifact generation failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host ""
    exit $LASTEXITCODE
}
