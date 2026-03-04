Write-Host "=== Metric Consistency Check ==="
$eceMentions = (Select-String -Path submission_bundle/OVERLEAF_TEMPLATE.tex -Pattern '0.1247').Count
$aucMentions = (Select-String -Path submission_bundle/OVERLEAF_TEMPLATE.tex -Pattern '0.8803').Count
Write-Host "Paper mentions of 0.1247 (ECE): $eceMentions"
Write-Host "Paper mentions of 0.8803 (AUC-AC): $aucMentions"

Write-Host "`n=== Artifact Existence Check ==="
$files = @(
  'artifacts/metrics_summary.json',
  'artifacts/metrics_summary.md',
  'artifacts/METRIC_RECONCILIATION_REPORT.md',
  'figures/reliability_diagram_verified.pdf',
  'figures/accuracy_coverage_verified.pdf',
  'scripts/verify_reported_metrics.py',
  'tests/test_metrics.py'
)
foreach ($file in $files) {
  if (Test-Path $file) {
    $item = Get-Item $file
    Write-Host ("✅ {0} ({1} bytes)" -f $file, $item.Length)
  } else {
    Write-Host "❌ $file MISSING"
  }
}

Write-Host "`n=== Unit Tests ==="
python -m pytest tests/test_metrics.py -q

Write-Host "`n=== Reproducibility Check ==="
$outDir = "artifacts/latest"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
python scripts/verify_reported_metrics.py *> "$outDir/run1.txt"
python scripts/verify_reported_metrics.py *> "$outDir/run2.txt"
$same = (Get-FileHash "$outDir/run1.txt").Hash -eq (Get-FileHash "$outDir/run2.txt").Hash
if ($same) {
  Write-Host "✅ Metrics reproducible (runs identical)"
} else {
  Write-Host "❌ Metrics NOT reproducible (runs differ)"
}

Write-Host "`n=== PDF Compilation ==="
Push-Location submission_bundle
pdflatex -interaction=nonstopmode -halt-on-error OVERLEAF_TEMPLATE.tex *> latex.log
if ((Test-Path OVERLEAF_TEMPLATE.pdf) -and ($LASTEXITCODE -eq 0)) {
  Write-Host "✅ PDF generated successfully"
  Get-Item OVERLEAF_TEMPLATE.pdf | Select-Object Name, Length, LastWriteTime
} else {
  Write-Host "❌ PDF generation failed"
  Write-Host "See submission_bundle/latex.log for details"
}
Pop-Location
