# Benchmark Progress Monitor
# Usage: .\monitor_benchmark.ps1

Write-Host "`n[CALIBRATEACH 20K BENCHMARK PROGRESS]`n" -ForegroundColor Cyan

# Find latest output directory
$outDir = Get-ChildItem 'evaluation/results' -Filter 'large_scale_20k_*' -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $outDir) {
    Write-Host "No benchmark output directory found`n" -ForegroundColor Red
    exit
}

# Count completed configs
$completed = (Get-ChildItem "$($outDir.FullName)/detailed_results" -Filter '*_metrics.csv' -ErrorAction SilentlyContinue | Measure-Object).Count
$percentage = [math]::Round(($completed / 8) * 100, 1)

# Status
Write-Host "Output Directory: $($outDir.Name)" -ForegroundColor Yellow
Write-Host "Last Updated: $($outDir.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))`n" -ForegroundColor Yellow

# Progress bar
Write-Host "Progress: $completed / 8 configs = $percentage%" -ForegroundColor Green

# List completed configs
if ($completed -gt 0) {
    Write-Host "`nCompleted:" -ForegroundColor Green
    Get-ChildItem "$($outDir.FullName)/detailed_results" -Filter '*_metrics.csv' -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name -replace '_metrics.csv', ''
        $data = Get-Content $_.FullName -Head 2 | Select-Object -Last 1
        Write-Host "  [OK] $name" -ForegroundColor Green
    }
}

# Check if running - look for venv Python or active jobs
$venvProcs = @(Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*\.venv\*' })
$jobs = @(Get-Job -State Running -ErrorAction SilentlyContinue)

Write-Host ""
if ($venvProcs -or $jobs) {
    Write-Host "Status: RUNNING" -ForegroundColor Green
    if ($venvProcs) {
        Write-Host "Processes: $($venvProcs.Count) venv Python"
        $totalCPU = ($venvProcs | Measure-Object -Property CPU -Sum).Sum
        Write-Host "CPU: $([math]::Round($totalCPU, 1))%" -ForegroundColor Yellow
    }
    if ($jobs) {
        Write-Host "Background Jobs: $($jobs.Count) active" -ForegroundColor Green
    }
} else {
    Write-Host "Status: STOPPED" -ForegroundColor Red
}

# Remaining configs
if ($completed -lt 8) {
    Write-Host "`nPending Configs:" -ForegroundColor Cyan
    $completedNames = Get-ChildItem "$($outDir.FullName)/detailed_results" -Filter '*_metrics.csv' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name -replace '_metrics.csv', '' }
    
    @("00_no_verification","01a_retrieval_only","01b_retrieval_nli","01c_ensemble","02_no_cleaning","03_with_artifact_persistence","04_no_batch_nli","05_with_online_authority") | 
    Where-Object { $_ -notin $completedNames } | 
    ForEach-Object { Write-Host "  [ ] $_" -ForegroundColor Gray }
}

Write-Host "`nLog: logs/benchmark_20k_*.log`n" -ForegroundColor Gray
