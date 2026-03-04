#!/usr/bin/env pwsh
# Monitor chunked benchmark progress

Write-Host "`n[CHUNKED BENCHMARK PROGRESS - $(Get-Date -Format 'HH:mm:ss')]`n" -ForegroundColor Cyan

# Check job status
$jobs = Get-Job | Where-Object {$_.Name -match "^chunk_\d+$"} | Sort-Object Name
$running = ($jobs | Where-Object {$_.State -eq "Running"}).Count
$completed = ($jobs | Where-Object {$_.State -eq "Completed"}).Count
$failed = ($jobs | Where-Object {$_.State -eq "Failed"}).Count

Write-Host "Job Status: $running running, $completed completed, $failed failed" -ForegroundColor Yellow

# Check each chunk's results
for ($i = 1; $i -le 4; $i++) {
    $resultDir = "evaluation/results/chunk_$i"
    $detailedDir = "$resultDir/detailed_results"
    
    if (Test-Path $detailedDir) {
        $files = Get-ChildItem "$detailedDir" -Filter "*_metrics.csv" -File
        $configs = $files.Count
        Write-Host "  Chunk $i/4: $configs/8 configs completed" -ForegroundColor Green
    } else {
        Write-Host "  Chunk $i/4: 0/8 configs (initializing...)" -ForegroundColor Gray
    }
}

# PyTorch GPU install status
$installJob = Get-Job -Name "install_pytorch_gpu" -ErrorAction SilentlyContinue
if ($installJob) {
    Write-Host "`nGPU PyTorch Install: $($installJob.State)" -ForegroundColor Magenta
}

Write-Host ""
