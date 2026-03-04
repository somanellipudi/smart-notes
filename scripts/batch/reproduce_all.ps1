Param()
Write-Host "Creating virtual environment..."
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

Write-Host "Running unit tests..."
pytest -q

Write-Host "Running evaluations..."
python src/evaluation/runner.py --mode baseline_retriever --out outputs/paper/baseline_ret
python src/evaluation/runner.py --mode baseline_nli --out outputs/paper/baseline_nli
python src/evaluation/runner.py --mode baseline_rag_nli --out outputs/paper/baseline_rag
python src/evaluation/runner.py --mode verifiable_full --out outputs/paper/verifiable_full

Write-Host "Running ablations..."
python src/evaluation/ablation.py --output_base outputs/paper/ablations

Write-Host "Updating experiment log..."
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_ret --label baseline_retriever
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_nli --label baseline_nli
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_rag --label baseline_rag
python scripts/update_experiment_log.py --run_dir outputs/paper/verifiable_full --label verifiable_full
python scripts/update_experiment_log.py --run_dir outputs/paper/ablations --label ablations

Write-Host "Reproduction complete. Results: outputs/"
Param()

# Reproduce everything end-to-end (Windows PowerShell)
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv_repro"
if (-Not (Test-Path $Venv)) {
    python -m venv $Venv
}

$Activate = Join-Path $Venv "Scripts/Activate.ps1"
& $Activate
python -m pip install --upgrade pip
pip install -r (Join-Path $Root "requirements-lock.txt")

Write-Host "Running unit tests..."
pytest -q

Write-Host "Running comprehensive evaluation..."
python (Join-Path $Root "run_comprehensive_evaluation.py")

Write-Host "Reproducibility run complete. Check outputs/benchmark_results or evaluation/results."
