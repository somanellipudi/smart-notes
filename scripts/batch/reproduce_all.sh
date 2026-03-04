#!/usr/bin/env bash
# Simple reproduce script for Linux/macOS.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VEV_DIR="$ROOT_DIR/.venv"

echo "Creating venv..."
python -m venv "$VEV_DIR"
source "$VEV_DIR/bin/activate"
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements-lock.txt

echo "Running unit tests..."
pytest -q

echo "Running evaluations (quick runs)..."
python src/evaluation/runner.py --mode baseline_retriever --out outputs/paper/baseline_ret
python src/evaluation/runner.py --mode baseline_nli --out outputs/paper/baseline_nli
python src/evaluation/runner.py --mode baseline_rag_nli --out outputs/paper/baseline_rag
python src/evaluation/runner.py --mode verifiable_full --out outputs/paper/verifiable_full

echo "Running ablations..."
python src/evaluation/ablation.py --output_base outputs/paper/ablations

echo "Updating experiment log..."
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_ret --label baseline_retriever
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_nli --label baseline_nli
python scripts/update_experiment_log.py --run_dir outputs/paper/baseline_rag --label baseline_rag
python scripts/update_experiment_log.py --run_dir outputs/paper/verifiable_full --label verifiable_full
python scripts/update_experiment_log.py --run_dir outputs/paper/ablations --label ablations

echo "Done. Results are under outputs/"
#!/usr/bin/env bash
set -euo pipefail

# Reproduce everything end-to-end (Linux / macOS)
# 1) create venv
# 2) install pinned requirements
# 3) run unit tests
# 4) run comprehensive evaluation
# 5) run ablations and plots

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_repro"

echo "ROOT_DIR=$ROOT_DIR"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements-lock.txt"

echo "Running unit tests..."
pytest -q

echo "Running comprehensive evaluation..."
python "$ROOT_DIR/run_comprehensive_evaluation.py"

echo "Done. Results in outputs/benchmark_results or evaluation/results depending on run." 
