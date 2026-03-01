# Reproducibility Instructions

This repository includes scripts and documentation to reproduce the evaluation outputs used for the manuscript drafts.

Quickstart (Linux/macOS):

```bash
bash scripts/reproduce_all.sh
```

Quickstart (Windows PowerShell):

```powershell
.\scripts\reproduce_all.ps1
```

Notes:
- `requirements-lock.txt` should ideally be produced with `pip freeze` from the environment used to produce the reported results. The provided file is a placeholder with candidate versions.
- Deterministic seeds are configured through `src/config/verification_config.py` (env var: `GLOBAL_RANDOM_SEED`).
- For PyTorch deterministic execution, set the following environment variables before running heavy GPU experiments:

```bash
export PYTHONHASHSEED=42
python -c "import torch; torch.use_deterministic_algorithms(True); import os; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'"
```

- All per-run metrics, figures, and metadata are written under `outputs/`. Use `scripts/update_experiment_log.py` to consolidate runs into `outputs/benchmark_results/experiment_log.json`.
# Reproducibility Guide

This document describes how to reproduce the evaluation and results for the Smart Notes project.

Prerequisites
- Python 3.10+ (recommended)
- Git
- ~8-16GB RAM depending on models used

Quickstart (Linux/macOS):

```bash
git clone <repo>
cd Smart-Notes
./scripts/reproduce_all.sh
```

Quickstart (Windows PowerShell):

```powershell
.\	ools\reproduce_all.ps1
```

Outputs
- `outputs/benchmark_results/latest/metrics.json` — consolidated metrics
- `outputs/benchmark_results/latest/figures/` — PNG/PDF figures
- `evaluation/results/` — detailed run outputs and ablation results

Determinism
- Seeds: `GLOBAL_RANDOM_SEED` and `VerificationConfig.random_seed` control randomness.
- Torch deterministic flags set where feasible inside evaluation runners.
- The scripts record git commit hash and package versions to `outputs/benchmark_results/latest/metadata.json`.
