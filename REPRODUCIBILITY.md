# Reproducibility Instructions

This document lists the exact commands used to reproduce the major-revision artifacts.

## 1) Environment setup

Run from project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
```

## 2) Calibration parity + plots + checksums

```powershell
python scripts/run_parity_all.py --results_glob evaluation/results/*_result.json --out_dir outputs/paper/calibration_parity_real --fig_dir figures/calibration_parity_real
```

Outputs:
- `outputs/paper/calibration_parity_real/` (calibrated predictions, metrics, temperature files, checksum manifest)
- `figures/calibration_parity_real/` (reliability and confidence plots)
- `outputs/paper/calibration_parity_real/ARTIFACT_CHECKSUMS.sha256`

## 3) CSClaimBench-Extended run

```powershell
python scripts/run_csclaimbench_extended.py --input evaluation/cs_benchmark/csclaimbench_extended.jsonl --normalized evaluation/cs_benchmark/csclaimbench_extended_normalized.jsonl --output-dir evaluation/results/csclaimbench_extended --seed 42
```

Notes:
- The runner normalizes extended JSONL into benchmark schema before evaluation.
- Use `--sample-size N` for faster smoke runs.

## 4) FEVER transfer run

```powershell
python scripts/run_fever_transfer.py --input evaluation/fever/fever_dev.jsonl --sample-size 300 --sampled-output evaluation/fever/fever_dev_sampled.jsonl --normalized evaluation/fever/fever_transfer_normalized.jsonl --output-dir evaluation/results/fever_transfer --seed 42
```

Notes:
- If FEVER input is missing, a synthetic placeholder file is generated automatically.
- Sampling is deterministic (`sample_jsonl_subset`) given `--seed`.

## 5) Optimization / minimal-deployment ablation

```powershell
python scripts/run_optimization_ablation.py --output-dir outputs/paper/optimization_ablation
```

Outputs:
- `outputs/paper/optimization_ablation/optimization_ablation_summary.csv`
- `outputs/paper/optimization_ablation/optimization_ablation_summary.md`
- `outputs/paper/optimization_ablation/optimization_ablation_summary.json`

## 6) Paper-ready calibration table generation

```powershell
python scripts/generate_paper_tables.py --parity-dir outputs/paper/calibration_parity_real --out-dir outputs/paper/tables
```

Outputs:
- `outputs/paper/tables/calibration_parity_table.csv`
- `outputs/paper/tables/calibration_parity_table.md`
- `outputs/paper/tables/calibration_parity_table.tex`

## Determinism notes

- Scripts use explicit seeds (`--seed`) and deterministic samplers where applicable.
- Calibration parity behavior is deterministic for fixed input result JSON files.
- Windows shell glob behavior is handled inside `scripts/calibration_parity.py`.
