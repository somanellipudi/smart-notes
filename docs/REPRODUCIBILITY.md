# CalibraTeach Reproducibility Guide

This guide provides exact commands to reproduce CalibraTeach results and verify paper artifacts.

---

## Quick Start (3 minutes)

Run these commands in order:

```bash
# 1. Run quickstart demo (smoke mode, CPU-only)
make quickstart

# 2. Verify artifacts
make verify-paper

# 3. Run tests
make test
```

**Expected outputs:**
- `artifacts/quickstart/output.json` - Demo results
- `artifacts/verification/VerificationReport.md` - Validation report
- Test suite passes

---

## Commands Reference

### `make quickstart`

Runs a lightweight demo on 5 CS claims in smoke mode (CPU-only, deterministic stubs).

**What it does:**
- Processes 5 predefined computer science claims
- Generates predictions with confidence scores
- Reports latency breakdown by pipeline stage
- Outputs structured JSON results

**Output:** `artifacts/quickstart/output.json`

**Options:**
```bash
# Custom options (bypass Makefile)
python scripts/quickstart_demo.py --help
python scripts/quickstart_demo.py --n 3 --out custom.json
python scripts/quickstart_demo.py --smoke --tau 0.85
```

---

### `make verify-paper`

Validates artifact structure and schema compliance.

**What it checks:**
1. Required directories exist (`artifacts/`, `artifacts/quickstart/`, etc.)
2. Quickstart output matches required JSON schema
3. If `artifacts/metrics_summary.json` exists, validates required keys (`accuracy`, `ece`, `auc_ac`)
4. Generates verification report

**Output:** `artifacts/verification/VerificationReport.md`

**Manual run:**
```bash
python scripts/verify_paper_artifacts.py --help
python scripts/verify_paper_artifacts.py --quickstart <path> --report <path>
```

---

### `make overleaf-bundle`

Builds a validated ZIP archive for Overleaf/IEEE submission.

**What it does:**
1. Validates all required paper assets (main.tex, figures, metrics)
2. Checks that all `\includegraphics` references point to existing files
3 Creates `dist/overleaf_submission.zip` with only compilation files
4. Excludes repo code, tests, artifacts

**Output:** `dist/overleaf_submission.zip`

**Manual run:**
```bash
python scripts/build_overleaf_bundle.py --help
python scripts/build_overleaf_bundle.py --validate-only
python scripts/build_overleaf_bundle.py --out custom.zip
```

**Windows:**
```powershell
python scripts/build_overleaf_bundle.py
```

---

### `make leakage-scan`

Automated lexical leakage analysis across all claims versus retrieved evidence passages.

**What it does:**
1. Scans all claims against retrieved top-k passages
2. Computes 3 lexical overlap metrics: LCO, LCS, SUBSTRING
3. Generates summary statistics and per-claim reports
4. Identifies claims with high lexical overlap (potential leakage)

**Important:** This measures *lexical overlap* (token reuse), not semantic leakage. High scores may indicate domain terminology reuse or legitimate phrasing overlap.

**Output:**
- `artifacts/leakage/leakage_report.json` - Complete results with per-claim metrics (includes `retrieval_mode` field for traceability)
- `artifacts/leakage/leakage_report.csv` - Tabular format for analysis
- `artifacts/leakage/README.md` - Detailed explanation and interpretation guide

#### Retrieval Modes

The leakage scanner supports three mutually exclusive retrieval modes, each serving different purposes:

| Mode | Claims Source | Passages Source | Paper-Usable | Use Case | CLI |
|------|---------------|-----------------|--------------|----------|-----|
| **smoke** | Synthetic (3 embedded) | Synthetic (6 embedded) | ❌ No | Testing, CI, CPU-only | `--smoke` |
| **mock** | Real (from file/auto-detect) | Deterministic dummy (6 fixed) | ❌ No | Unit tests, rapid development | `--retrieval_mode mock` |
| **real** | Real (from file/auto-detect) | Real retrieved passages | ✅ Yes | Paper-usable leakage analysis | `--retrieval_mode real` |

**Mode Details:**

- **smoke mode**: Deterministic, no external data required. Use for CI/testing.
- **mock mode**: Real claims loaded from file, but passages are fixed mock data (deterministic across runs). Useful for testing leakage metrics without retrieval infrastructure. Marked in JSON as `"retrieval_mode": "mock"`.
- **real mode**: Both claims and passages are real. Requires retrieval infrastructure (e.g., `src/eval/retrieval_module`). Fails with clear error if retrieval unavailable. This is the **paper-usable** mode.

**Retrieval index setup (required for real mode):**
```bash
python scripts/build_retrieval_index.py --corpus <path>
```

Real mode can also build index on the fly with `--corpus`.

**Manual run:**
```bash
# Smoke mode (no data files needed)
python scripts/leakage_scan.py --smoke --max_claims 3

# Mock mode (auto-detects claims file, uses dummy passages)
python scripts/leakage_scan.py --retrieval_mode mock --k 5 --k2 15

# Mock mode with explicit claims file
python scripts/leakage_scan.py --retrieval_mode mock --claims data/claims/my_claims.jsonl --k 5

# Real mode (paper-usable, requires retrieval infrastructure)
python scripts/leakage_scan.py --retrieval_mode real --claims data/CSClaimBench/claims.jsonl --k 5 --k2 15

# Real mode with explicit corpus build (recommended first run)
python scripts/leakage_scan.py --retrieval_mode real --claims data/CSClaimBench/claims.jsonl --corpus evaluation/cs_benchmark/csclaimbench_v1.jsonl --k 5 --k2 15

# Full pipeline (detaults to real mode + auto-detect claims)
python scripts/leakage_scan.py --k 5 --k2 15 --outdir artifacts/leakage
```

**Windows:**
```powershell
python scripts/leakage_scan.py --smoke --max_claims 3
python scripts/leakage_scan.py --retrieval_mode mock --k 5
python scripts/leakage_scan.py --retrieval_mode real --k 5 --k2 15 --outdir artifacts/leakage
```

#### Data Loading

When using **mock** or **real** modes (non-smoke), the scanner auto-detects claims files from known repo paths:
- `data/CSClaimBench/`
- `data/claims/`
- `artifacts/datasets/`
- `data/`

**Supported formats (auto-detected by extension):**
- **.jsonl**: Newline-delimited JSON with `{"text": "claim..."}` or `{"claim": "..."}` fields
- **.json**: JSON list `[...]` or object with `{"claims": [...]}` / `{"data": [...]}` / `{"items": [...]}`
- **.csv**: CSV with `claim`, `text`, or first column as claim field

**Explicit format override:**
```bash
python scripts/leakage_scan.py --retrieval_mode mock --claims data/claims.csv --format csv
```

**Fail-fast behavior:** If no claims file can be loaded in non-smoke mode, exits with code 2 and error message listing searched paths.

#### Options

- `--retrieval_mode {smoke|mock|real}`: Choose retrieval mode (default: real)
- `--claims <path>`: Explicit claims file path (optional, auto-detects if omitted)
- `--corpus <path>`: Corpus file to build retrieval index for real mode
- `--format {auto|jsonl|json|csv}`: Claims file format (default: auto)
- `--k <int>`: Minimum top-k passages (default: 5)
- `--k2 <int>`: Maximum top-k passages (default: 15)
- `--outdir <path>`: Output directory (default: artifacts/leakage)
- `--seed <int>`: Random seed for determinism (default: 0)
- `--max_claims <int>`: Limit claims scanned (optional)
- `--smoke`: Shorthand for `--retrieval_mode smoke`
- `--fail_on_threshold <float>`: Exit non-zero if max metric exceeds threshold

---

### `make test`


Runs the test suite using pytest.

**What it tests:**
- Quickstart demo runs successfully
- Output JSON parses and matches schema
- Verification script validates correctly
- All required fields present

**Manual run:**
```bash
pytest tests/ -v
pytest tests/test_quickstart_demo.py -v
pytest tests/test_verify_paper_artifacts.py -v
```

---

## File Locations

### Generated Artifacts

| File | Description | Created By |
|------|-------------|------------|
| `artifacts/quickstart/output.json` | Quickstart demo results | `make quickstart` |
| `artifacts/verification/VerificationReport.md` | Validation report | `make verify-paper` |
| `artifacts/leakage/leakage_report.json` | Leakage analysis results | `make leakage-scan` |
| `artifacts/leakage/leakage_report.csv` | Leakage analysis (CSV format) | `make leakage-scan` |
| `artifacts/leakage/README.md` | Leakage analysis guide | `make leakage-scan` |
| `dist/overleaf_submission.zip` | IEEE Access paper bundle | `make overleaf-bundle` |

### Source Files

| File | Description |
|------|-------------|
| `scripts/quickstart_demo.py` | Quickstart demo script |
| `scripts/verify_paper_artifacts.py` | Artifact verification script |
| `scripts/build_overleaf_bundle.py` | Overleaf bundle builder |
| `scripts/leakage_scan.py` | Automated leakage analysis scanner |
| `paper/` | Canonical paper directory (main.tex, figures, etc.) |
| `Makefile` | Command shortcuts |
| `tests/test_quickstart_demo.py` | Quickstart tests |
| `tests/test_verify_paper_artifacts.py` | Verification tests |
| `tests/test_overleaf_bundle.py` | Overleaf bundle tests |
| `tests/test_leakage_metrics.py` | Leakage metrics unit tests |
| `tests/test_leakage_scan_smoke.py` | Leakage scan integration tests |

---

## Output Schema

### Quickstart Output (`artifacts/quickstart/output.json`)

```json
{
  "run_id": "YYYYMMDD_HHMMSS",
  "smoke": true,
  "n": 5,
  "tau": 0.90,
  "examples": [
    {
      "claim": "string",
      "pred_label": "SUPPORTED|REFUTED|ABSTAIN",
      "confidence": 0.0,
      "abstained": false,
      "top_evidence": ["string", "string", "string"],
      "stage_latency_ms": {
        "retrieval": 0.0,
        "filtering": 0.0,
        "nli": 0.0,
        "aggregation": 0.0,
        "calibration": 0.0,
        "selective": 0.0,
        "explanation": 0.0,
        "total": 0.0
      }
    }
  ]
}
```

**Field definitions:**
- `run_id`: Timestamp of execution
- `smoke`: Whether smoke mode was used
- `n`: Number of claims processed
- `tau`: Abstention threshold (confidence cutoff)
- `examples`: List of per-claim results
  - `claim`: Input claim text
  - `pred_label`: Prediction (SUPPORTED, REFUTED, or ABSTAIN)
  - `confidence`: Calibrated confidence score [0, 1]
  - `abstained`: Whether system abstained (confidence < tau)
  - `top_evidence`: Top-3 evidence snippets
  - `stage_latency_ms`: Per-stage latency breakdown

### Leakage Report Output (`artifacts/leakage/leakage_report.json`)

```json
{
  "retrieval_mode": "mock|real",
  "claims_path": "string",
  "n_claims_scanned": 5,
  "k": 5,
  "k2": 15,
  "seed": 42,
  "timestamp": "YYYYMMDD_HHMMSS",
  "metrics": {
    "lco_min": 0.0,
    "lco_max": 0.95,
    "lco_mean": 0.35,
    "lcs_min": 0.0,
    "lcs_max": 0.88,
    "lcs_mean": 0.28,
    "substring_min": 0.0,
    "substring_max": 0.75,
    "substring_mean": 0.22
  },
  "per_claim": [
    {
      "claim_id": 0,
      "claim_text": "string",
      "passages_count": 10,
      "lco": 0.45,
      "lcs": 0.38,
      "substring": 0.25,
      "summary": "str"
    }
  ]
}
```

**Field definitions:**
- `retrieval_mode`: **IMPORTANT** - Indicates data source. Values:
  - `"smoke"`: Synthetic embedded data (testing only)
  - `"mock"`: Real claims + mock passages (unit tests only)
  - `"real"`: Real claims + real passages (paper-usable)
- `claims_path`: Path to claims file loaded (or "synthetic" for smoke mode)
- `n_claims_scanned`: Number of claims analyzed
- `k`, `k2`: Passage range parameters
- `seed`: Random seed used for determinism
- `timestamp`: Execution timestamp
- `metrics`: Aggregate statistics across all claims
  - `lco_*`: Lexical Containment Overlap metrics (min, max, mean)
  - `lcs_*`: Longest Common Subsequence metrics
  - `substring_*`: Substring overlap metrics
- `per_claim`: Array of per-claim results (metrics, passage stats)

For real mode runs, JSON also includes:
- `corpus_path`: Corpus path used by retrieval index
- `retrieval_manifest`: Index metadata (`num_docs`, `corpus_sha256`, etc.)

**Traceability Note:** The `retrieval_mode` field ensures transparency about data sources. Only reports with `"retrieval_mode": "real"` are suitable for paper results.

**Retrieval Method Note:** Leakage scan retrieval uses deterministic keyword-overlap ranking for reproducibility. It is intended for stable leakage auditing, not retrieval quality benchmarking.

---

## Smoke Mode vs Full Pipeline

### Smoke Mode (default for `make quickstart`)
- **Purpose**: CPU-only execution for CI/testing
- **Behavior**: Deterministic stub outputs (no model inference)
- **Speed**: ~1 second for 5 claims
- **Requirements**: None (Python 3.7+)
- **Use when**: Running in CI, on CPU-only machines, or verifying infrastructure

### Full Pipeline
- **Purpose**: Actual fact verification with models
- **Behavior**: Real NLI inference + evidence retrieval
- **Speed**: ~60-100ms per claim (GPU required)
- **Requirements**: GPU, model checkpoints, evidence corpus
- **Use when**: Reproducing paper results

**To run full pipeline** (when implemented):
```bash
python scripts/quickstart_demo.py --n 5  # omit --smoke flag
```

---

## Determinism

All outputs are deterministic when using `--smoke` mode:
- Fixed random seed: `42`
- Stable ordering of claims
- Deterministic pseudo-random generation based on claim hash

**Verification:**
```bash
# Run twice, outputs should be identical
make quickstart
cp artifacts/quickstart/output.json output1.json
make quickstart
diff output1.json artifacts/quickstart/output.json
# Should show no differences
```

---

## Troubleshooting

### `make: command not found`

**On Windows (PowerShell):**
```powershell
# Run scripts directly
python scripts/quickstart_demo.py --smoke
python scripts/verify_paper_artifacts.py
pytest tests/ -v
```

**On Windows (with WSL):**
```bash
wsl make quickstart
```

### Quickstart fails

1. Check Python version: `python --version` (requires 3.7+)
2. Check working directory: `pwd` (should be repo root)
3. Run with verbose output:
   ```bash
   python scripts/quickstart_demo.py --smoke --out artifacts/quickstart/output.json
   ```

### Verification fails

1. Run quickstart first: `make quickstart`
2. Check output exists: `ls artifacts/quickstart/output.json`
3. Validate JSON manually:
   ```bash
   python -m json.tool artifacts/quickstart/output.json
   ```

### Tests fail

1. Install pytest: `pip install pytest`
2. Run specific test:
   ```bash
   pytest tests/test_quickstart_demo.py::test_smoke_mode_output -v
   ```

---

## Legacy Reproducibility (Full Pipeline)

For reproducing full paper results with GPU models:

```bash
# Linux/macOS
bash scripts/reproduce_all.sh

# Windows PowerShell
.\scripts\reproduce_all.ps1
```

**Notes:**
- Requires GPU and model checkpoints
- Deterministic seeds configured via `src/config/verification_config.py`
- Set `GLOBAL_RANDOM_SEED` environment variable
- For PyTorch determinism:
  ```bash
  export PYTHONHASHSEED=42
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```

---

## Next Steps

After verifying Step 1 works:

1. **Extend to full pipeline**: Integrate actual verification models
2. **Add more benchmarks**: FEVER transfer, cross-domain evaluation
3. **Statistical tests**: Significance testing, ablation studies
4. **Full reproducibility**: 20-minute end-to-end reproduction protocol

See `artifacts/MANIFEST.md` for a complete list of artifacts.

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/somanellipudi/smart-notes/issues
- Email: she4@kennesaw.edu (Selena He, corresponding author)

---

**Last Updated**: March 4, 2026  
**Status**: Step 1 - Quickstart infrastructure complete
