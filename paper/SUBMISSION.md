# CalibraTeach IEEE Access Submission Guide

## Quick Overleaf Compile

### Compiler Settings
- **Compiler**: pdfLaTeX
- **Main document**: main.tex
- **TeX Live version**: 2023 or later

### Compilation Steps
1. Upload the complete `paper/` directory to Overleaf (or extract from `overleaf_submission.zip`)
2. Set main.tex as the main document
3. Select pdfLaTeX compiler
4. Click Recompile

**Expected output**: Single-column IEEE Access format with:
- Full paper content (~1430 lines)
- 3 figures (architecture, reliability diagram, accuracy-coverage)
- Multiple tables (main results, baselines, ablations)
- Embedded bibliography (no separate .bib file needed)

### File Dependencies

**Required files** (must be present for compilation):
- `main.tex` - Main paper source (entrypoint)
- `metrics_values.tex` - Auto-generated metric macros (accuracy, ECE, AUC-AC)
- `figures/architecture.pdf` - System architecture diagram
- `figures/reliability_diagram_verified.pdf` - Calibration reliability plot
- `figures/accuracy_coverage_verified.pdf` - Selective prediction trade-off

**Optional files**:
- `references.bib` - Currently empty (bibliography embedded in main.tex)

### Compilation Safeguards

**If `metrics_values.tex` is missing**: Paper includes fallback macros in main.tex:
```latex
\IfFileExists{metrics_values.tex}{
    \input{metrics_values.tex}
}{
    \newcommand{\AccuracyValue}{80.77\%}
    \newcommand{\ECEValue}{0.1076}
    \newcommand{\AUCACValue}{0.8711}
}
```

**If figures are missing**: Placeholders are rendered automatically (see main.tex \IfFileExists guards)

### Troubleshooting

**Compilation fails with "File not found"**:
- Check that all required files are uploaded
- Verify main.tex is set as the main document
- Confirm pdfLaTeX compiler is selected

**Unicode/encoding errors**:
- main.tex includes comprehensive Unicode sanitization (see lines 1-100)
- If warnings persist, ensure Overleaf is using UTF-8 encoding

**Figures don't appear**:
- Verify PDF figures are in `figures/` subdirectory
- Check console for "File not found" warnings
- Graphics paths are relative to main.tex location

### Build Artifacts

LaTeX compilation produces:
- `main.pdf` - Final compiled paper
- `main.aux`, `main.log`, `main.out` - Build artifacts (not needed for submission)

Only `main.pdf` is required for IEEE Access final submission.

---

## Paper Artifact Build & Manifest Contract

Before final submission or when metrics/figures change, rebuild all paper-facing artifacts deterministically:

### Quick Rebuild (30 seconds)

```bash
python scripts/rebuild_paper_artifacts.py
```

This regenerates:
1. **metrics_values.tex** - From `artifacts/metrics_summary.json` (seed=0)
2. **significance_values.tex** - From significance tests (seed=42) and CSV
3. **Verified figures** (PDFs) - Deterministic regeneration
4. **Manifest** - SHA256 hashes for all artifacts

**Expected output**:
```
[OK] rebuilt metrics_values.tex
[OK] rebuilt significance_values.tex
[OK] verified figures present/generated
[OK] wrote manifest: artifacts/manifest/paper_artifacts_manifest.json
[OK] audit passed
```

### Integration with Validation Pipeline

Rebuild artifacts as part of full validation:

```bash
python scripts/validate_paper_ready.py --rebuild-paper-artifacts --quick
```

This runs rebuild, then proceeds with:
- Paper consistency audit
- Metric verification
- Significance test validation
- Figure integrity checks

### Seed Policy for Determinism

All artifact regeneration follows strict seed policies to ensure determinism:

| Artifact | Seed | Purpose | Source |
|----------|------|---------|--------|
| metrics_values.tex | 0 | Extract from deterministic metrics_summary.json | Built during evaluation |
| significance_values.tex | 42 | Reproducible significance tests | run_significance_tests.py |
| Verified figures | Fixed | Deterministic plot generation | Matplotlib with fixed seed |

**Note**: Seeds are not re-tunable; they are fixed policy to match training/evaluation seeds.

### Manifest Contract

The build process writes `artifacts/manifest/paper_artifacts_manifest.json` with:

```json
{
  "manifest_version": "1.0",
  "generated_at_utc": "ISO 8601 timestamp",
  "git_commit": "Hash of current git commit",
  "python_version": "Python version used",
  "platform": "Windows/Linux/macOS version",
  "seed_policy": {
    "metrics_seed": 0,
    "significance_seed": 42
  },
  "artifacts": [
    {
      "path": "paper/metrics_values.tex",
      "sha256": "256-bit hex hash",
      "bytes": file_size,
      "description": "Brief description"
    }
  ]
}
```

**Purpose**: The manifest proves artifact immutability (no files changed without rebuild) and enables:
- Verification that artifacts match expected hashes
- Detection of accidental modifications
- Proof of deterministic regeneration
- Environment/platform audit trail

### Full Build + Validation + Submit Workflow

```bash
# 1. Rebuild all paper artifacts
python scripts/rebuild_paper_artifacts.py

# 2. Run comprehensive validation
python scripts/validate_paper_ready.py --rebuild-paper-artifacts --quick

# 2.5 Verify hygiene on COMPILED PDF bundle (if pdflatex is available)
python scripts/compile_and_check_pdf.py

# 3. Create Overleaf submission bundle
python scripts/build_overleaf_bundle.py

# 4. Verify bundle can compile
python scripts/build_overleaf_bundle.py --validate-only

# 5. Upload bundle to IEEE Access
# (submission system accepts ZIP or individual files)
```

Canonical architecture figure path used by the paper: `paper/figures/architecture.pdf` (included as `figures/architecture.pdf` from `paper/main.tex`).

---

**Last Updated**: March 4, 2026  
**Paper Version**: Camera-ready (post-revision)  
**IEEE Access Submission ID**: TBD
