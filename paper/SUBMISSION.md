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

**Last Updated**: March 4, 2026  
**Paper Version**: Camera-ready (post-revision)  
**IEEE Access Submission ID**: TBD
