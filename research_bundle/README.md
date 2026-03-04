# Research Bundle - Clean Structure

**Last Updated**: March 4, 2026  
**Purpose**: Streamlined research artifacts for future work and testing

---

## Directory Structure

```
research_bundle/
├── future/           # Reference materials for future work (15 files)
│   ├── data/        # Key metric CSVs from paper
│   │   ├── calibration_bins_ece_correctness.csv
│   │   ├── confusion_matrix_and_per_class_metrics.csv
│   │   ├── optimization_ablation_logs.csv
│   │   └── risk_coverage_curve.csv
│   └── templates/   # IEEE LaTeX paper templates
│       ├── main.tex
│       ├── references.bib.template
│       └── sections-*.tex (9 modular sections)
│
├── published/        # Final published materials (11 files)
│   ├── OVERLEAF_TEMPLATE.tex    # Complete camera-ready LaTeX
│   ├── metrics_values.tex       # Auto-generated metric macros
│   ├── ABSTRACT_PLAINTEXT.txt
│   ├── KEYWORDS.txt
│   ├── SUBMISSION_METADATA.yaml
│   ├── figures/                 # Publication-ready PDFs
│   │   ├── accuracy_coverage_verified.pdf
│   │   ├── architecture.pdf
│   │   └── reliability_diagram_verified.pdf
│   └── tables/                  # LaTeX table fragments
│       ├── seed_policy.tex
│       ├── table_2_main_results.tex
│       └── table_3_multiseed.tex
│
└── tests/            # Testing and verification infrastructure (8 files)
    ├── compile/     # Compilation scripts
    │   ├── compile_and_verify.bat
    │   ├── compile_and_verify_final.bat
    │   ├── compile_camera_ready.bat
    │   └── Makefile
    └── verify/      # Verification scripts
        ├── verify.py
        ├── verify_submission.py
        ├── verify_pdf_text.py
        └── sanitize_unicode.py
```

---

## File Counts

| Directory | Files | Purpose |
|-----------|-------|---------|
| `future/` | 15 | Reference materials for future papers/iterations |
| `published/` | 11 | Final camera-ready submission artifacts |
| `tests/` | 8 | Verification and compilation infrastructure |
| **Total** | **34** | Clean, deduplicated structure |

---

## Usage

### For Future Research
```bash
# Reference key metrics
cd research_bundle/future/data/

# Start new paper from template
cp -r research_bundle/future/templates/ new_paper/
```

### For Testing/Verification
```bash
# Run verification suite
cd research_bundle/tests/verify/
python verify_submission.py

# Compile LaTeX
cd research_bundle/tests/compile/
make
```

### For Publication
```bash
# Upload to Overleaf
research_bundle/published/OVERLEAF_TEMPLATE.tex
research_bundle/published/figures/
research_bundle/published/tables/
```

---

## Changes from Original

**Removed**:
- 133 Markdown documentation files (outdated)
- Duplicate metric files (`metrics_values.tex`, `multiseed_values.tex`)
- All `.md.bak` backup artifacts
- Empty `submission_bundle/` (consolidated here)

**Kept**:
- Only files useful for future work or testing
- Publication-ready LaTeX + figures
- Key CSV datasets for reference
- Verification infrastructure

---

## Notes

- All markdown documentation removed to avoid version drift
- Single source of truth: `published/OVERLEAF_TEMPLATE.tex` (ready to compile)
- CSV data in `future/data/` preserved for reproducibility
- Test scripts remain functional for future verification runs
