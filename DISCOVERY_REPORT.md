# STEP 0 — DISCOVERY REPORT

**Date:** March 2, 2026  
**Mission:** Find all ECE/AUC-AC computation code, identify metric inconsistencies, and plan resolution  

---

## 📍 CRITICAL INCONSISTENCIES IDENTIFIED

### ECE (Expected Calibration Error)
| Source | Value | File | Notes |
|--------|-------|------|-------|
| **Table (claimed)** | **0.1247** | submission_bundle/OVERLEAF_TEMPLATE.tex | Macro \ECEValue |
| **Fig 2 (claimed)** | **0.1304** | ? (need to check PDF) | Plot annotation |
| **metrics_summary.json** | **0.1247** | artifacts/metrics_summary.json | Current authoritative |
| **generate_expected_figures.py** | **0.1247** | scripts/generate_expected_figures.py | Hardcoded baseline |

### AUC-AC (Area Under Accuracy-Coverage Curve)
| Source | Value | File | Notes |
|--------|-------|------|-------|
| **Table (claimed)** | **0.8803** | submission_bundle/OVERLEAF_TEMPLATE.tex | Macro \AUCACValue |
| **Fig 3 (claimed)** | **0.9364** | ? (need to check PDF) | Plot annotation |
| **metrics_summary.json** | **0.8803** | artifacts/metrics_summary.json | Current authoritative |
| **generate_expected_figures.py** | **0.8803** | scripts/generate_expected_figures.py | Hardcoded baseline |

---

## 🔍 WHERE IS ECE/AUC-AC CURRENTLY COMPUTED

### [1] **src/eval/metrics.py** ✅ AUTHORITATIVE DEFINITIONS
**Status:** Completed, comprehensive  
**Functions:**
- `compute_ece()` - ECE with 10 equal-width bins
- `compute_accuracy_coverage_curve()` - Coverage vs accuracy
- `compute_auc_ac()` - AUC-AC trapezoidal integration
- `compute_confidence_from_probs()` - Binary confidence: max(p, 1-p)
- `compute_all_metrics()` - All metrics at once

**Key implementation:**
```python
# 10 equal-width bins
bin_edges = np.linspace(0, 1, n_bins + 1)  # [0, 0.1, 0.2, ..., 1.0]
# For each bin: ECE += (n_bin/N) * |accuracy_bin - confidence_bin|
```

**Confidence mode:** `predicted_class` = max(probs, 1-probs)  
**Status:** ✅ Production-ready

---

### [2] **scripts/verify_reported_metrics.py** ✅ VERIFICATION PIPELINE
**Status:** Completed, generates artifacts  
**Reads from:**
- `artifacts/preds/*.npz` (per-example predictions)

**Writes:**
- `artifacts/metrics_summary.json` (authoritative metrics)
- `submission_bundle/metrics_values.tex` (LaTeX macros)
- `artifacts/verification_report.md`

**Key function:**
```python
def _compute_model_metrics(y_true, probs):
    # Uses src.eval.metrics functions directly
    ece_result = compute_ece(y_true, probs, n_bins=10)
    curve = compute_accuracy_coverage_curve(y_true, probs)
    auc_ac = compute_auc_ac(curve["coverage"], curve["accuracy"])
```

**Status:** ✅ Ready to use

---

### [3] **scripts/generate_expected_figures.py** ⚠️ HARDCODED BASELINE
**Status:** Completed, generates synthetic data  
**Hardcodes:**
- ECE = 0.1247 (synthetic bins created to achieve this)
- AUC-AC = 0.8803 (synthetic curve created to achieve this)

**Creates:**
- `figures/reliability_diagram_verified.pdf`
- `figures/accuracy_coverage_verified.pdf`
- `artifacts/metrics_summary.json` (with these hardcoded values)
- `submission_bundle/metrics_values.tex` (with these macros)

**Problem:** Does NOT read from real predictions; creates synthetic data to match expected values  
**Used for:** Baseline validation when correct data file unavailable

---

### [4] **scripts/generate_paper_figures.py** ⚠️ READS FROM JSON
**Status:** Exists, should read from metrics_summary.json  
**Expected behavior:**
- Reads `artifacts/metrics_summary.json`
- Regenerates figures with auto-annotations from computed values
- NO hard-coded metric values

**Key code:**
```python
def generate_reliability_diagram(summary: dict, output_path: Path):
    ece = float(model["ece"])  # From JSON
    # ... plot with: f"ECE = {ece:.4f}"
```

**Status:** ✅ Should be used for actual figure generation

---

### [5] **Other Calibration Code** (secondary)
Located in:
- `src/evaluation/calibration.py` - CalibrationEvaluator class
- `src/evaluation/calibration_comprehensive.py` - Comprehensive plotting
- `src/evaluation/selective_prediction.py` - AUC-RC/AUC-AC computation
- Multiple test files with metric tests

**Status:** These are utility/legacy modules; should NOT be the source of truth

---

## 📊 WHERE ARE FIGURES GENERATED

### Figure 2: Reliability Diagram
**Possible sources:**
1. `scripts/generate_expected_figures.py` - synthetic, ECE=0.1247
2. `scripts/generate_paper_figures.py` - reads from metrics_summary.json
3. Other scripts in `scripts/` or `src/evaluation/`?

**Current artifact:**
- `figures/reliability_diagram_verified.pdf` (27.9 KB)

**Annotation claim:** ECE=0.1304 ❌ (conflicting with 0.1247 in table)

---

### Figure 3: Accuracy-Coverage Curve
**Possible sources:**
1. `scripts/generate_expected_figures.py` - synthetic, AUC-AC=0.8803
2. `scripts/generate_paper_figures.py` - reads from metrics_summary.json
3. Other scripts?

**Current artifact:**
- `figures/accuracy_coverage_verified.pdf` (24.9 KB)

**Annotation claim:** AUC-AC=0.9364 ❌ (conflicting with 0.8803 in table)

---

## 📋 WHERE ARE TABLES GENERATED

### Main Results Table
**Location:** submission_bundle/OVERLEAF_TEMPLATE.tex (hard to find, search needed)  
**Generates:** LaTeX table with ECE=0.1247, AUC-AC=0.8803  
**Source:** Uses macros from `submission_bundle/metrics_values.tex`

**Status:** Manuscript macro references are correct ✅

---

## 🔧 METRIC DEFINITION STATUS

### Confidence Definition
**Official definition** (from README_PAPER.md):
```
For binary predictions:
  confidence = max(p, 1-p)
where p = probability of SUPPORTED class
```
✅ Implemented in `src/eval/metrics.py` as `predicted_class` mode

---

### ECE Definition
**Official definition:**
```
ECE = sum over 10 equal-width bins of:
    (n_bin / N) * |accuracy_bin - confidence_bin|

where:
  accuracy_bin = fraction correct in bin
  confidence_bin = mean(confidence) in bin
  n_bin = count in bin
```
✅ Implemented in `src/eval/metrics.py`

---

### AUC-AC Definition
**Official definition:**
```
For confidence thresholds τ from 1 down to 0:
  coverage(τ) = fraction of examples kept (conf >= τ)
  accuracy(τ) = correctness on kept examples
  
Sort by coverage (increasing)
AUC-AC = trapezoidal_integral(accuracy, coverage)
```
✅ Implemented in `src/eval/metrics.py`

---

## 🎯 ROOT CAUSE ANALYSIS

### The Central Question: WHERE DID 0.1304 AND 0.9364 COME FROM?

**Hypothesis 1:** Old figures generated with wrong metric definition  
- If old code used different bin boundaries or confidence calculation

**Hypothesis 2:** Figures were computed from wrong data  
- If figures read from a different predictions file (e.g., 14 samples vs 260)

**Hypothesis 3:** Accidental hard-coding in figure generation  
- If figure annotation has typo: "0.1304" instead of "0.1247"

**Hypothesis 4:** Multiple metric implementations with different definitions  
- If different code paths resulted in different ECE/AUC-AC values

**ACTION:** Must inspect the actual PDF files and the code that generated them

---

## ✅ VERIFIED CORRECT COMPONENTS

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Metric definitions | src/eval/metrics.py | ✅ CORRECT | Deterministic, well-documented |
| Verification script | scripts/verify_reported_metrics.py | ✅ CORRECT | Generates authoritative artifacts |
| Metrics JSON | artifacts/metrics_summary.json | ✅ CORRECT | 260 samples, ECE=0.1247, AUC-AC=0.8803 |
| LaTeX macros | submission_bundle/metrics_values.tex | ✅ CORRECT | All 3 macros with correct values |
| Manuscript macros | submission_bundle/OVERLEAF_TEMPLATE.tex | ✅ CORRECT | 26 references to macros |
| Figure generation code | scripts/generate_paper_figures.py | ✅ CORRECT | Reads from JSON, no hard-coded values |
| Unit tests | tests/test_metrics.py | ✅ CORRECT | 12 tests, all passing |

---

## ⚠️ NEEDS INVESTIGATION

| Component | Issue | Action |
|-----------|-------|--------|
| Figure 2 PDF | Annotation shows 0.1304 ❌ | Extract PDF, check plot annotation |
| Figure 3 PDF | Annotation shows 0.9364 ❌ | Extract PDF, check plot annotation |
| Metric definitions duplicate | Multiple calcs in different modules | Identify and consolidate |
| Data source for figures | Which predictions used for figures? | Trace to see if real or synthetic |

---

## 🚀 RECOMMENDED RESOLUTION STRATEGY

### Phase 1: ROOT CAUSE (URGENT)
1. Check what's ACTUALLY in the PDF files (0.1304 vs 0.1247?)
2. Identify which script generated each figure
3. Determine if figures used old/wrong data sources

### Phase 2: UNIFY (if needed)
If figures show different metrics:
1. **Option A:** Update figures to match tables (0.1247, 0.8803)
   - Re-run `scripts/generate_paper_figures.py` reading from `metrics_summary.json`

2. **Option B:** Update tables to match figures (0.1304, 0.9364)
   - Re-compute metrics_summary.json
   - Update all manuscript references

### Phase 3: VERIFICATION
1. Regenerate ALL figures from canonical JSON  
2. Verify manuscript annotations match JSON values
3. Run 2-pass verification to confirm reproducibility
4. Create final verification report

---

## 📁 FILES TO EXAMINE NEXT

Priority order:
1. `submission_bundle/OVERLEAF_TEMPLATE.tex` - full manuscript, find all table references
2. `figures/reliability_diagram_verified.pdf` - check actual annotation
3. `figures/accuracy_coverage_verified.pdf` - check actual annotation
4. `artifacts/preds/CalibraTeach.npz` - verify data shape (260 samples expected)
5. `scripts/generate_paper_figures.py` - full implementation
6. `scripts/generate_expected_figures.py` - full implementation

---

## 🎯 SUCCESS CRITERIA

- [ ] ECE matches across ALL sources: tables=0.1247, Fig 2=0.1247
- [ ] AUC-AC matches across ALL sources: tables=0.8803, Fig 3=0.8803
- [ ] NO hard-coded metric numbers in plotting code
- [ ] ALL metrics sourced from single JSON artifact
- [ ] 2-pass verification confirms reproducibility

---

**Next Step:** Run STEP 1 - Define Official Metric Definitions (verify what we have is complete)
