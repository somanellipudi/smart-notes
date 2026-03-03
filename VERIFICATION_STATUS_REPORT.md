# COMPREHENSIVE METRICS & FIGURES VERIFICATION REPORT
**Date:** March 2, 2026  
**Status:** ⚠️ CRITICAL ISSUES REQUIRE IMMEDIATE ACTION

---

## EXECUTIVE SUMMARY

The end-to-end metrics pipeline is **architecturally complete and deterministically reproducible**, but **BLOCKED on a data issue**: CalibraTeach predictions exported from a 14-sample batch instead of the expected 260-sample full test set. This cascades into all metrics mismatching the manuscript's reported values.

**Action Required:** Identify and provide the correct full 260-claim CalibraTeach predictions file.

---

## 1. METRICS VERIFICATION

### Current State vs. Expected Values

| Metric | Current | Expected | Status |
|--------|---------|----------|--------|
| **Sample Size** | 14 samples | 260 samples | ✗ **CRITICAL** (18.5x too small) |
| **Accuracy** | 57.14% | 80.77% | ✗ **CRITICAL** (−23.63pp) |
| **ECE** | 0.2521 | 0.1247 | ✗ **CRITICAL** (0.1274 delta) |
| **AUC-AC** | 0.7474 | 0.8803 | ✗ **CRITICAL** (−0.1329) |

### Root Cause
CalibraTeach.npz was computed from `outputs/benchmark_results/benchmark_20260218_085718_predictions.jsonl` which contains only **14 binary-labeled predictions**, not the full test set.

**Missing:** The correct 260-sample predictions file for CalibraTeach.

---

## 2. MANUSCRIPT MACRO STATUS

### LaTeX Macros (Auto-Generated)
```latex
\newcommand{\AccuracyValue}{57.14\%}
\newcommand{\ECEValue}{0.2521}
\newcommand{\AUCACValue}{0.7474}
```

**Issue:** Macros contain incorrect values propagated from the 14-sample data.

### Macro Usage in Manuscript
| Location | Macro | Status |
|----------|-------|--------|
| Abstract (line 44) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | ✓ Using macros |
| Table 1: Main Results (line 330-333) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | **FIXED** → Now using macros |
| Table 2: Baseline Comparison (line 377) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | ✓ Using macros |
| Failure Case Analysis (line 546) | \AccuracyValue{} | **FIXED** → Now using macro |
| Reliability Analysis (line 414, 423) | \ECEValue{} | ✓ Using macros |
| Ablation Study (line 443) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | ✓ Using macros |
| Authority Sensitivity (line 469) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | ✓ Using macros |
| Accuracy-Coverage Section (line 482, 491) | \AUCACValue{} | ✓ Using macros |
| Extended Ablation (line 866) | \AccuracyValue{}, \ECEValue{}, \AUCACValue{} | ✓ Using macros |

**Fixed in This Session:**
- ✅ Line 330: Table 1 Accuracy value now uses `\AccuracyValue{}`
- ✅ Line 546: Failure case description now uses `\AccuracyValue{}`

**Verification:** All 19 metric references now use auto-generated LaTeX macros. When correct data is provided and re-exported, ALL values will update automatically with a single command.

---

## 3. FIGURES VERIFICATION

### Files Generated
| Figure | File | Size | Status |
|--------|------|------|--------|
| Reliability Diagram | `figures/reliability_diagram_verified.pdf` | 15,642 bytes | ✓ EXISTS |
| Accuracy-Coverage Curve | `figures/accuracy_coverage_verified.pdf` | 12,801 bytes | ✓ EXISTS |

### Auto-Annotation Status
Both figures are **correctly wired to read metrics from `artifacts/metrics_summary.json`** (not hard-coded values):
- Reliability diagram displays: ECE = {value from JSON}
- Accuracy-coverage curve displays: AUC-AC = {value from JSON}

**Current Displayed Values:** ECE=0.2521, AUC-AC=0.7474 (from 14-sample data)

**When Fixed:** Figures will **automatically update** with correct values from 260-sample data.

---

## 4. RED FLAGS SUMMARY

### Critical Issues (Blocking Publication)

1. **✗ DATA MISMATCH:** CalibraTeach.npz contains 14 samples instead of 260
   - Source identified: `outputs/benchmark_results/benchmark_20260218_085718_predictions.jsonl` (14 binary rows)
   - Status: NOT THE CORRECT FULL TEST SET
   - **Required Action:** User must identify correct 260-sample predictions file

2. **✗ METRIC VALUE MISMATCH:** All primary metrics differ significantly
   - Accuracy: 57.14% vs expected 80.77% (−23.63pp)
   - ECE: 0.2521 vs expected 0.1247 (delta +0.1274)
   - AUC-AC: 0.7474 vs expected 0.8803 (delta −0.1329)
   - **Root Cause:** Computing from 14-sample batch instead of 260-sample full set

### Warnings Fixed
- ✅ ~~Line 330: Hard-coded '80.77\%' in table~~ → FIXED (now uses \AccuracyValue{})
- ✅ ~~Line 546: Hard-coded '80.77\%' in text~~ → FIXED (now uses \AccuracyValue{})

### Verification of Pipeline Components

| Component | Status | Notes |
|-----------|--------|-------|
| Metrics computation (src/eval/metrics.py) | ✓ Working | Authoritative module, deterministic |
| NPZ export (scripts/export_predictions_npz.py) | ⚠️ Wrong data | Exports from incorrect source file |
| Metric verification (scripts/verify_reported_metrics.py) | ✓ Working | Generates correct JSON/macros/reports |
| Figure generation (scripts/generate_paper_figures.py) | ✓ Working | Auto-annotates from JSON (not hard-coded) |
| Manuscript macros | ✓ Working | All references now use \AccuracyValue{} etc. |
| Tests suite (tests/test_metrics.py) | ✓ All Pass | 4/4 tests passing |
| Reproducibility (deterministic compare) | ✓ Validated | 2 runs identical within 1e-6 tolerance |

---

## 5. CONSISTENCY VERIFICATION

### Single Source of Truth
- ✓ Metrics computed once per model (artifacts/metrics_summary.json)
- ✓ Macros generated from JSON (submission_bundle/metrics_values.tex)
- ✓ Figures read metrics from JSON (not hard-coded in plotting code)
- ✓ Manuscript references all use LaTeX macros
- **Result:** All values will propagate consistently when corrected data is provided

### No Hard-Coded Values (Verification)
```
Before fix: 2 instances of hard-coded "80.77\%" found (lines 330, 546)
After fix:  ZERO instances of hard-coded metric numbers in metric tables/descriptions
```

### Macro Dependency Chain (Correct)
```
artifact/preds/CalibraTeach.npz
    ↓ (read by)
scripts/verify_reported_metrics.py
    ↓ (generates)
artifacts/metrics_summary.json (canonical golden source)
    ↓ (read by)
submission_bundle/metrics_values.tex (auto-generated macros)
    ↓ (included by)
submission_bundle/OVERLEAF_TEMPLATE.tex
    ↓ (all references now use)
\AccuracyValue{}, \ECEValue{}, \AUCACValue{} macros
```

---

## 6. NEXT STEPS TO RESOLUTION

### Immediate (User Action Required)
1. **Identify Correct Predictions File**
   - Current CalibraTeach source: 14 samples ← **WRONG**
   - Required: 260-sample test set predictions
   - Candidates found: `evaluation/results/research_logs/benchmark_20260302_130658_inference.jsonl` (13,334 rows, may need filtering)
   
   **Question for User:** Which file contains the correct 260 binary predictions for the test set?

### Upon User Confirmation
2. **Update Export Configuration**
   - Modify `scripts/export_predictions_npz.py` to use correct predictions file
   - Re-run: `python scripts/export_predictions_npz.py`
   - Expected output: `artifacts/preds/CalibraTeach.npz` with 260 samples

3. **Re-compute Metrics**
   - Run: `python scripts/verify_reported_metrics.py`
   - Expected output: Updated `artifacts/metrics_summary.json` and `submission_bundle/metrics_values.tex`
   - Verify accuracy ≈ 80.77%, ECE ≈ 0.1247, AUC-AC ≈ 0.8803

4. **Regenerate Figures**
   - Run: `python scripts/generate_paper_figures.py`
   - Figures automatically annotation with correct values from JSON

5. **Verify Determinism**
   - Run: `python scripts/verify_reported_metrics.py --compare artifacts/metrics_summary.json`
   - Confirm reproducibility (2 runs must be identical within 1e-6)

6. **Final Manuscript Compilation**
   - Recompile `submission_bundle/OVERLEAF_TEMPLATE.tex` in Overleaf
   - All 19 metric references now automatically show correct values
   - Verify: Abstract, tables, figures, captions all match

---

## 7. ARCHITECTURE QUALITY ASSESSMENT

### Design Strengths ✓
- **Single Source of Truth:** All metrics computed once, reused everywhere
- **No Hard-Coded Values:** All manuscript metrics pulled from macros
- **Deterministic Reproducibility:** 2 runs identical within floating-point tolerance
- **Auto-Propagation:** Metrics update everywhere with single command
- **Clean API:** Unified metrics.py with functions + compatibility class
- **Error Handling:** Tests validate metric definitions (ECE, AUC-AC, curves)

### Immediate Issue 
- **Data Sourcing:** CalibraTeach predictions from wrong (tiny) batch
- **Impact:** All cascade metrics wrong by 18-23pp

### Risk Mitigation
- ✅ Architecture is CORRECT for publication
- ✅ Once correct data provided, metrics will be correct
- ✅ Determinism ensures reproducibility for reviewers
- ✅ No future manual edits needed (macro-based)

---

## SUMMARY TABLE

| Aspect | Assessment | Details |
|--------|-----------|---------|
| **Metrics Computation** | ✓ Correct | Unified module, deterministic |
| **Data Export** | ✗ Wrong Input | 14 samples instead of 260 |
| **Metric Values** | ✗ Incorrect | ECE/AUC-AC wrong due to small sample |
| **Manuscript Integration** | ✓ Correct | All metrics now use macros |
| **Figures** | ✓ Correct | Auto-annotate from JSON |
| **Reproducibility** | ✓ Verified | Deterministic within 1e-6 |
| **Publication Readiness** | ⚠️ On Hold | Waiting for correct data file |

---

## RECOMMENDATION

**Status:** Pipeline is publication-ready *in architecture*, but **BLOCKED on data identification**.

**Action:** 
1. User identifies correct 260-sample CalibraTeach predictions file
2. Re-export and verify (5 minutes)
3. All metrics auto-propagate (zero manual edits needed)
4. Ready for submission

**Estimated Time to Fix:** 10 minutes after correct data identified
