# RESOLUTION STRATEGY — COMPREHENSIVE ACTION PLAN

**Status:** Critical metric inconsistencies detected  
**Priority:** BLOCKER — Must resolve before publication  

---

## 🎯 THE PROBLEM (SUMMARY)

The codebase has **two different computed values** for metrics:

| Metric | Current in Code | In Previous Verification | Difference | Issue |
|--------|-----------------|--------------------------|------------|-------|
| **ECE** | 0.1247 (hardcoded) | 0.1304 (computed from 260 real samples) | +0.0057 | Which is real? |
| **AUC-AC** | 0.8803 (hardcoded) | 0.9364 (computed from 260 real samples) | +0.0561 | Which is real? |
| **Accuracy** | 80.77% | 81.15% | +0.38pp | Consistent pattern |

**Key Evidence:**
- `EXECUTION_AND_INTEGRATION_STATUS.md` shows: ECE=0.1304, AUC-AC=0.9364 "recomputed values"
- `IMPLEMENTATION_STATUS.md` shows: ECE=0.1304, AUC-AC=0.9364 "FIXED values"
- Current `artifacts/metrics_summary.json` has: ECE=0.1247, AUC-AC=0.8803 (hardcoded)
- Current manuscript macros use: 0.1247 and 0.8803

**Root Cause:**
The previous work used `scripts/generate_expected_figures.py` which **hardcoded expected values** (0.1247, 0.8803) as a temporary baseline instead of reading from actual test split predictions.

---

## ⚠️ CRITICAL QUESTIONS (MUST ANSWER)

1. **What are the REAL metrics from the actual 260-sample CSClaimBench test set?**
   - Are they 0.1304/0.9364 (from previous verification run)?
   - Or something else entirely?
   - **Action:** Locate and inspect `artifacts/preds/CalibraTeach.npz` to verify sample count

2. **Where did 0.1247 and 0.8803 come from?**
   - Are they the reported paper values that we're trying to match?
   - Or are they synthetic/incorrect?
   - **Action:** Check `QUICK_ACTION_CHECKLIST.md` and README to understand the context

3. **Which values should be in the manuscript?**
   - If real computed values are 0.1304/0.9364, manuscript tables should show these
   - If the paper intended to report 0.1247/0.8803, we have a problem
   - **Action:** Clarify the intended values with paper authors

---

## 🔍 PHASE 1: INVESTIGATE TRUE METRICS (THIS IS CRITICAL)

### Step 1.1: Check the actual predictions file

```bash
# Navigate to artifacts and check CalibraTeach.npz
cd artifacts/preds
python -c "
import numpy as np
data = np.load('CalibraTeach.npz', allow_pickle=True)
y_true = data['y_true']
probs = data['probs']
print(f'Sample count: {len(y_true)}')
print(f'Prob range: [{probs.min():.4f}, {probs.max():.4f}]')
print(f'Y_true values: {np.unique(y_true)}')
"
```

**Expected output:**
- Sample count: Should be 260 (not 14!)
- Prob range: [0, 1]
- Y_true values: [0, 1] (binary)

### Step 1.2: Recompute metrics from real data

```bash
python scripts/verify_reported_metrics.py \
    --preds_dir artifacts/preds \
    --output_dir artifacts \
    --primary_model CalibraTeach
```

**What to check:**
- What ECE value is printed?
- What AUC-AC value is printed?
- Do they match 0.1304/0.9364 or something else?

### Step 1.3: Document the output

Create a report showing:
- Actual computed ECE from 260 samples
- Actual computed AUC-AC from 260 samples
- Compare to current macros (0.1247, 0.8803)
- Compare to previous "fixed" values (0.1304, 0.9364)

---

## 🔧 PHASE 2: UNIFY TO SINGLE SOURCE OF TRUTH (IF NEEDED)

### Option A: Current Values Are Correct (0.1247, 0.8803)

If `artifacts/preds/CalibraTeach.npz` is synthetic or wrong:

1. **Regenerate figures using `generate_expected_figures.py`** ✅ Already done
2. **Keep current macros** (0.1247, 0.8803)
3. **Verify manuscript consistency** - check all tables
4. **Status:** Ready for submission with expected/hardcoded values

### Option B: Recomputed Values Are Correct (0.1304, 0.9364)

If `artifacts/preds/CalibraTeach.npz` has real 260-sample data:

1. **Update all metrics:**
   ```bash
   # Regenerate metrics_summary.json with real data
   python scripts/verify_reported_metrics.py \
       --preds_dir artifacts/preds \
       --output_dir artifacts
   
   # This will... 
   # Actually what does it do? Let me check the code!
   ```

2. **Update LaTeX macros:**
   ```bash
   # Update submission_bundle/metrics_values.tex with real values
   # (verify_reported_metrics.py should do this)
   ```

3. **Regenerate figures with new metrics:**
   ```bash
   python scripts/generate_paper_figures.py \
       --metrics artifacts/metrics_summary.json \
       --output_dir submission_bundle/figures
   ```

4. **Update manuscript tables** (if different from current values)

5. **Status:** Publication ready with real computed values

### Option C: Mixed Data Problem

If artifacts contain BOTH old/new test sets:

1. Identify which `.npz` file is the official 260-sample test set
2. Delete all others: `rm artifacts/preds/*.npz` (keep only official)
3. Rerun verification
4. Proceed with Option A or B

---

## 📋 PHASE 3: VERIFICATION & VALIDATION (MUST PASS 2X)

### Step 3.1: First computation run

```bash
python scripts/verify_reported_metrics.py \
    --preds_dir artifacts/preds \
    --output_dir artifacts
```

**Expected output:**
- `artifacts/metrics_summary.json` generated
- `submission_bundle/metrics_values.tex` updated
- `artifacts/verification_report.md` created

### Step 3.2: Deterministic reproducibility test

```bash
python scripts/verify_reported_metrics.py \
    --preds_dir artifacts/preds \
    --output_dir artifacts \
    --compare artifacts/metrics_summary.json
```

**Must pass:** Both runs identical to 1e-6 tolerance

### Step 3.3: Figure regeneration

```bash
python scripts/generate_paper_figures.py \
    --pred_dir artifacts/preds \
    --metrics artifacts/metrics_summary.json \
    --out_dir submission_bundle/figures
```

**Verify:**
- `submission_bundle/figures/reliability_diagram_verified.pdf` created
- `submission_bundle/figures/accuracy_coverage_verified.pdf` created
- Plot annotations match `metrics_summary.json` values

### Step 3.4: Comprehensive verification

```bash
python comprehensive_verification.py
```

**Must pass:** All 50+ checks with 0 errors

---

## 🚨 IMMEDIATE ACTIONS (YOU SHOULD DO NOW)

### Action 1: Check what data we actually have

```powershell
# Terminal: Check if CalibraTeach.npz exists and what size it is
Get-ChildItem d:\dev\ai\projects\Smart-Notes\artifacts\preds\*.npz | ForEach-Object {
    Write-Host "File: $($_.Name), Size: $($_.Length) bytes"
}

# Python: Check sample count inside
python -c "
import numpy as np
from pathlib import Path

file = Path('artifacts/preds/CalibraTeach.npz')
if file.exists():
    data = np.load(file, allow_pickle=True)
    print(f'y_true shape: {data[\"y_true\"].shape}')
    print(f'probs shape: {data[\"probs\"].shape}')
else:
    print('File not found!')
"
```

### Action 2: Run verification to see what happens

```bash
python scripts/verify_reported_metrics.py \
    --verbose
```

**See what values are actually computed**

### Action 3: Compare expected vs computed

```bash
python -c "
import json
import sys
sys.path.insert(0, '.')

from src.eval.metrics import compute_ece, compute_accuracy_coverage_curve, compute_auc_ac
import numpy as np

# Load real data
try:
    data = np.load('artifacts/preds/CalibraTeach.npz', allow_pickle=True)
    y_true = data['y_true']
    probs = data['probs']
    
    # Compute real metrics
    ece_result = compute_ece(y_true, probs, n_bins=10)
    curve = compute_accuracy_coverage_curve(y_true, probs)
    auc_ac = compute_auc_ac(curve['coverage'], curve['accuracy'])
    
    print(f'REAL FROM DATA:')
    print(f'  ECE: {ece_result[\"ece\"]:.4f}')
    print(f'  AUC-AC: {auc_ac:.4f}')
    
    print(f'\\nCURRENT IN MACROS:')
    print(f'  ECE: 0.1247')
    print(f'  AUC-AC: 0.8803')
    
    print(f'\\nPREVIOUS \"FIXED\":')
    print(f'  ECE: 0.1304')
    print(f'  AUC-AC: 0.9364')
    
except Exception as e:
    print(f'Error: {e}')
"
```

---

## 🎯 SUCCESS CRITERIA

- [ ] Verify sample count in `artifacts/preds/CalibraTeach.npz` = 260
- [ ] Run `verify_reported_metrics.py` and capture actual computed values
- [ ] Confirm reproducibility (2nd run identical to first run)
- [ ] All metrics propagate to: JSON → LaTeX macros → manuscript tables → figures
- [ ] All figure annotations match computed values (not hard-coded)
- [ ] Comprehensive verification passes all 50+ checks
- [ ] Final verification report documents all values and consistency

---

## ⏭️ NEXT STEP

**RUN IMMEDIATELY:** Replace "generate_expected_figures.py" with real computation

The file `scripts/generate_expected_figures.py` should be REPLACED or REMOVED because it:
- Hardcodes ECE=0.1247
- Hardcodes AUC-AC=0.8803  
- Creates SYNTHETIC data to match these values
- Does NOT read from actual test split `artifacts/preds/CalibraTeach.npz`

**What should be used instead:**
- `scripts/verify_reported_metrics.py` - generates real metrics from real data
- `scripts/generate_paper_figures.py` - reads from `metrics_summary.json` (single source of truth)

---

**BLOCKING DECISION NEEDED:**
Do you want to:
1. **Keep current setup** (0.1247/0.8803 hardcoded, synthetic validation)
2. **Switch to real data** (0.1304/0.9364 computed, real validation)
3. **Investigate which is correct** (run diagnostics first)

Answer this and I'll proceed with the appropriate steps!
