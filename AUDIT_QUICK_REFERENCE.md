# Paper Consistency Audit - Quick Reference Guide

**Status**: ✅ COMPLETE & TESTED  
**Last Updated**: March 4, 2026

---

## 🚀 Quick Start (30 seconds)

Run this before submitting to IEEE Access:

```bash
python scripts/audit_paper_consistency.py
```

**Expected Output**:
```
[OK] Macro verification
[OK] Significance verification
[OK] Figure presence check
[OK] Dataset size consistency
[OK] Overleaf bundle integrity
[OK] Unicode sanitation

Result: 6 passed, 0 failed
```

---

## 📋 What It Does

The audit checks that the paper and released artifacts stay synchronized:

| Check | What It Verifies | Files Involved |
|-------|------------------|-----------------|
| **Macro Verification** | Metrics in LaTeX match usage in paper | `metrics_values.tex` ↔ `main.tex` |
| **Significance Verification** | Test values match CSV exactly | `significance_values.tex` ↔ `significance_table.csv` |
| **Figure Presence** | All required PDFs exist and are readable | `figures/architecture.pdf`, `reliability_diagram_verified.pdf`, `accuracy_coverage_verified.pdf` |
| **Dataset Size** | Paper explains n=260 vs n=1000 difference | `main.tex` (text search) |
| **Bundle Integrity** | Overleaf can compile all files | `paper/` files structure |
| **Unicode Safety** | PDF copy-paste prevents replacement glyphs | Unicode declarations in `main.tex` |

---

## 🎯 Usage Modes

### Mode 1: Quick Check (30 seconds)
```bash
python scripts/audit_paper_consistency.py
```
- Fast pass/fail result
- Shows only [OK] or [ERROR] lines
- Exit code 0 = pass, 1 = fail

### Mode 2: Verbose Diagnostics (30 seconds)
```bash
python scripts/audit_paper_consistency.py --verbose
```
- Shows detailed diagnostic info for each check
- Prints macro values, file sizes, explanation snippets
- Useful for troubleshooting

### Mode 3: Integrated Validation (2-5 minutes)
```bash
python scripts/validate_paper_ready.py --quick
```
- Runs audit FIRST
- Then runs: paper tests, demo, leakage scan
- Comprehensive validation before submission

---

## ✅ Check Details

### 1. Macro Verification
**Checks**: `paper/metrics_values.tex` macros are used

| Macro | Value | Used In |
|-------|-------|---------|
| `\AccuracyValue` | 80.77% | main.tex |
| `\ECEValue` | 0.1076 | main.tex |
| `\AUCACValue` | 0.8711 | main.tex |

**Fix if failing**: Verify `paper/metrics_values.tex` contains these macros and `paper/main.tex` uses them

### 2. Significance Verification
**Checks**: `significance_values.tex` macros match CSV

Example macros:
- `\SigAccDiffRetrievalNLI` → CSV accuracy_diff
- `\SigPvalMcNemarBaseline` → CSV mcnemar_p

**Fix if failing**: Regenerate with `python scripts/generate_significance_tex.py`

### 3. Figure Presence
**Checks**: 3 required PDFs exist

```
paper/figures/
  ├── architecture.pdf (OK, 37.5 KB)
  ├── reliability_diagram_verified.pdf (OK, 16.2 KB)
  └── accuracy_coverage_verified.pdf (OK, 15.8 KB)
```

**Fix if failing**: Re-run figure generation scripts

### 4. Dataset Size Consistency
**Checks**: Paper mentions both n=260 and n=1000 WITH explanation

Required text pattern: "Significance tests use $n=1000$ ... whereas primary evaluation ... 260-claim"

**Fix if failing**: Add explanation to `paper/main.tex` (see REPRODUCIBILITY.md for exact text)

### 5. Overleaf Bundle Integrity
**Checks**: All files needed to compile exist

```
paper/
  ├── main.tex (REQUIRED - compilation root)
  ├── metrics_values.tex (REQUIRED - metric macros)
  ├── significance_values.tex (optional - test macros)
  ├── references.bib (REQUIRED - bibliography)
  └── figures/ (REQUIRED - all 3 PDFs)
```

**Fix if failing**: Ensure all files exist and `main.tex` safely includes others

### 6. Unicode Sanitation
**Checks**: PDF has minimal invisible character protections

```
\DeclareUnicodeCharacter{00AD}  % Soft hyphen (primary artifact source)
\DeclareUnicodeCharacter{200B}  % Zero-width space
\DeclareUnicodeCharacter{FEFF}  % BOM / zero-width no-break space
```

**Fix if failing**: Verify these lines exist in `paper/main.tex` preamble

---

## 🔧 Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `Missing macro definition: \AccuracyValue` | `metrics_values.tex` not found | `python scripts/verify_reported_metrics.py` |
| `Mismatch: significance_values.tex vs CSV` | CSV values changed but `.tex` not regenerated | `python scripts/generate_significance_tex.py` |
| `Missing: figures/accuracy_coverage_verified.pdf` | Figure not generated or wrong name | Re-run figure generation script |
| `No explanation found for n=260 vs n=1000` | Paper missing dataset size explanation | Add explanation to `main.tex` |
| `Overleaf bundle: missing required main.tex` | Paper directory misconfigured | Check `paper/main.tex` permissions |
| `Unicode sanitation check failed` | Missing Unicode declarations | Verify preamble has all 3 declarations |
| Exit code 1 | Any check failed | Read error messages and fix (tool explains what's wrong) |

---

## 📊 Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | All checks passed ✅ | Safe to submit |
| 1 | One or more checks failed ❌ | Fix issues and retry |

---

## 🔄 Workflow

### Before Submission
```bash
# 1. Run quick audit (30 sec)
python scripts/audit_paper_consistency.py

# 2. If all pass, run full validation (2-5 min)
python scripts/validate_paper_ready.py --quick

# 3. Build Overleaf bundle for submission
python scripts/build_overleaf_bundle.py

# 4. Submit to IEEE Access ✅
```

### If Audit Fails
```bash
# 1. Run verbose mode to see what failed
python scripts/audit_paper_consistency.py --verbose

# 2. Read error message and apply fix (see Troubleshooting table above)

# 3. Re-run audit to verify fix
python scripts/audit_paper_consistency.py

# 4. If still failing, check REPRODUCIBILITY.md for more details
```

---

## 📚 Additional Resources

- **Full Implementation Summary**: `artifacts/AUDIT_IMPLEMENTATION_SUMMARY.md`
- **Completion Checklist**: `artifacts/AUDIT_CHECKLIST.md`
- **Reproducibility Guide**: `docs/REPRODUCIBILITY.md` (search for "Paper Consistency Audit")

---

## ✨ Key Features

✅ **Automated**: No manual checking needed  
✅ **Fast**: Runs in ~1 second  
✅ **Clear**: Explains exactly what failed  
✅ **Integrated**: Part of validation pipeline  
✅ **Comprehensive**: 6 critical checks  
✅ **Safe**: Windows & Linux compatible  
✅ **Production Ready**: Ready for IEEE Access submission  

---

## 📞 Support

All 6 audit checks are passing. If you encounter issues:

1. Run `python scripts/audit_paper_consistency.py --verbose`
2. Check error message (tool explains the issue)
3. Refer to Troubleshooting table above
4. See `docs/REPRODUCIBILITY.md` for detailed guides

---

**Status**: ✅ Ready for IEEE Access Submission
