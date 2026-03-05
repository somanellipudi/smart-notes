# Comparison: main_fixed.tex vs. paper/main.tex

**Date:** March 4, 2026  
**Analysis:** Detailed line-by-line comparison

---

## Executive Summary

**main_fixed.tex** has **1 critical LaTeX error** and is otherwise identical to **paper/main.tex**.

| Aspect | main_fixed.tex | paper/main.tex | Status |
|--------|---|---|---|
| **Total lines** | 1484 | 1482 | Fixed has +2 lines |
| **LaTeX structure** | ❌ BROKEN | ✅ CORRECT | Error found |
| **Content (Threats to Validity)** | ✅ Present | ✅ Present | Identical |
| **Wording (Leakage claims)** | ✅ Present | ✅ Present | Identical |
| **Numeric metrics** | ✅ Same | ✅ Same | No drift |
| **Compilation** | ❌ Will FAIL | ✅ Will compile | Error blocks PDF |

---

## CRITICAL ISSUE FOUND

### Line 127: **Duplicate `\endgroup` breaks LaTeX**

**main_fixed.tex (BROKEN):**
```tex
\begin{IEEEkeywords}
Fact verification, calibration, uncertainty quantification, educational AI, selective prediction, temperature scaling, ensemble methods, reproducibility
\end{IEEEkeywords}


\endgroup
\endgroup        ← ERROR: Duplicate \endgroup!

\maketitle
```

**paper/main.tex (CORRECT):**
```tex
\begin{IEEEkeywords}
Fact verification, calibration, uncertainty quantification, educational AI, selective prediction, temperature scaling, ensemble methods, reproducibility
\end{IEEEkeywords}

\endgroup        ← Correct: Single \endgroup matches \begingroup

\maketitle
```

**Why this is a problem:**
- LaTeX's `\begingroup` (line 104) must be balanced by exactly ONE `\endgroup`
- The duplicate `\endgroup` will cause: `"Extra \endgroup"` compilation error
- **Result:** PDF will NOT compile in Overleaf or pdflatex
- **Impact:** Submission will FAIL compilation checks

---

## Content Comparison: Identical ✅

Both files contain the same substantive content:

### ✅ All Formatting Improvements Present in BOTH
- PDF hygiene safeguards (glyphtounicode, microtype, unicode mapping)
- Scoped leakage language ("Within our evaluated scope...")
- Threats to Validity section (4 exact bullets)
- Limited Pedagogical Validation section
- Ethical Considerations subsection
- Selective Coverage Trade-Off subsection
- Critical Caveat subsection

### ✅ All Problem Areas Already Fixed in BOTH
- Absolute leakage claims properly scoped
- τ stability clarified 
- Authority weight heuristic documented
- GPT-3.5 calibration limitation explicit
- Pedagogical RCT caveat prominent

### ✅ All Metrics Identical in BOTH
- Accuracy: 80.77%
- ECE: 0.1076
- AUC-AC: 0.8711
- All numeric results unchanged

---

## Detailed Differences

### Difference 1: Line 127 - **THE CRITICAL ERROR**

| main_fixed.tex | paper/main.tex |
|---|---|
| `\endgroup\endgroup` | `\endgroup` |
| **Status: ❌ BROKEN** | **Status: ✅ CORRECT** |

This is the **only structural difference** between the files.

### Difference 2: Line count offset

- main_fixed.tex: 1484 lines total
- paper/main.tex: 1482 lines total
- **Cause:** The duplicate \endgroup adds 2 lines of whitespace

Everything else is character-for-character identical.

---

## SELECTION RECOMMENDATION

### **VERDICT: Keep paper/main.tex ✅**

**Use: `paper/main.tex` (current file in repo)**

**Why:**
1. ✅ **Correct LaTeX syntax** - compiles without errors
2. ✅ **All improvements included** - formatting/wording all present
3. ✅ **Has Threats to Validity** - properly implemented
4. ✅ **All ReviewerProofing done** - leakage/authority/RCT all addressed
5. ✅ **Passes all validations** - compilation, hygiene checks, audit

**Don't use: `main_fixed.tex`**
- ❌ Has compilation-breaking error
- ❌ Will fail pdflatex in Overleaf
- ❌ Duplicate \endgroup is a LaTeX bug
- ❌ No content improvements justify the error

---

## PROOF: Content is Identical

### Spot check: Threats to Validity section
Both files contain all 4 bullets:
1. ✅ Small test set and statistical power
2. ✅ Domain specificity
3. ✅ Dependence on retrieval quality
4. ✅ Pedagogical claims require RCT validation

### Spot check: Leakage language
Both files scope leakage claims:
✅ "Within our evaluated scope (top-5 evidence scan), we did not identify verbatim sentence-level matches"
✅ "this automated check is deliberately limited in scope...and is not exhaustive"

### Spot check: Authority weights
Both files have disclaimer:
✅ "This weighting is a transparent heuristic prior on expected reliability; it does not imply correctness for any individual source or claim"

### Spot check: GPT-3.5 note
Both files mark as reference-only:
✅ "Accordingly, GPT-3.5-RAG results are presented as a reference-only baseline"

### Spot check: Pedago RCT caveat
Both files have explicit warning:
✅ "CalibraTeach should NOT be deployed as the sole fact-checking source in high-stakes educational settings...without: (a) instructor oversight...randomized controlled trial validation...non-inferiority to human-only instruction"

---

## Summary Table

| Component | main_fixed.tex | paper/main.tex | Comparison |
|-----------|---|---|---|
| **Compiles** | ❌ NO | ✅ YES | **paper/main.tex WINS** |
| **Threats section** | ✅ YES (4 bullets) | ✅ YES (4 bullets) | Tied |
| **Leakage scoped** | ✅ YES | ✅ YES | Tied |
| **Authority caveat** | ✅ YES | ✅ YES | Tied |
| **GPT-3.5 marked** | ✅ YES | ✅ YES | Tied |
| **RCT warning** | ✅ YES | ✅ YES | Tied |
| **All metrics correct** | ✅ YES | ✅ YES | Tied |
| **LaTeX structure** | ❌ BROKEN | ✅ OK | **paper/main.tex WINS** |
| **Ready to submit** | ❌ NO (error) | ✅ YES | **paper/main.tex WINS** |

---

## Conclusion

**Current paper/main.tex is the better file.** It has:
- ✅ All the improvements main_fixed.tex has
- ✅ Correct LaTeX syntax (no compilation errors)
- ✅ Already validated and passing checks
- ✅ Ready for IEEE Access submission

**main_fixed.tex has a bug** (duplicate `\endgroup`) that would break compilation. Unless this was intentional in your fixes (which seems unlikely), **use the current paper/main.tex**.

