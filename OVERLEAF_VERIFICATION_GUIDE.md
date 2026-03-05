# How to Verify IEEE Access Red Flags are Fixed - Overleaf Guide

**After uploading dist/overleaf_submission.zip to Overleaf, follow these steps to verify:**

---

## VERIFICATION STEP 1: Figure 1 Visual Inspection

**What to check:** Figure 1 displays a clean 7-stage pipeline diagram

**How:**
1. In Overleaf, view the compiled PDF
2. Scroll to Figure 1 (should be on page with Introduction section)
3. **SHOULD SEE:** Clean flow diagram with only stage labels: Retrieval → Filtering → Ensemble → Weighting → Aggregation → Calibration → Prediction
4. **SHOULD NOT SEE:** 
   - Title banner: "CalibraTeach: 7-Stage Real-Time Fact Verification Pipeline"
   - GPU specs: "GPU: NVIDIA RTX 4090"
   - Framework versions: "PyTorch 2.0.1, CUDA 11.8, Transformers 4.35.0"
   - Any other embedded text

**Expected Result:** ✅ Clean architecture diagram with only stage labels and arrows

---

## VERIFICATION STEP 2: Copy-Paste Test from Abstract

**What to check:** Extracting abstract text produces no "￾" soft-hyphen artifacts

**How:**
1. In Overleaf PDF viewer, select and copy the entire Abstract section
2. Paste into a text editor (Notepad, VS Code, etc.)
3. Search for "￾" (soft-hyphen character) - should find ZERO matches
4. Search for suspicious characters like "Â­" or other Unicode artifacts - should find ZERO
5. Text should be clean: "...real-time feedback in classroom settings..."

**Common artifacts to look for:**
```
WRONG: "reproduc￾tion", "well￾calibrated", "confidently"
RIGHT: "reproduction", "well-calibrated", "confidently"
```

**Expected Result:** ✅ No "￾" characters in extracted abstract text

---

## VERIFICATION STEP 3: Copy-Paste Test from Keywords

**What to check:** Keywords section produces no "￾" artifacts

**How:**
1. In Overleaf PDF viewer, select and copy the Keywords section
2. Paste into text editor
3. Search for "￾" - should find ZERO
4. Search for "Â­" or other artifacts - should find ZERO
5. Keywords should be complete and readable: "fact verification, calibration, uncertainty quantification, ..."

**Expected Result:** ✅ No "￾" characters in keywords text

---

## VERIFICATION STEP 4: Figure 1 Caption Verification

**What to check:** Figure 1 caption contains no hardware or framework specs

**How:**
1. In Overleaf, look at the caption below Figure 1
2. You should see something like: "Figure 1: CalibraTeach 7-stage pipeline for educational fact verification."
3. **SHOULD NOT CONTAIN:** GPU specs, framework versions, or technical specifications
4. Caption should be purely descriptive of the pipeline stages

**Expected Result:** ✅ Caption is clean descriptive text only

---

## VERIFICATION STEP 5: Search PDF for Banned Strings

**Advanced: Manual search for embedded specs**

**How (if supported by Overleaf PDF viewer):**
1. Use Overleaf's search function (Ctrl+F or Cmd+F in PDF viewer)
2. Search for: "CalibraTeach X" (where X is a number)
3. Should find ZERO results for Figure 1 embedded text
4. Search for: "NVIDIA"
5. Should find ZERO results for GPU specs in figures
6. Search for: "PyTorch"
7. Should find ZERO results in figures

**Expected Result:** ✅ No embedded specs found via search

---

## VERIFICATION STEP 6: Full Compilation Status

**What to check:** Paper compiles without critical LaTeX errors

**How:**
1. In Overleaf, check the "Recompile" button status
2. Should show green checkmark (successful compilation)
3. Logs should show: `PDF generated successfully`
4. No critical errors about missing macros or files
5. Check page count: Should match expected (e.g., 12 pages, 13 pages, etc.)

**Expected Result:** ✅ PDF compiles successfully without critical errors

---

## TROUBLESHOOTING: What If Issues Persist?

### Issue: Figure 1 still shows embedded specs
**Solution:**
1. Download dist/overleaf_submission.zip again
2. Extract and check figures/architecture.pdf contains only stage labels
3. Verify local script: `python scripts/check_pdf_text_hygiene.py figures/architecture.pdf --check-architecture`
4. If local check fails, regenerate: `python scripts/regenerate_architecture_pdf.py`
5. Re-bundle: `python build_overleaf_bundle.py`
6. Re-upload to Overleaf

### Issue: Abstract still shows "￾" artifacts
**Solution:**
1. Check local compilation (if pdflatex available)
2. Run: `python scripts/compile_and_check_pdf.py`
3. Verify main.tex contains `\hyphenpenalty=10000` in scoped block
4. Verify main.tex contains `\usepackage[final]{microtype}`
5. If fixes missing, re-apply via local rebuild and re-upload

### Issue: LaTeX compilation error in Overleaf
**Solution:**
1. Check error message in Overleaf logs
2. Common fix: Microtype compatibility - usually not an issue on recent Overleaf
3. If microtype causes issues: Remove `\usepackage[final]{microtype}` (though unlikely)
4. Run locally: `python build_overleaf_bundle.py --validate-only`
5. Check output for error messages

---

## Expected PDF Characteristics After Fixes

| Characteristic | Before Fix | After Fix |
|---|---|---|
| Figure 1 embedded text | "CalibraTeach: 7-Stage...", "GPU:", specs | Only stage labels: "Retrieval → ... → Prediction" |
| Abstract extracted text | Contains "￾" artifacts | Clean text, no artifacts |
| Keywords extracted text | May contain "￾" | Clean text, no artifacts |
| LaTeX packages | cmap only | cmap + microtype [final] |
| Hyphenation scope | Global (affects all) | Scoped to abstract+keywords |
| PDF glyph mapping | Partial | Complete with cmap + glyphtounicode |

---

## Verification Checklist (Overleaf)

After uploading to Overleaf, mark these as complete:

- [ ] Figure 1 visually clean (no embedded text)
- [ ] Abstract copy-paste has no "￾" characters
- [ ] Keywords copy-paste has no "￾" characters
- [ ] Figure 1 caption is clean (no specs)
- [ ] PDF search finds no banned strings in figures
- [ ] LaTeX compilation successful (green checkmark)
- [ ] All figures render correctly
- [ ] Metrics values visible in tables
- [ ] References render correctly
- [ ] Page count as expected

**If all checkboxes pass:** ✅ Ready to submit!

---

## Contact / Questions

If you encounter issues during verification:

1. First, verify locally: `python scripts/validate_paper_ready.py --quick`
2. Check the validation report: `artifacts/TEST_STATUS_STEP2_8.md`
3. Review: `artifacts/IEEE_ACCESS_RED_FLAGS_RESOLUTION.md` for technical details
4. Regenerate bundle if needed: `python build_overleaf_bundle.py`

---

**Created:** March 4, 2026  
**Status:** Ready for Overleaf verification
