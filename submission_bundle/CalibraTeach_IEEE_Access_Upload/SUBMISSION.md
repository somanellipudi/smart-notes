# CalibraTeach IEEE Access Submission Guide

**Paper**: Calibrated Selective Prediction for Real-Time Educational Fact Verification  
**Target**: IEEE Access  
**Last Updated**: March 4, 2026

---

## Pre-Submission Checklist

### 1. Unicode Sanitization (CRITICAL)
**Problem**: Invisible Unicode characters can cause PDF copy/paste artifacts ("￾" replacement glyphs).

**Solution**: Run the Unicode sanitizer before final submission:

```bash
# Check for problematic characters (exit code 1 if issues found)
python scripts/sanitize_unicode.py --check

# Auto-fix (creates .bak backups)
python scripts/sanitize_unicode.py --fix

# Verbose mode (shows all non-ASCII chars)
python scripts/sanitize_unicode.py --check --verbose
```

**What it detects**:
- Zero-width spaces (U+200B, U+200C, U+200D, U+2060)
- Invisible operators (U+2061..U+2064)
- Soft hyphens (U+00AD)
- BOM markers (U+FEFF)
- Noncharacters (U+FFFE, U+FFFF)
- Typographic quotes → LaTeX ligatures (`` '', --, ---)

**When to run**:
- Before uploading to Overleaf
- After any copy/paste from external sources (web, Word, Google Docs)
- Before final PDF generation for submission

---

### 2. PDF Text Extraction Verification (CRITICAL)

**Problem**: Bad Unicode characters in source → PDF copy/paste shows "￾" or "�" replacement glyphs → Reviewer frustration.

**Solution**: Build PDF and verify clean text extraction:

```bash
# Build PDF + verify text extraction (all-in-one)
python scripts/verify_pdf_text.py

# Verify existing PDF without rebuilding
python scripts/verify_pdf_text.py --pdf-only OVERLEAF_TEMPLATE.pdf

# Keep temporary LaTeX files (.aux, .log) for debugging
python scripts/verify_pdf_text.py --keep-temp
```

**What it checks**:
- Builds PDF using pdflatex (3 passes for cross-references)
- Extracts text using `pdftotext` (poppler-utils)
- Searches for bad glyphs: `￾`, `�`, `□` (empty box)
- Detects broken words from soft hyphens: "of￾ten", "in￾formation"
- Reports exact page estimate and context for each issue

**Dependencies**:
```bash
# Linux (Debian/Ubuntu)
sudo apt-get install texlive-full poppler-utils

# macOS
brew install mactex poppler

# Windows
# Install MiKTeX: https://miktex.org/download
# Install poppler: https://github.com/oschwartz10612/poppler-windows/releases
# Add poppler bin/ to PATH
```

**Expected output**:
```
✓ PASS: PDF text extraction is clean!
  No replacement characters (￾, �) or broken words found.

PDF ready for IEEE Access submission: OVERLEAF_TEMPLATE.pdf
```

---

### 3. Unified Verification Pipeline (ONE-COMMAND)

**Recommended**: Run all verification checks before submission:

```bash
# Run all checks (Unicode + Integrity + PDF build + Text extraction)
python verify.py

# Auto-fix Unicode issues, then run all checks
python verify.py --fix

# Skip PDF build (faster, for source-only checks)
python verify.py --skip-build
```

**Pipeline steps**:
1. **Unicode Sanitization**: Check .tex/.bib/.sty for invisible characters
2. **Submission Integrity**: Verify refs, metrics, document structure
3. **PDF Build**: Compile PDF using pdflatex (3 passes)
4. **Text Extraction**: Verify clean text (no ￾ � artifacts)

**Expected output**:
```
======================================================================
VERIFICATION SUMMARY
======================================================================
✓ ALL CHECKS PASSED (4/4)

✅ READY FOR IEEE ACCESS SUBMISSION

Next steps:
  1. Upload OVERLEAF_TEMPLATE.tex + figures/ to Overleaf
  2. Compile in Overleaf (verify no issues in cloud environment)
  3. Submit to IEEE Access ScholarOne
```

**If checks fail**: Fix reported issues, then re-run `python verify.py`.

---

### 4. Figure 1 (Architecture Diagram)

**Current Status**: `figures/architecture.pdf` has embedded header/footer text that may be flagged by reviewers.

**Current Workaround**: LaTeX uses `trim=0pt 40pt 0pt 30pt,clip` to crop embedded text.

**Recommended**: Regenerate figure from clean source without embedded text.

#### Option A: Regenerate from Vector Source (BEST)
If you have the original draw.io / PowerPoint / Inkscape source:
1. Open `figures/architecture.drawio` or `figures/architecture.pptx`
2. Remove any title bar text ("CalibraTeach System Architecture")
3. Remove hardware/software specification lines at bottom
4. Keep only: stage boxes, arrows, and short stage labels (1-7)
5. Export as PDF: `figures/architecture.pdf`

#### Option B: Manual Trim Adjustment
If regeneration is not feasible, adjust trim parameters in `OVERLEAF_TEMPLATE.tex` line 192:
```latex
\includegraphics[width=\textwidth,trim=0pt 40pt 0pt 30pt,clip]{figures/architecture.pdf}
```
Adjust values (in points) until embedded text is hidden:
- `trim=LEFT BOTTOM RIGHT TOP`
- Test with `pdflatex OVERLEAF_TEMPLATE.tex` and visually inspect Figure 1

#### Option C: Quick Fix with ImageMagick
```bash
cd figures/
convert -density 300 architecture.pdf -crop 0x-70+0+30 +repage architecture_clean.pdf
mv architecture_clean.pdf architecture.pdf
```

**Hardware/Software Specs**: Now in Figure 1 caption (keeps figure clean, specs in text).

---

### 5. Compilation Check

**IEEE Access requires**: pdflatex compatibility with IEEEtran class.

```bash
cd submission_bundle/CalibraTeach_IEEE_Access_Upload/
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex
bibtex OVERLEAF_TEMPLATE
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex
```

**Expected output**: `OVERLEAF_TEMPLATE.pdf` with:
- No "??" in references
- All figures/tables numbered sequentially
- PDF text extraction clean (no "￾" characters)

**Test PDF text extraction**:
```bash
pdftotext OVERLEAF_TEMPLATE.pdf - | grep -o "￾" | wc -l
# Should output: 0
```

---

### 6. Metric Preservation Verification

**CRITICAL**: No metric values should change from original manuscript.

Run the paper table verification script:
```bash
cd ../../  # Back to repo root
python scripts/verify_paper_tables.py
```

**Expected**: All 10 core metrics unchanged:
- Accuracy: 80.77%
- ECE: 0.1076
- AUC-AC: 0.8711
- Latency: 67.68ms
- Throughput: 14.78 claims/sec
- FEVER transfer: 74.3% accuracy, 0.150 ECE
- Operating point: τ=0.90 → 74.0% coverage, 90.2% selective accuracy
- Pilot: n=25, 92% instructor agreement

---

### 7. Final Style Checks

- [ ] All tables/figures referenced in order
- [ ] No "Table X" placeholders (search: `grep "Table X" OVERLEAF_TEMPLATE.tex`)
- [ ] All `\ref{}` labels resolve (no "??" in PDF)
- [ ] Bibliography entries consistent (conference names, capitalization)
- [ ] URLs use `\url{}` command (no raw http://)
- [ ] Section numbering matches all cross-references
- [ ] Abstract ≤ 200 words (IEEE Access limit)
- [ ] Keywords: 5-8 terms (currently 8: ✓)

---

### 8. Overleaf Upload Preparation

**Files to upload**:
```
submission_bundle/CalibraTeach_IEEE_Access_Upload/
├── OVERLEAF_TEMPLATE.tex      # Main manuscript
├── IEEEtran.cls               # IEEE Access class file
├── references.bib             # Bibliography (if separate)
├── figures/
│   ├── architecture.pdf       # Figure 1 (cleaned)
│   ├── reliability_diagram_verified.pdf
│   └── accuracy_coverage_verified.pdf
└── metrics_values.tex         # Auto-generated metrics (optional)
```

**Do NOT upload**:
- `*.bak` files (from sanitizer)
- `*.aux`, `*.log`, `*.bbl`, `*.blg` (LaTeX temp files)
- `scripts/` directory (local tooling only)

---

### 9. Known Issues / Future Improvements

**Low Priority**:
- Appendix numbering: If IEEE requests single-level appendix letters (A, B, C) instead of subsections (A.1, A.2), update `\subsection{...}` → `\section*{Appendix X: ...}` in appendix sections.

**Deferred to Camera-Ready**:
- Author ORCID IDs (if accepted)
- Copyright notice footer (IEEE provides template)
- Final DOI assignment

---

## Quick Reference: Pre-Submission Verification Commands

**ONE-COMMAND VERIFICATION (Recommended)**:
```bash
python verify.py              # Run all checks
python verify.py --fix        # Auto-fix Unicode, then verify all
```

**Individual Checks**:

| Task | Command | Exit Code |
|------|---------|-----------|
| Check Unicode | `python scripts/sanitize_unicode.py --check` | 0 = clean, 1 = issues |
| Fix Unicode | `python scripts/sanitize_unicode.py --fix` | Creates `.bak` backups |
| Verify submission | `python scripts/verify_submission.py OVERLEAF_TEMPLATE.tex` | 0 = pass, 1 = fail |
| Build + verify PDF | `python scripts/verify_pdf_text.py` | 0 = clean, 1 = artifacts |
| Verify existing PDF | `python scripts/verify_pdf_text.py --pdf-only my.pdf` | 0 = clean, 1 = artifacts |

**Manual Compilation** (if needed):
```bash
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex
bibtex OVERLEAF_TEMPLATE
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex
pdflatex -interaction=nonstopmode OVERLEAF_TEMPLATE.tex

# Test PDF text extraction manually
pdftotext OVERLEAF_TEMPLATE.pdf - | grep -o "￾" | wc -l  # Should output: 0
```

---

## How to Verify PDF Passes Extraction Checks

### Step-by-Step Process

**1. Run unified verification pipeline**:
```bash
cd submission_bundle/CalibraTeach_IEEE_Access_Upload/
python verify.py
```

**2. If Unicode issues found**:
```bash
python verify.py --fix  # Auto-fix + re-verify
```

**3. If PDF build fails**:
- Check LaTeX errors in terminal output
- Ensure all figure files exist in `figures/` directory
- Run `python scripts/verify_submission.py OVERLEAF_TEMPLATE.tex` for detailed diagnostics

**4. If PDF text extraction has artifacts**:
- Review reported bad glyphs (￾, �, etc.) with page/context
- Run `python scripts/sanitize_unicode.py --fix` to clean source
- Ensure preamble has:
  ```latex
  \IfFileExists{glyphtounicode.tex}{\input{glyphtounicode}}{}
  \pdfgentounicode=1
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
  ```
- Rebuild PDF and re-verify

**5. Manual PDF text extraction test**:
```bash
# Extract all text from PDF
pdftotext OVERLEAF_TEMPLATE.pdf extracted_text.txt

# Search for bad glyphs
grep "￾" extracted_text.txt  # Should find nothing
grep "�" extracted_text.txt   # Should find nothing

# Count bad glyphs (should output 0)
pdftotext OVERLEAF_TEMPLATE.pdf - | grep -o "￾" | wc -l
```

**6. Visual inspection**:
- Open PDF in Adobe Acrobat or browser
- Copy/paste a paragraph from Results section
- Paste into plain text editor (Notepad, TextEdit)
- Verify no "￾" or "�" characters appear
- Check that words like "often", "information", "reference" are intact (not "of￾ten")

### Expected Final State

✅ **All checks passed**:
```
======================================================================
VERIFICATION SUMMARY
======================================================================
✓ ALL CHECKS PASSED (4/4)

✅ READY FOR IEEE ACCESS SUBMISSION
```

✅ **PDF opens in Adobe Acrobat without warnings**  
✅ **Copy/paste from PDF → plain text shows clean, readable text**  
✅ **No "￾", "�", or "□" characters anywhere**  
✅ **All references numbered correctly (no "??")**  
✅ **All figures/tables display correctly**

---

## Quick Reference

| Task | Command | Exit Code |
|------|---------|-----------|
| Check Unicode | `python scripts/sanitize_unicode.py --check` | 0 = clean, 1 = issues |
| Fix Unicode | `python scripts/sanitize_unicode.py --fix` | Creates `.bak` backups |
| Compile PDF | `pdflatex OVERLEAF_TEMPLATE.tex` (3x) | 0 = success |
| Verify metrics | `python scripts/verify_paper_tables.py` | 0 = unchanged |

---

## Support

For questions about submission formatting or IEEE Access requirements:
- **IEEE Access Author Guidelines**: https://ieeeaccess.ieee.org/author-instructions/
- **Overleaf IEEE Access Template**: https://www.overleaf.com/latex/templates/ieee-access-latex-template/

For paper-specific questions:
- Contact: she4@kennesaw.edu (corresponding author)

---

**Last validation**: March 4, 2026  
**Status**: Ready for IEEE Access submission after Unicode sanitization + Figure 1 cleanup
