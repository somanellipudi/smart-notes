# IEEE Access Submission Verification Infrastructure

**Created**: March 4, 2026  
**Purpose**: Ensure CalibraTeach manuscript passes IEEE Access technical checks (clean PDF text extraction, no Unicode artifacts, valid references)

---

## 📁 Files Created

### Core Verification Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **verify.py** | Unified pipeline (runs all checks) | `python verify.py` |
| **scripts/sanitize_unicode.py** | Detect/fix invisible Unicode chars | `python scripts/sanitize_unicode.py --check` |
| **scripts/verify_pdf_text.py** | Build PDF + verify text extraction | `python scripts/verify_pdf_text.py` |
| **scripts/verify_submission.py** | Check refs, metrics, structure | *(pre-existing, enhanced)* |

### Documentation

| File | Purpose |
|------|---------|
| **SUBMISSION.md** | Complete submission guide (updated with verification instructions) |
| **Makefile** | Make commands for verification (`make verify`, `make verify-fix`) |
| **VERIFICATION_INFRASTRUCTURE.md** | This file |

---

## 🚀 Quick Start

**ONE-COMMAND VERIFICATION** (Recommended before submission):

```bash
cd submission_bundle/CalibraTeach_IEEE_Access_Upload/
python verify.py
```

**Expected output**:
```
======================================================================
VERIFICATION SUMMARY
======================================================================
✓ ALL CHECKS PASSED (4/4)

✅ READY FOR IEEE ACCESS SUBMISSION
```

If checks fail, fix reported issues and re-run.

---

## 🔍 What Gets Verified

### 1. Unicode Sanitization (Source Files)
**Checks**: .tex, .bib, .sty files for invisible Unicode codepoints that cause PDF artifacts

**Problematic characters**:
- `U+00AD` (soft hyphen) → causes "of￾ten" in PDF
- `U+200B` (zero-width space)
- `U+200C` (zero-width non-joiner)
- `U+200D` (zero-width joiner)
- `U+2060..U+2064` (invisible operators)
- `U+FEFF` (BOM marker)
- `U+FFFE, U+FFFF` (noncharacters)

**Auto-fix**:
```bash
python scripts/sanitize_unicode.py --fix  # Creates .bak backups
```

---

### 2. Submission Integrity Checks
**Verifies**:
- ✓ All `\ref{}` labels have matching `\label{}`
- ✓ All 11 core metrics preserved (80.77%, 0.1076, 0.8711, 67.68ms, etc.)
- ✓ Document structure valid (15 tables, 3 figures balanced)
- ✓ Required files present (figures/, metrics_values.tex)

**Manual run**:
```bash
python scripts/verify_submission.py OVERLEAF_TEMPLATE.tex
```

---

### 3. PDF Build (pdfLaTeX Compilation)
**Process**:
1. Compiles PDF using `pdflatex` (3 passes for cross-references)
2. Checks for compilation errors
3. Verifies PDF generates successfully

**Dependencies**:
- pdflatex (TeX Live, MiKTeX, or Overleaf-compatible)

---

### 4. PDF Text Extraction Verification
**Checks**: Extracts text from PDF and searches for bad glyphs

**Bad glyphs detected**:
- `￾` (replacement character - most common artifact)
- `�` (REPLACEMENT CHARACTER U+FFFD)
- `□` (empty box - missing glyph)
- Broken words: "of￾ten", "in￾formation", "re￾fer"

**Manual run**:
```bash
python scripts/verify_pdf_text.py
```

**Verify existing PDF** (skip build):
```bash
python scripts/verify_pdf_text.py --pdf-only OVERLEAF_TEMPLATE.pdf
```

**Dependencies**:
- pdftotext (poppler-utils package)

---

## 📦 Installation (Dependencies)

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install texlive-full poppler-utils
```

### macOS
```bash
brew install mactex poppler
```

### Windows
1. **LaTeX**: Install [MiKTeX](https://miktex.org/download)
2. **pdftotext**: Download [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
   - Extract ZIP
   - Add `poppler-XX.XX.X/Library/bin/` to PATH

**Verify installation**:
```bash
pdflatex --version
pdftotext --version
```

---

## 🛠️ Usage Scenarios

### Scenario 1: Pre-Submission Check
**Goal**: Verify manuscript is ready for IEEE Access submission

```bash
python verify.py
```

If all checks pass → Upload to Overleaf and submit.

---

### Scenario 2: After Copy/Paste from Word/Google Docs
**Problem**: Copy/paste often introduces invisible Unicode characters

**Solution**:
```bash
python verify.py --fix  # Auto-fix Unicode + verify all
```

---

### Scenario 3: Debugging PDF Artifacts
**Problem**: PDF has "￾" characters when copy/pasting

**Diagnosis**:
```bash
# Step 1: Check source files for bad Unicode
python scripts/sanitize_unicode.py --check --verbose

# Step 2: Fix source
python scripts/sanitize_unicode.py --fix

# Step 3: Rebuild PDF and verify
python scripts/verify_pdf_text.py
```

---

### Scenario 4: Fast Source-Only Check (Skip PDF Build)
**Use case**: Quick check before committing code changes

```bash
python verify.py --skip-build
```

Runs Unicode + Integrity checks only (skips slow PDF build).

---

### Scenario 5: Using Make Commands
**Prerequisites**: GNU Make installed (Linux/Mac default, Windows: install via Chocolatey or use WSL)

```bash
make verify        # Run all checks
make verify-fix    # Auto-fix Unicode + verify all
make verify-source # Source checks only (fast)
make verify-pdf    # PDF build + text extraction only
make clean         # Remove temporary LaTeX files
```

**View all commands**:
```bash
make help
```

---

## 🔧 LaTeX Preamble Robustness

The manuscript preamble includes these safeguards:

```latex
% Guard: gracefully degrade if glyphtounicode.tex is missing
\IfFileExists{glyphtounicode.tex}{\input{glyphtounicode}}{}
\pdfgentounicode=1  % Enable Unicode mapping

% Font encoding (prevents PDF artifacts)
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
```

**Why this matters**:
- `glyphtounicode` guard: Prevents compilation failure if file missing in Overleaf
- `\pdfgentounicode=1`: Improves PDF text extraction
- `T1 fontenc + lmodern`: Ensures clean glyph rendering
- `DeclareUnicodeCharacter`: Strips 13 problematic codepoints at source

---

## 📊 Exit Codes

All verification scripts use standard exit codes:

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| **0** | All checks passed | ✅ Ready for submission |
| **1** | One or more checks failed | ❌ Fix reported issues and re-run |

**CI Integration** (for automated checks):
```bash
# In CI pipeline (GitHub Actions, GitLab CI, etc.)
python verify.py --skip-build || exit 1  # Fail CI if checks fail
```

---

## 🐛 Troubleshooting

### Issue: "pdflatex: command not found"
**Solution**: Install TeX Live (Linux/Mac) or MiKTeX (Windows)

---

### Issue: "pdftotext: command not found"
**Solution**: Install poppler-utils

```bash
# Linux
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
```

---

### Issue: PDF has "￾" artifacts in copy/paste
**Diagnosis**: Invisible Unicode in source files

**Solution**:
```bash
# Fix source files
python scripts/sanitize_unicode.py --fix

# Rebuild PDF
python scripts/verify_pdf_text.py

# Manually verify
pdftotext OVERLEAF_TEMPLATE.pdf - | grep "￾"  # Should find nothing
```

---

### Issue: Compilation fails with "File not found: glyphtounicode.tex"
**Diagnosis**: Old LaTeX preamble without `\IfFileExists` guard

**Solution**: The current preamble already has this guard:
```latex
\IfFileExists{glyphtounicode.tex}{\input{glyphtounicode}}{}
```

If you see this error, ensure you're using the latest OVERLEAF_TEMPLATE.tex.

---

### Issue: Reference shows "??" in PDF
**Diagnosis**: Missing `\label{}` or need more pdflatex passes

**Solution**:
```bash
# Run 3 passes (required for cross-references)
pdflatex OVERLEAF_TEMPLATE.tex
pdflatex OVERLEAF_TEMPLATE.tex
pdflatex OVERLEAF_TEMPLATE.tex

# Or use verification script (does this automatically)
python scripts/verify_pdf_text.py
```

---

## 📚 Script Documentation

### verify.py
**Purpose**: Unified verification pipeline (run all checks)

**Options**:
- `--fix`: Auto-fix Unicode issues before running checks
- `--skip-build`: Skip PDF build (faster, for source-only checks)

**Pipeline steps**:
1. Unicode Sanitization
2. Submission Integrity (refs, metrics, structure)
3. PDF Build (pdflatex x3)
4. PDF Text Extraction Verification

---

### scripts/sanitize_unicode.py
**Purpose**: Detect and fix problematic Unicode characters in source files

**Options**:
- `--check` (default): Report issues only (exit 1 if found)
- `--fix`: Auto-fix in-place (creates `.bak` backups)
- `--verbose`: Report all non-ASCII characters

**Examples**:
```bash
# Check all .tex/.bib/.sty files
python scripts/sanitize_unicode.py --check

# Fix all files (creates .bak backups)
python scripts/sanitize_unicode.py --fix

# Check specific file
python scripts/sanitize_unicode.py --check OVERLEAF_TEMPLATE.tex
```

---

### scripts/verify_pdf_text.py
**Purpose**: Build PDF and verify clean text extraction

**Options**:
- `--tex FILENAME`: Specify .tex file (default: OVERLEAF_TEMPLATE.tex)
- `--pdf-only FILENAME`: Verify existing PDF without building
- `--keep-temp`: Keep .aux/.log files (for debugging)

**Examples**:
```bash
# Build PDF + verify (default)
python scripts/verify_pdf_text.py

# Verify existing PDF
python scripts/verify_pdf_text.py --pdf-only OVERLEAF_TEMPLATE.pdf

# Build specific file
python scripts/verify_pdf_text.py --tex my_paper.tex
```

---

### scripts/verify_submission.py
**Purpose**: Check submission integrity (refs, metrics, structure)

**Examples**:
```bash
python scripts/verify_submission.py OVERLEAF_TEMPLATE.tex
```

---

## 🎯 Recommended Workflow

**Before final submission**:

1. **Fix Unicode issues**:
   ```bash
   python verify.py --fix
   ```

2. **Verify all checks pass**:
   ```bash
   python verify.py
   ```

3. **Visual inspection**:
   - Open PDF in Adobe Acrobat
   - Copy/paste a paragraph from Results section
   - Confirm no "￾" or "�" characters

4. **Upload to Overleaf**:
   - Upload `OVERLEAF_TEMPLATE.tex`
   - Upload `figures/` directory
   - Compile in Overleaf (verify cloud build works)

5. **Final check**:
   - Download PDF from Overleaf
   - Re-run: `python scripts/verify_pdf_text.py --pdf-only OVERLEAF_TEMPLATE.pdf`

6. **Submit to IEEE Access**

---

## 📞 Support

**For technical issues with verification scripts**:
- Check this documentation
- Review SUBMISSION.md for detailed instructions

**For IEEE Access submission questions**:
- IEEE Access Author Guidelines: https://ieeeaccess.ieee.org/author-instructions/
- Contact: she4@kennesaw.edu (corresponding author)

---

**Last Updated**: March 4, 2026  
**Status**: Production-ready verification infrastructure
