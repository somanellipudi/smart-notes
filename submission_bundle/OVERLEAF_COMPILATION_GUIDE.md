# Overleaf Template Compilation Guide
## CalibraTeach IEEE Access Manuscript

**Template File**: `OVERLEAF_TEMPLATE.tex`  
**Status**: ✅ Complete and ready to compile  
**Last Updated**: March 2, 2026

---

## 🚀 Quick Start: Upload to Overleaf

### Step 1: Create Overleaf Project

1. Go to https://overleaf.com
2. Click **"New Project"** → **"Upload Project"**
3. Upload `OVERLEAF_TEMPLATE.tex`

**OR** manually create a new project:

1. Click **"New Project"** → **"Blank Project"**
2. Name it: `CalibraTeach_IEEE_Access`
3. Delete the default `main.tex` content
4. Copy entire contents of `OVERLEAF_TEMPLATE.tex`
5. Paste into the Overleaf editor

### Step 2: Set Compiler

1. Click the **Menu** icon (☰) in top-left
2. Under **"Settings"**:
   - **Compiler**: Select **"pdfLaTeX"**
   - **TeX Live version**: **2023** or newer
3. Close menu

### Step 3: Compile

1. Click **"Recompile"** button (or Ctrl+S / Cmd+S)
2. Wait 10-20 seconds for compilation
3. PDF preview appears on right side

**Expected Output**: 15-20 page IEEE-formatted document with:
- Title & authors
- Abstract (245 words)
- Main sections (Intro, Related Work, Methods, Results, Limitations, Conclusion)
- References (20 citations)
- Appendices (Dataset details, Hyperparameters, Ablation, Re-calibration protocol)

---

## ✅ What's Included in the Template

### Complete Content (No Placeholders)

- ✅ **Abstract**: Full 245-word abstract with pedagogical caveat
- ✅ **Introduction**: Motivation, contributions (5 bullet points)
- ✅ **Related Work**: 4 subsections (Fact Verification, Calibration, Selective Prediction, Educational AI)
- ✅ **Technical Approach**: 
  - 7-stage pipeline description
  - 6-component ensemble with full equations
  - Temperature scaling formulation
  - Selective prediction framework
- ✅ **Experimental Setup**:
  - CSClaimBench dataset details (1,045 claims)
  - 7 baselines with calibration parity
  - Evaluation metrics (Accuracy, ECE, AUC-AC, CIs)
- ✅ **Results**: 
  - 6 tables with latest metrics (80.77%, 0.1247 ECE, 0.8803 AUC-AC)
  - Multi-seed stability results
  - Baseline comparison
  - Ablation study
  - Latency breakdown
  - Transfer learning (FEVER)
  - Infrastructure validation
- ✅ **Limitations**: 8 detailed subsections including critical pedagogical RCT caveat
- ✅ **Conclusion**: Emphasis on calibration value & future work
- ✅ **References**: 20 complete citations in IEEE format
- ✅ **Appendices**: 4 appendices (Dataset, Hyperparameters, Extended Ablation, Re-calibration)

### Professional Formatting

- IEEE Access journal template (`\documentclass[journal]{IEEEtran}`)
- Proper section hierarchy
- Professional tables using `booktabs` package
- Mathematical equations numbered and formatted
- Hyperlinked references (blue color)
- Balanced columns on last page

---

## 📊 Key Metrics in Template (Verified Latest)

All metrics updated to **March 2, 2026 full-pipeline run**:

| Metric | Value | CI | Multi-Seed |
|--------|-------|-----|------------|
| **Accuracy** | 80.77% | [75.38%, 85.77%] | 0.8169 ± 0.0071 |
| **Macro-F1** | 0.7998 | [0.7536, 0.8480] | - |
| **ECE** | 0.1247 | [0.0989, 0.1679] | 0.1317 ± 0.0088 |
| **AUC-AC** | 0.8803 | [0.8207, 0.9386] | 0.8872 ± 0.0219 |
| **Latency** | 67.68 ms | (mean) | - |
| **Throughput** | 14.78 claims/sec | - | - |

---

## 🛠️ Customization Options

### Add Figures (Optional)

If you have generated figures, add to Overleaf:

1. Upload figure files (PDF format recommended):
   - `fig_architecture.pdf`
   - `fig_reliability_diagram.pdf`
   - `fig_latency_breakdown.pdf`
   - etc.

2. Insert in LaTeX after relevant paragraphs:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\columnwidth]{fig_architecture.pdf}
\caption{CalibraTeach 7-stage pipeline architecture.}
\label{fig:architecture}
\end{figure}
```

3. Reference in text: `see Figure~\ref{fig:architecture}`

### Adjust Formatting

**Page length**: Currently ~15-20 pages. IEEE Access has no strict page limit, but typical range is 8-30 pages.

**Font size**: Standard 10pt (IEEE journal format)

**Spacing**: Single-column format for review; IEEE handles final two-column layout

### Add Author Photos (Optional)

For final publication, IEEE may request author photos:

```latex
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{author1.jpg}}]{Nidhhi Behen Patel}
Biography text here.
\end{IEEEbiography}
```

---

## 🐛 Troubleshooting Common Issues

### Issue: Compilation Timeout

**Cause**: Large document or slow Overleaf server

**Solution**:
1. Premium Overleaf account (faster compile servers)
2. Remove appendices temporarily and compile in stages
3. Use local LaTeX installation (see below)

### Issue: Missing Package Errors

**Error**: `! LaTeX Error: File 'booktabs.sty' not found`

**Solution**: Overleaf includes all packages by default. If error occurs:
1. Check TeX Live version: Menu → Settings → TeX Live version = 2023
2. Clear cache: Menu → "Logs and output files" → "Clear cached files"
3. Recompile

### Issue: Equation Rendering Problems

**Common cause**: Unmatched braces `{ }` or missing `$` delimiters

**Check**:
- All equations wrapped in `\begin{equation}...\end{equation}` or `$ ... $`
- Matching braces in `\frac{}{}`, `\text{}`, etc.
- Use `\textit{}` for italics inside equations

### Issue: Table Formatting Breaks

**Symptom**: Table overflows page margins

**Solutions**:
- Use `\small` or `\footnotesize` before table
- Reduce column widths
- Split into multiple tables
- Rotate with `\begin{sidewaystable}` (requires `rotating` package)

---

## 💻 Local Compilation (Alternative to Overleaf)

### Requirements

- **TeX Distribution**:
  - Windows: MiKTeX or TeX Live
  - Mac: MacTeX
  - Linux: TeX Live

### Compile Command

```bash
pdflatex OVERLEAF_TEMPLATE.tex
pdflatex OVERLEAF_TEMPLATE.tex  # Run twice for references
```

**Why twice?** First run processes content; second run resolves cross-references and citations.

### Recommended Editor

- **TeXstudio** (Windows/Mac/Linux): https://www.texstudio.org/
- **Texmaker**: https://www.xm1math.net/texmaker/
- **VS Code** with LaTeX Workshop extension

---

## 📄 Export PDF for Submission

### From Overleaf

1. Click **"Menu"** (☰)
2. Click **"Download"** → **"PDF"**
3. Save as `CalibraTeach_IEEE_Access.pdf`

### Verify PDF Quality

**Checklist**:
- ✅ All text is searchable (not images)
- ✅ All equations render correctly
- ✅ All tables fit within margins
- ✅ All references are hyperlinked (blue color)
- ✅ No "??" symbols (indicates missing references)
- ✅ Page numbers correct
- ✅ Abstract on first page
- ✅ File size <10 MB

---

## 🎯 Ready for ManuscriptCentral

Once PDF is generated:

1. ✅ Upload to IEEE Access ManuscriptCentral: https://mc.manuscriptcentral.com/ieeeaccess
2. ✅ Use files from `submission_bundle/`:
   - `CalibraTeach_IEEE_Access.pdf` (main paper)
   - `COVER_LETTER.md` (cover letter)
   - `ABSTRACT_PLAINTEXT.txt` (copy into abstract field)
   - `KEYWORDS.txt` (copy into keywords field)
3. ✅ Follow steps in `SUBMISSION_CHECKLIST.md`

---

## 📚 Additional Template Features

### Hyperlinks

All references are hyperlinked (blue color) for easy navigation:
- Internal: Equations, figures, tables, sections
- External: DOIs in references (if added)

### Professional Tables

Using `booktabs` package for publication-quality tables:
- `\toprule`, `\midrule`, `\bottomrule` instead of `\hline`
- Proper spacing and alignment

### Mathematics

All equations properly numbered and formatted:
- Display equations: `\begin{equation}...\end{equation}`
- Inline math: `$...$`
- Multi-line derivations: `\begin{align}...\end{align}`

### IEEE Compliance

Template follows IEEE Access author guidelines:
- Proper section numbering
- IEEE citation style
- Standard nomenclature
- Professional tone

---

## ✨ Template Validation

**Compilation Status**: ✅ Verified  
**Length**: ~20 pages (appropriate for IEEE Access)  
**Equations**: 13 numbered equations (all render correctly)  
**Tables**: 9 tables (all professional quality)  
**References**: 20 citations (IEEE format)  
**Placeholders**: 0 remaining (all replaced with actual content)

---

## 🔄 Update Workflow

If you need to update metrics or content:

1. Edit `OVERLEAF_TEMPLATE.tex` in Overleaf
2. Recompile (Ctrl+S / Cmd+S)
3. Verify changes in PDF preview
4. Download updated PDF when satisfied

**Pro tip**: Use Overleaf's version history (Menu → History) to track changes

---

## 📞 Support

**Overleaf Help**: https://www.overleaf.com/learn  
**LaTeX Stack Exchange**: https://tex.stackexchange.com/  
**IEEE Author Resources**: https://ieeeauthorcenter.ieee.org/

---

**Last Updated**: March 2, 2026  
**Status**: ✅ Production-Ready

