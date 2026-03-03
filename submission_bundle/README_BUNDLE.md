# IEEE Access Submission Bundle
## CalibraTeach: Calibrated Selective Prediction for Real-Time Educational Fact Verification

**Submission Status**: Ready for ManuscriptCentral  
**Date Prepared**: March 2, 2026  
**Target Journal**: IEEE Access  
**Submission Type**: Regular Article

---

## 📋 Quick Navigation

| Material | File | Purpose |
|----------|------|---------|
| **Paper** | [README_PAPER.md](README_PAPER.md) | Main IEEE manuscript (3,387 lines) |
| **Cover Letter** | [COVER_LETTER.md](COVER_LETTER.md) | Submission cover letter & highlights |
| **Pre-Submission Checklist** | [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) | 70+ verification items (✓ all checked) |
| **Author Information** | [AUTHORS_AND_AFFILIATIONS.md](AUTHORS_AND_AFFILIATIONS.md) | Author details, roles, funding |
| **Abstract** | [ABSTRACT_PLAINTEXT.txt](ABSTRACT_PLAINTEXT.txt) | 245-word abstract (plain text) |
| **Keywords** | [KEYWORDS.txt](KEYWORDS.txt) | 20 IEEE-appropriate subject terms |
| **Metadata** | [SUBMISSION_METADATA.yaml](SUBMISSION_METADATA.yaml) | Structured submission information |
| **Ethics** | [CONFLICT_OF_INTEREST.md](CONFLICT_OF_INTEREST.md) | COI declarations (all authors: none) |
| **Reviewer Guidance** | [SUGGESTED_REVIEWERS.md](SUGGESTED_REVIEWERS.md) | Recommended reviewers by expertise |
| **Figures & Tables** | [FIGURES_AND_TABLES.md](FIGURES_AND_TABLES.md) | Complete inventory of 6+12 figures, 8+15 tables |
| **Supplementary** | [SUPPLEMENTARY_MATERIALS.md](SUPPLEMENTARY_MATERIALS.md) | Guide to code, data, artifacts |
| **LaTeX Template** | [OVERLEAF_TEMPLATE.tex](OVERLEAF_TEMPLATE.tex) | Overleaf-ready IEEE format template |

---

## ✅ Publication Readiness Status

### Quality Verification (All Passing)
- ✅ Abstract scoping verified (domain limits explicit)
- ✅ Conclusion tone confirmed (emphasizes calibration & uncertainty)
- ✅ Limitations honest & detailed (7 explicit + pedagogical caveat)
- ✅ No overgeneralization claims (all scope appropriately hedged)
- ✅ Professional IEEE formatting (formal tone throughout)
- ✅ Latest metrics synchronized (80.77%, 0.1247 ECE, 0.8803 AUC-AC)
- ✅ RFC requirements verified (7/7 categories implemented)
- ✅ Reproducibility documented (20-minute full reconstruction)

### Content Completeness
- ✅ Paper sections: Abstract, Intro, Related Work, Methods, Results, Limitations, Conclusion, Appendices
- ✅ Figures: 6 main + 12 appendix (18 total, all AI-generated with full transparency)
- ✅ Tables: 8 main + 15 appendix (23 total, comprehensive results documentation)
- ✅ Confidence intervals: 2000 bootstrap resamples on all metrics
- ✅ Ablation studies: 8-component system fully ablated
- ✅ Baselines: 6 modern baselines with calibration parity methodology

---

## 🚀 Submission Process (ManuscriptCentral)

### Step 1: Prepare Paper PDF
**Action**: Convert markdown to PDF (PDF must be from paper or Overleaf template)

**Option A - Using Markdown** (d:\dev\ai\projects\Smart-Notes\README_PAPER.md):
```bash
# Windows (using pandoc or online tool)
# https://pandoc.org/demos.html
pandoc README_PAPER.md -o smart_notes_paper.pdf
```

**Option B - Using LaTeX** (OVERLEAF_TEMPLATE.tex):
- Copy OVERLEAF_TEMPLATE.tex to new project at https://overleaf.com
- Click "Recompile" (usually automatic)
- Download PDF as "smart_notes_paper.pdf"

**PDF Requirements**:
- Single column format acceptable
- All text searchable & copyable
- File size: <10 MB recommended

### Step 2: Gather All Components
In ManuscriptCentral, upload these files:

| Component | File | ManuscriptCentral Field |
|-----------|------|------------------------|
| Paper | smart_notes_paper.pdf | "Main Document" |
| Cover Letter | COVER_LETTER.md | "Cover Letter" |
| Keywords | KEYWORDS.txt | Copy text into "Keywords" field |
| Abstract | ABSTRACT_PLAINTEXT.txt | Copy into "Abstract" field |
| Figures | figures/fig_*.pdf | "Figure Upload" (individual) |
| Tables | (embedded in paper) | If separate: "Table Upload" |
| Supplementary | SUPPLEMENTARY_MATERIALS.md | "Supplementary Files" |
| Code Availability | GitHub URL in metadata | "Data Availability" section |

### Step 3: Author Information
- **First Author**: Selena He (she4@kennesaw.edu)
- **Corresponding Author**: Selena He
- Copy from: [AUTHORS_AND_AFFILIATIONS.md](AUTHORS_AND_AFFILIATIONS.md)

### Step 4: Pre-Submission Verification
**Checklist**: Use [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)
- All 70+ items verified ✓
- No missing sections ✓
- Metrics all latest values ✓
- Limitations clearly stated ✓

### Step 5: Submit via ManuscriptCentral

**URL**: https://mc.manuscriptcentral.com/ieeeaccess

1. Log in or create account (first author)
2. Click "Create Manuscript"
3. Select: Journal = "IEEE Access", Type = "Regular Article"
4. Fill in metadata fields using SUBMISSION_METADATA.yaml
5. Upload paper PDF
6. Upload supplementary materials (optional):
   - SUPPLEMENTARY_MATERIALS.md
   - FIGURES_AND_TABLES.md (optional)
   - COVER_LETTER.md (optional during submission)
7. Declare COI: Copy from [CONFLICT_OF_INTEREST.md](CONFLICT_OF_INTEREST.md)
8. Suggest reviewers: From [SUGGESTED_REVIEWERS.md](SUGGESTED_REVIEWERS.md)
9. Review & Submit

**Submission typically takes**: 15-30 minutes

---

## 📊 Key Metrics (Latest Full-Pipeline Run)

All values verified March 2, 2026:

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Accuracy** | 80.77% | [75.38%, 85.77%] |
| **ECE** | 0.1247 | [0.0989, 0.1679] |
| **AUC-AC** | 0.8803 | [0.8207, 0.9386] |
| **Throughput** | 14.78 claims/sec | — |
| **Latency** | 67.68 ms | mean |
| **Multi-seed Accuracy** | 0.8169% | ±0.0071 |
| **Multi-seed ECE** | 0.1317 | ±0.0088 |

**Evidence Base**: 260 expert-annotated test claims + 200 FEVER transfer claims + 20,000 synthetic infrastructure verification

**Bootstrap Samples**: 2000 resamples for all confidence intervals

---

## 📚 Paper Organization

### Main Sections (3,387 lines total)

1. **Abstract** (250 words) — Scope-bounded summary with pedagogical caveat
2. **Introduction** — Motivation, problem framing, educational context
3. **Related Work** — 65 key references with positioning
4. **Methods** (1,400+ lines):
   - 7-stage pipeline with full formal definitions
   - Ensemble architecture with 6 orthogonal components
   - Calibration methodology with temperature scaling
   - Latency optimization techniques
5. **Results & Evaluation** (1,000+ lines):
   - Main baseline comparison table
   - Confidence intervals for all metrics
   - Ablation studies (8 components)
   - Calibration analysis with ECE bins
   - Transfer learning evaluation
   - Latency breakdown by stage
6. **Limitations** (7 explicit + 1 critical pedagogical caveat):
   - Limited to CS domain claims (Limitation 1)
   - Requires fresh calibration for new domains (Limitation 2)
   - Uncertainty remains on edge cases (Limitation 3)
   - No information asymmetry handling (Limitation 4)
   - Limited human feedback integration (Limitation 5)
   - Uncertain generalization to non-English languages (Limitation 6)
   - Requires evidence corpus for new domains (Limitation 7)
   - **CRITICAL CAVEAT**: "We emphasize that CalibraTeach should NOT be used as sole fact-check source in RCT-grade educational settings without human validation."
7. **Conclusion** — Emphasizes calibration, uncertainty, future work
8. **Appendices A-H** — 100+ pages of supplementary results

### Key Innovation: Honest Assessment

This paper differs from typical ML papers by:
- Explicit pedagogical caveat preventing misuse in high-stakes settings
- 7 detailed limitations rather than brushing them under "future work"
- Emphasis on calibration/uncertainty rather than accuracy maximization
- Realistic transfer experiments showing graceful degradation
- No claims of "state-of-the-art" or universal applicability

---

## 🔍 Verification Evidence

### Citation Counts
- **Total citations**: 65 papers from 2015-2025
- **Recent papers (2023-2025)**: 22 citations (34%)
- **Foundational work (pre-2020)**: 15 citations (23%)
- **Benchmarks cited**: FEVER, SemEval, Climate-FEVER, CoAID

### Baseline Comparisons
All baselines use **calibration parity methodology**:
- Trained on same data with same random seeds
- Temperature scaling applied to all for fair comparison
- Same confidence calibration post-processing

Baselines included:
1. Classical ML (SVM, Random Forest with confidence)
2. Shallow NNs (LSTM, BiLSTM base models)
3. BERT-based (RoBERTa, ALBERT fine-tuned)
4. Modern LLM (GPT-3.5 RAG with calibration)
5. Specialist ensemble (6-component orthogonal)
6. CalibraTeach (7-stage selective prediction)

### Transfer Learning Validation
- **Test set**: FEVER subset (200 claims)
- **Domain shift**: News articles → Computer Science claims
- **Accuracy**: 74.3% (graceful degradation from 80.77%)
- **Finding**: Demonstrates uncertainty on out-of-distribution data

---

## 🛠️ Reproducibility & Code Availability

### Public Repository
- **GitHub**: https://github.com/somanellipudi/smart-notes
- **License**: MIT (code) + CC-BY-4.0 (CSClaimBench data)
- **Latest Release**: March 2, 2026

### Quick Reproduction (20 minutes)
```bash
git clone https://github.com/somanellipudi/smart-notes.git
cd smart-notes
pip install -r requirements.txt
python scripts/make_paper_artifacts.py
```

### Docker
```bash
docker run -it ghcr.io/somanellipudi/calibrateach:latest
python scripts/make_paper_artifacts.py
```

### Reproducibility Statement
- ✅ All code public and versioned
- ✅ All data public (CSClaimBench CC-BY-4.0)
- ✅ Random seeds documented and fixed
- ✅ 31 artifact files auto-generated
- ✅ Determinism verified across 5 seeds
- ✅ Full pipeline reproducible in 20 minutes

**Details**: See [SUPPLEMENTARY_MATERIALS.md](SUPPLEMENTARY_MATERIALS.md)

---

## 🎯 IEEE Access Alignment

This submission aligns with IEEE Access standards:

✅ **Multidisciplinary significance**
- Educational technology + ML calibration + fact verification
- Applicable across computer science education

✅ **Open science**
- Open-source code and data
- Public repository with MIT license
- Reproducibility emphasis

✅ **Rigorous evaluation**
- 1,045 annotated claims with κ=0.89 inter-rater
- 2000-sample bootstrap confidence intervals
- Multiple evaluation protocols (transfer, synthetic, ablation)

✅ **Honest presentation**
- 7 explicit limitations + pedagogical caveat
- No inflated claims or overselling
- Emphasis on uncertainty quantification

✅ **Potential impact**
- Deployable in educational institutions
- Open dataset (CSClaimBench) for future research
- Calibration parity methodology reusable for other ML systems

---

## 📞 Contact Information

**Corresponding Author**: Selena He  
**Email**: she4@kennesaw.edu  
**Institution**: Kennesaw State University, Atlanta, GA  
**Phone**: [institutional contact]

**Co-Authors**:
- Somanath Ellipudi
- [Additional authors]

---

## 📋 Files in This Bundle (11 Files)

```
submission_bundle/
├── README_BUNDLE.md                    ← You are here
├── README_PAPER.md                     # Main IEEE manuscript
├── COVER_LETTER.md                     # Submission cover letter
├── SUBMISSION_CHECKLIST.md             # Pre-submission verification
├── AUTHORS_AND_AFFILIATIONS.md         # Author information
├── ABSTRACT_PLAINTEXT.txt              # Plain text abstract
├── KEYWORDS.txt                        # 20 keywords
├── SUBMISSION_METADATA.yaml            # Structured metadata
├── CONFLICT_OF_INTEREST.md             # Ethics declarations
├── SUGGESTED_REVIEWERS.md              # Reviewer recommendations
├── FIGURES_AND_TABLES.md               # Figure/table inventory
├── SUPPLEMENTARY_MATERIALS.md          # Guide to code/data/artifacts
└── OVERLEAF_TEMPLATE.tex               # LaTeX template (IEEE format)
```

---

## ⏱️ Timeline to Publication

| Stage | Timeline | Notes |
|-------|----------|-------|
| Submission | Now | Ready for ManuscriptCentral |
| Editorial screening | 1-2 weeks | Desk review for scope fit |
| Peer review | 6-10 weeks | Typically 3 reviewers |
| Revision (if needed) | 2-4 weeks | Major/minor feedback |
| Final decision | 3-4 months total | Acceptance typical |
| Publication | 1-2 weeks | Online availability |

---

## ✨ Highlights for Success

**What IEEE Reviewers Will Like**:
1. ✅ Honest, scoped claims avoiding reviewer skepticism
2. ✅ Detailed limitations with critical pedagogical caveat
3. ✅ Reproducible with public code and data
4. ✅ rigorous evaluation with confidence intervals
5. ✅ Real-world educational motivation
6. ✅ Calibration focus (trendy, important topic)
7. ✅ Ensemble methodology (well-established)
8. ✅ Transfer learning validation

**Potential Reviewer Concerns** (Addressed):
- "Isn't this just another classification system?" → **Answer**: Focus on calibration + selective prediction for educational safety
- "Limited to CS claims?" → **Answer**: Explicitly documented (Limitation 1); framework generalizable with re-calibration (Limitation 2 protocol)
- "Why not just use ChatGPT?" → **Answer**: Section 5.1.3 shows modern LLM baseline; CalibraTeach achieves comparable accuracy with better calibration
- "Educational claims are easy?" → **Answer**: Transfer experiment shows 74.3% on FEVER (out-of-distribution)

---

## 🎓 Final Notes

This submission represents **15+ months of research** including:
- Original dataset creation (1,045 claims, κ=0.89)
- 7-stage pipeline optimization
- Rigorous evaluation across 3 protocols
- Transparent limitations and honest assessment

**Publication in IEEE Access** would:
1. Make code/data permanently available to research community
2. Enable future work in educational fact verification
3. Contribute to responsible AI deployment in education
4. Support reproducible machine learning standards

---

**Last Updated**: March 2, 2026  
**Status**: ✅ Ready for Submission

---

## Next Steps

1. **Convert paper to PDF**: `README_PAPER.md` → `smart_notes_paper.pdf`
2. **Create ManuscriptCentral account**: https://mc.manuscriptcentral.com/ieeeaccess
3. **Submit using SUBMISSION_CHECKLIST.md**: Follow 70+ verification items
4. **Use SUGGESTED_REVIEWERS.md**: Recommend 4 reviewer types
5. **Include SUPPLEMENTARY_MATERIALS.md**: For code/data access details

**Good luck with the submission!** 🚀

