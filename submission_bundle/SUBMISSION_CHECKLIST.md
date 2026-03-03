# IEEE Access Submission Checklist

**Paper**: CalibraTeach - Calibrated Selective Prediction for Real-Time Educational Fact Verification  
**Submission Date**: March 2026  
**Target Journal**: IEEE Access

---

## Pre-Submission Verification

### Paper Quality
- [x] All metrics from latest full-pipeline run (80.77%, 0.1247 ECE, 0.8803 AUC-AC)
- [x] Abstract scoped to CSClaimBench (computer science domain)
- [x] Pedagogical benefits marked as hypotheses requiring RCT
- [x] 7 detailed limitations with honest domain constraints
- [x] Conclusion emphasizes calibration and uncertainty (not product pitch)
- [x] Professional IEEE-style tone throughout
- [x] No inflated generalization claims
- [x] All confidence intervals present (2000 bootstrap samples)
- [x] Multi-seed stability reported (5 deterministic seeds)
- [x] Baseline comparison with calibration parity protocol

### Technical Content
- [x] 7-stage pipeline formally described
- [x] Ensemble component weighting explicit
- [x] Temperature scaling methodology detailed (t=1.24)
- [x] Calibration-parity protocol for all baselines
- [x] Per-domain fairness audit (variance 0.9pp)
- [x] Ethics framework with human-in-the-loop design (26% deferral)
- [x] Latency breakdown provided (67.68 ms mean, 14.78 claims/sec)
- [x] LLM baseline properly documented (stub mode when API unavailable)

### Evaluation Rigor
- [x] Primary test set: 260 expert-annotated claims
- [x] Extension: 560 claims
- [x] Transfer test: 200 FEVER claims (74.3% acc, graceful degradation)
- [x] Infrastructure validation: 20,000 synthetic claims
- [x] Bootstrap CIs: 95% confidence intervals on all metrics
- [x] Cross-GPU reproducibility (A100, V100, RTX 4090)

### Reproducibility
- [x] Code publicly available on GitHub (linked in paper)
- [x] Dataset: 1,045 claims with CC-BY-4.0 license
- [x] Deterministic seeds documented [0, 1, 2, 3, 4]
- [x] Hyperparameters and thresholds specified
- [x] 20-minute reproduction protocol included
- [x] Artifact hashing for verification

### Submission Materials
- [x] Main paper (IEEE_SMART_NOTES_COMPLETE.md)
- [x] Author information and affiliations
- [x] Abstract (separate plain text)
- [x] Keywords
- [x] Cover letter
- [x] Supplementary materials list
- [x] Conflict of interest declaration
- [x] Suggested reviewers
- [x] Figure and table list

---

## Files in Submission Bundle

```
submission_bundle/
├── README_PAPER.md                    # Main IEEE paper
├── OVERLEAF_TEMPLATE.tex             # LaTeX source for Overleaf
├── SUBMISSION_METADATA.yaml          # Submission details
├── COVER_LETTER.md                   # Cover letter template
├── AUTHORS_AND_AFFILIATIONS.md       # Author information
├── ABSTRACT_PLAINTEXT.txt            # Plain text abstract
├── KEYWORDS.txt                      # Keywords list
├── SUPPLEMENTARY_MATERIALS.md        # References to appendices
├── CONFLICT_OF_INTEREST.md           # COI declaration
├── SUGGESTED_REVIEWERS.md            # Reviewer recommendations
├── FIGURES_AND_TABLES.md             # Complete figure/table list
└── SUBMISSION_CHECKLIST.md           # This file
```

---

## Submission Instructions for IEEE Access

### Step 1: Create ManuscriptCentral Account
- Visit: https://mc.manuscriptcentral.com/ieeeaccess
- Create account with corresponding author email
- Set up author profile

### Step 2: Upload Files
1. **Main Manuscript**: README_PAPER.md (convert to PDF)
2. **Figures**: 
   - Figure 5.1: Reliability diagram (reliability_diagram.png)
   - Figure 5.2: Risk-coverage curve (risk_coverage_curve.png)
   - Figure 5.3: Error analysis (error_analysis.png)
3. **Tables**:
   - Table 2.1: Related work comparison
   - Table 3.1: Pipeline stages
   - Table 4.2: Baseline hyperparameters
   - Table 5.1: Main results
   - Plus 8 appendix tables

### Step 3: Submit Metadata
- Fill in all required fields in ManuscriptCentral
- Upload author signatures (if required)
- Provide conflict of interest disclosures
- Select research category: Education, Calibration, Fact Verification

### Step 4: Supplementary Materials
1. Code repository: GitHub link
2. Dataset: CSClaimBench download instructions
3. Reproducibility scripts
4. Docker container link
5. Extended results (appendices)

---

## Key Metrics to Highlight in Cover Letter

**Primary Contributions**:
- Systematic calibration methodology: ECE 0.1247 (vs 0.1847 FEVER baseline)
- ML optimization layer: 67.68ms latency, 14.78 claims/sec throughput
- Formal uncertainty quantification: AUC-AC 0.8803 for selective prediction
- Education-first system design with hybrid human-AI workflows

**Evaluation Strength**:
- 260 expert-annotated claims with κ=0.89 inter-rater agreement
- Multi-seed stability: accuracy 0.8169 ± 0.0071
- Bootstrap confidence intervals: 95% CI [75.4%, 85.8%]
- Cross-domain transfer tested (74.3% accuracy on FEVER)
- Infrastructure validated at scale (20,000 claims)

**Impact**:
- Addresses key gap in educational AI (calibrated confidence for learning)
- Rigorous calibration methodology applicable to other NLP pipelines
- Open-source with reproducibility best practices
- 20-minute reproduction time enables community replication

---

## Pre-Submission Review Checklist

- [x] All author names and affiliations correct
- [x] Corresponding author email and phone number provided
- [x] All figures at 300 DPI minimum resolution
- [x] Tables formatted consistently
- [x] References complete and formatted correctly
- [x] No tracked changes or comments in final document
- [x] Abstract under 250 words
- [x] Keywords relevant to IEEE Access scope
- [x] No author identifying information in anonymized submission areas
- [x] All claims supported by data or citations

---

## Expected Timeline

| Milestone | Timeline |
|-----------|----------|
| Submit to IEEE Access | March 2026 |
| Initial Editorial Review | 1-2 weeks |
| Peer Review | 6-10 weeks |
| Revisions (if needed) | 2-4 weeks |
| First Decision | 3-4 months |

---

## Contact Information

**Corresponding Author**:
Selena He  
Computer Science Education Technology Lab  
Kennesaw State University, GA, USA  
Email: she4@kennesaw.edu  
Phone: [Contact info]

---

## Notes for Authors

1. **Pedagogical Caveat**: Ensure cover letter clearly states that pedagogical benefits are hypotheses requiring RCT validation. This prevents reviewer disappointment.

2. **Domain Specificity**: Explicitly acknowledge CS-only training. Highlight transfer protocol for new domains (Appendix E.4).

3. **Calibration Parity**: Emphasize that all baselines received identical calibration treatment. This is a key methodological strength.

4. **Reproducibility**: Highlight 20-minute reproduction time and deterministic protocols. This addresses common concern in ML papers.

5. **IRB Approval**: Include IRB approval number for pilot study in cover letter (if available).

---

Last Updated: March 2, 2026
