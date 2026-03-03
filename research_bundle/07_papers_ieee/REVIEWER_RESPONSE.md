# RESPONSE TO REMAINING MAJOR CONCERNS (IEEE ACCESS)

This document provides a point-by-point response to concerns 3.1–3.5 and minor issues. All items below are now addressed in the revised manuscript.

---

## 3.1 Dataset Size Still Limits Strength of Claims

### Reviewer concern
The expert test set (n=260) may be too small for robust calibration, stable risk-coverage interpretation, and generalization claims.

### Action taken
1. Added explicit bootstrap 95% confidence intervals for headline metrics (accuracy, ECE, AUC-AC).
2. Added detailed per-class results (confusion matrix + precision/recall/F1).
3. Added explicit seed-variance and determinism analysis subsection.
4. Added extended expert-annotated evaluation (n=560) with CIs to demonstrate stability.
5. Clarified that 20K synthetic runs validate infrastructure scalability, not primary predictive claims.

### Evidence in manuscript
- Section 5.3: 95% CIs for Accuracy/ECE/AUC-AC (bootstrap, 10,000 iterations)
- Section 5.6: Paired bootstrap significance analysis
- Section 5.6.1: Seed Variance and Determinism Analysis
- Section 5.6.2: Per-Class Detailed Results
- Section 5.5: CSClaimBench-Extended (n=560) with 95% CIs
- Sections 5.1.3 and Appendix E.11: explicit separation of real-claim evaluation vs synthetic infrastructure stress testing

### Clarification on “why not solved by 20K test?”
The 20K benchmark is synthetic and used for system stress/scalability validation. Statistical scientific claims (accuracy/calibration/selective prediction quality) are anchored to expert-labeled CSClaimBench data (n=260 primary, n=560 extended), hence reviewer emphasis on CI and variance reporting for expert-labeled sets.

---

## 3.2 Calibration Analysis Could Be Deeper

### Reviewer concern
Temperature scaling alone may be insufficient; reviewers requested method comparison, bin stability, and probabilistic clarity.

### Action taken
1. Added calibration-method comparison (Temperature vs Platt vs Isotonic), including rationale for final method choice.
2. Added reliability-diagram discussion and pointer to detailed bin-level table.
3. Added ECE sensitivity analysis across bin counts in appendix.
4. Clarified probabilistic meaning of calibrated correctness confidence.

### Evidence in manuscript
- Section 3.5.1: Calibration method comparison (including isotonic regression)
- Section 5.1.2: Reliability Diagram discussion
- Appendix E.1: Detailed bin-by-bin calibration table and ECE bin sensitivity
- Sections 3.5 and 4.3: definition of calibrated correctness probability and ECE_correctness

---

## 3.3 Baseline Scope

### Reviewer concern
Need contemporary LLM baseline context (RAG-style GPT-class systems with confidence output) or narrower claims.

### Action taken
1. Added dedicated LLM baseline section with GPT-4o, Claude Sonnet 4, and Llama 3.2 results.
2. Reported accuracy, macro-F1, ECE, AUC-AC, latency, and cost per claim.
3. Scoped claims explicitly: CalibraTeach emphasizes calibrated, low-latency, deterministic, low-cost educational deployment rather than highest raw accuracy.

### Evidence in manuscript
- Section 5.1.3: Large Language Model Baseline Comparison
- Section 9 and Abstract language updated to avoid overclaiming vs larger proprietary LLMs

---

## 3.4 Error Analysis Needs Structure

### Reviewer concern
Need categorized error taxonomy, example failures, and stage-level attribution.

### Action taken
1. Added structured error taxonomy table by pipeline stage.
2. Added representative failure examples.
3. Added retrieval vs NLI vs aggregation vs annotation/input breakdown and mitigation paths.

### Evidence in manuscript
- Section 6.3: Error Taxonomy and Failure Mode Analysis (n=49 errors)

---

## 3.5 Deployment Interpretation of Selective Prediction

### Reviewer concern
Risk-coverage must be operationally actionable (recommended coverage, acceptable risk, abstention workflow).

### Action taken
1. Added explicit operating-point table (threshold, coverage, risk, precision).
2. Added selected operating point recommendation with rationale (90.4% precision @ 74% coverage).
3. Added operational deployment decision tree with scenario-specific thresholds.
4. Added abstention-routing workflow and safety overrides.

### Evidence in manuscript
- Section 5.2: Risk-Coverage operating points and selected threshold rationale
- Section 7.3.2: Operational Deployment Decision Tree

---

## 4. Minor Issues

### 4.1 Informal formatting
Resolved: Informal checkmark markers removed from manuscript text and tables; high-visibility sections normalized to formal IEEE style.

### 4.2 AUC-AC definition
Resolved: AUC-AC is formally defined, linked to AURC, and the numerical interpretation is stated in methodology/results.

### 4.3 Cost assumptions
Resolved: Added explicit “Cost Assumptions and Limitations” block clarifying what is included/excluded and how to interpret cloud-equivalent GPU-time costs.

### 4.4 Statistical notation consistency
Resolved in major result sections: CI formatting and significance phrasing standardized in Sections 5.3/5.6 and related tables.

---

## Final status

All remaining major concerns (3.1–3.5) and minor issues have corresponding manuscript revisions with explicit section-level evidence. The revision now provides:

- Statistical rigor (bootstrap CIs, per-class analysis, seed/determinism variance handling)
- Deeper calibration analysis (method comparison, reliability, bin sensitivity)
- Competitive baseline context (LLM comparison)
- Structured failure analysis
- Operational deployment guidance for selective prediction

No additional experiments are required to answer the listed reviewer requests; the current revision package addresses them directly.