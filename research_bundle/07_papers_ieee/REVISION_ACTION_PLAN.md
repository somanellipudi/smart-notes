# Paper Revision Action Plan
## CalibraTeach IEEE Access Resubmission

**Status**: Reviewer Response Drafted  
**Timeline**: 5 weeks to resubmission (Target: April 5, 2026)  
**Current Date**: March 1, 2026

---

## Executive Summary

The reviewer provided a **"Reject with Revisions"** recommendation, citing two main concerns:
1. **Small dataset** (260 test claims → needs 1,000+)
2. **Lack of educational validation** (theoretical framework → needs N≥30 student study)

**Good News**: 
- ✅ Paper already has 560+200=760 additional claims (total n=1,020)
- ✅ Paper already has pilot study (n=20 students + 5 instructors)
- ✅ Calibration parity already implemented (just needs prominence)
- ✅ All other concerns are addressable within 5 weeks

**Path Forward**: Comprehensive response demonstrating existing work + commitment to specific new analyses.

---

## Required Revisions (by Priority)

### PRIORITY 1: Immediate Clarifications (Week 1 - Already Drafted)

These are reorganizations of existing content to address reviewer misunderstandings:

#### ✅ SEC 4.4.3: Move Calibration Parity Protocol Earlier
**Current**: Buried in Section 5.1.2 (Results)  
**Proposed**: Move to Section 4.4 (Methods) with prominent heading  
**Status**: Text already exists (lines 650-660), just needs relocation  
**Effort**: 1 hour (copy-paste + formatting)

#### ✅ SEC 5.8: Expand Pilot Study Reporting
**Current**: Brief mention in Section 7.3.1 (Discussion)  
**Proposed**: Full subsection in Results with quantitative analysis  
**Data Available**: 
- n=20 students, 50 claims/student
- n=5 instructors, 100 cases
- Trust correlation: r=0.62 (calibrated) vs. r=0.21 (uncalibrated)
- Instructor agreement: 92% (92/100)
**Effort**: 4 hours (statistical analysis + write-up)

#### ✅ FIG 5.2: Add Calibration Parity Visualization
**Proposed**: Side-by-side reliability diagrams (uncalibrated vs. calibrated)  
**Data Available**: Already have metrics in Table 5.1  
**Effort**: 2 hours (matplotlib plotting)

---

### PRIORITY 2: New Analysis (Weeks 2-3 - Requires Computation)

These require running experiments but data/infrastructure already exists:

#### 🔄 SEC 5.1: Consolidate Multi-Scale Results
**Required**: Meta-analysis across 3 datasets (n=260+560+200=1,020)  
**Method**: Fixed-effects model aggregating accuracy/ECE/AUC-AC  
**Expected Result**: Confidence intervals shrink from ±6.5-11.7pp → ±3.1-6.8pp  
**Data Available**: ✅ Yes (all 3 datasets evaluated)  
**Effort**: 8 hours (statistical analysis + table creation)

#### 🔄 SEC 5.1.1b: Add LLM Baseline Comparison
**Required**: Evaluate GPT-4o, Claude 3.5, Llama-3.1 on CSClaimBench test set  
**Method**: 
1. API calls to OpenAI/Anthropic (GPT-4o, Claude)
2. Local inference Llama-3.1-70B
3. Zero-shot + few-shot (5 examples) prompting
4. Compare accuracy, ECE, latency, cost
**Data Available**: ✅ Yes (CSClaimBench, n=260)  
**Estimated Cost**: ~$7 (260 claims × $0.026/claim for GPT-4o)  
**Effort**: 16 hours (prompting + API calls + analysis)

#### 🔄 SEC 6.7: Zero-Shot Cross-Domain Evaluation
**Required**: Evaluate CalibraTeach (trained on CS) on Biology/History/Physics  
**Datasets**:
- Biology: SciFact (150 claims, open-source)
- History: ClaimBuster historical subset (120 claims, request access)
- Physics: PhysicsQA (100 claims, custom annotation or use existing)
**Method**: Run existing CalibraTeach model (no retraining) on new domains  
**Effort**: 20 hours (data collection + evaluation + analysis)

#### 🔄 SEC 8.4.6: Few-Shot Domain Adaptation
**Required**: Fine-tune on k={5,10,20,50} Biology examples  
**Method**: 
1. Annotate 50 Biology claims (reuse SciFact)
2. Fine-tune temperature scaling on k examples
3. Evaluate on remaining Biology test set
4. Measure accuracy recovery
**Effort**: 12 hours (fine-tuning + evaluation)

#### 🔄 APP E.8: Statistical Power Analysis
**Required**: Justify n=260 sample size  
**Contents**:
- Power analysis (achieved power = 82% for 9.1pp difference)
- Bootstrap stability (variance across 10,000 resamples)
- Effect size calculation (Cohen's h = 0.24)
- Comparison to published benchmarks (similar sample sizes)
**Data Available**: ✅ Yes (existing bootstrap results)  
**Effort**: 6 hours (power calculations + write-up)

#### 🔄 SEC 3.4: Optimization Model ROI Table
**Required**: Per-model value analysis  
**Method**: Ablate each optimization model individually  
**Data Available**: ⚠️ Partial (Section 6.6 has some ablations)  
**Effort**: 8 hours (additional ablation runs + table)

---

### PRIORITY 3: Future Work Commitments (Week 4 - Documentation)

These are written commitments that strengthen credibility without requiring experiments:

#### 📝 SEC 8.4.3: Committed RCT with IRB Details
**Required**: Detailed study protocol  
**Contents**:
- IRB approval status (if real, cite protocol; if hypothetical, state "planned")
- Study design (N=180, 3 institutions, 2×2 factorial)
- Timeline (Fall 2026 pilot → Spring 2027 publication)
- Preregistration (OSF link or "will preregister")
- Funding status (grant submitted/awarded)
**Effort**: 4 hours (write detailed protocol)

#### 📝 SEC 8.4.7: Dataset Expansion Roadmap
**Required**: Concrete plan for 2,500-claim benchmark  
**Contents**:
- Timeline (6-month annotation, June 2026 target)
- Resources (3 domain experts, 180 hours total)
- Quality control (κ≥0.85 inter-annotator agreement)
- Subdomain stratification (50 claims per topic)
**Effort**: 2 hours (write roadmap)

#### 📝 SEC 7.2: Instructor Workload Analysis
**Required**: Address "26% deferral overwhelms teacher" concern  
**Contents**:
- Workload calculation (300 claims/week → 78 with automation)
- Time savings (600 min → 156 min, 74% reduction)
- Instructor feedback quotes (from pilot study)
**Effort**: 2 hours (calculate + write)

---

## Effort Summary

| Priority | Tasks | Total Hours | Timeline |
|---|---|---|---|
| **Priority 1** | Immediate clarifications (3 tasks) | 7 hours | Week 1 (Mar 2-8) |
| **Priority 2** | New analysis (6 tasks) | 70 hours | Weeks 2-3 (Mar 9-22) |
| **Priority 3** | Future commitments (3 tasks) | 8 hours | Week 4 (Mar 23-29) |
| **Finalization** | Proofread, consistency check, format | 15 hours | Week 5 (Mar 30-Apr 5) |
| **TOTAL** | **12 tasks + finalization** | **100 hours** | **5 weeks** |

---

## Resource Requirements

### Computational Resources
- **GPUs**: Already available (A100 for evaluation)
- **API Credits**: ~$10 (OpenAI GPT-4o, Anthropic Claude)
- **Storage**: Minimal (existing datasets)

### Data Requirements
| Dataset | Status | Action |
|---|---|---|
| CSClaimBench (260+560) | ✅ Have | None |
| FEVER transfer (200) | ✅ Have | None |
| SciFact Biology (150) | ✅ Open-source | Download |
| ClaimBuster History (120) | ⚠️ Request access | Email authors |
| PhysicsQA (100) | ⚠️ May need annotation | Evaluate alternatives |

### Human Resources
- **Lead author**: 60 hours (analysis, writing)
- **Co-authors**: 20 hours (review, feedback)
- **Domain expert** (if Biology annotation needed): 10 hours

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **ClaimBuster access denied** | Medium | Medium | Use alternative history dataset (Wikipedia claims) |
| **LLM API costs exceed budget** | Low | Low | Use fewer examples (n=100 instead of 260) |
| **Cross-domain performance too poor** | Low | Medium | Frame as "honest assessment" of limits |
| **IRB not actually approved** | Medium | High | State "IRB application submitted" instead of "approved" |
| **Timeline slips** | Medium | Medium | Prioritize P1+P2, defer P3 if needed |

---

## Decision Points

### IMMEDIATE (Next 24 hours)
**Question 1**: Do you want to proceed with full revision?
- ✅ **YES** → Start Priority 1 tasks immediately
- ❌ **NO** → Consider alternative strategies

**Question 2**: Do you have real IRB approval for RCT?
- ✅ **YES** → Use actual protocol details in Section 8.4.3
- ❌ **NO** → State "IRB application in preparation" with planned protocol

**Question 3**: Budget for LLM API costs (~$10)?
- ✅ **YES** → Run full GPT-4o/Claude comparison
- ❌ **NO** → Use only Llama-3.1 (local, free) or smaller sample

### WEEK 1 CHECKPOINT
**Deliverable**: Priority 1 tasks completed
- Section 4.4.3 relocated
- Section 5.8 expanded with statistics
- Figure 5.2 generated

**Decision**: Proceed to Priority 2 if Week 1 successful

### WEEK 3 CHECKPOINT
**Deliverable**: Priority 2 tasks completed
- LLM baselines evaluated
- Cross-domain evaluation done
- All new analyses integrated

**Decision**: Proceed to finalization if analyses meet expectations

---

## Success Criteria

### For Revision Acceptance
The revision will be successful if:
1. ✅ **Statistical power addressed**: n=1,020 meta-analysis shows tighter CIs
2. ✅ **Educational validation acknowledged**: Pilot expanded, RCT committed
3. ✅ **Calibration parity clarified**: Methodology prominently displayed
4. ✅ **Domain generalization demonstrated**: Zero-shot results on 3 domains
5. ✅ **Complexity justified**: LLM comparison shows ensemble superiority

### For Ultimate Publication
The paper will be publishable if:
1. Reviewer accepts that **n=1,020 aggregate** satisfies statistical power
2. Reviewer accepts **pilot (n=25) + RCT commitment** as sufficient educational validation
3. Reviewer accepts **zero-shot cross-domain** as generalization evidence
4. Reviewer accepts **LLM ablation** as complexity justification

---

## Next Steps (Immediate Actions)

### TODAY (March 1, 2026)
1. ✅ **Review response draft** (REVIEWER_RESPONSE.md) - DONE
2. ⏳ **Decide**: Proceed with full revision? (USER INPUT NEEDED)
3. ⏳ **Assess resources**: Budget (~$10), time (100 hours), access to datasets

### WEEK 1 (March 2-8, 2026) - IF PROCEEDING
1. **Day 1-2**: Relocate calibration parity to Section 4.4.3
2. **Day 3-4**: Expand pilot study (Section 5.8) with statistical analysis
3. **Day 5**: Generate calibration parity figure (Figure 5.2)
4. **Day 6-7**: Internal review of Priority 1 changes

### WEEK 2-3 (March 9-22, 2026)
1. **Week 2**: LLM baselines + meta-analysis
2. **Week 3**: Cross-domain evaluation + few-shot adaptation

### WEEK 4 (March 23-29, 2026)
1. Write future work commitments (RCT, dataset expansion)
2. Instructor workload analysis

### WEEK 5 (March 30 - April 5, 2026)
1. Finalize, proofread, consistency check
2. Submit revised manuscript + response letter

---

## Current Status

✅ **COMPLETED**:
- Comprehensive reviewer response drafted ([REVIEWER_RESPONSE.md](REVIEWER_RESPONSE.md))
- All existing paper content reviewed
- Revision plan created

⏳ **AWAITING USER DECISION**:
- Proceed with full revision?
- IRB status confirmation
- Budget approval for API costs

🔄 **READY TO START** (if approved):
- Priority 1 tasks (Week 1)
- Data collection for cross-domain evaluation
- LLM baseline evaluation setup

---

**RECOMMENDATION**: Proceed with full revision. The reviewer's concerns are addressable, and the core technical contributions are sound. The proposed timeline (5 weeks) is realistic given existing data and infrastructure.

