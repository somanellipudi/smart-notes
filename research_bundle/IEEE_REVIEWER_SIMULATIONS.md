# Simulated IEEE Access Reviewer Reports
**Paper**: Smart Notes: Calibrated Fact Verification for Educational AI  
**Submission Date**: February 2026  
**Decision Simulation**: Based on corrected draft (after §5.1 restructuring to separate CSClaimBench from synthetic)

---

## Reviewer 1: Dr. Sarah Chen (Calibration & Uncertainty Quantification Expert)
**Affiliation**: Stanford, Probabilities & Calibration Lab  
**Expertise**: Temperature scaling, ECE metrics, selective prediction  
**Review Tone**: Constructive, technically rigorous, wants to promote good calibration work

### Summary
This paper makes a solid contribution to fact verification by prioritizing calibration alongside accuracy. The ECE 0.0823 result is competitive with modern QA systems, and the component-based ensemble approach is principled. However, the paper oversells the pedagog contributions and under-addresses cross-domain generalization concerns.

### Detailed Comments

**STRENGTH 1: Rigorous Calibration Methodology** ⭐⭐⭐⭐⭐
- Multi-component ensemble (§3.4) with learned weights is well-motivated
- Temperature scaling post-aggregation (§3.5) correctly avoids refitting (prevents overfitting) 
- **Suggested metric name**: Authors call it "ECE_correctness" but should clarify vs. standard ECE definitions. Recommend brief footnote explaining single-event calibration vs. multiclass simplex calibration
- **Missing**: Calibration reliability diagram. Not mentioned in main paper—should include 10-bin reliability plot showing Smart Notes vs. FEVER baseline. This would visually validate calibration claims powerfully

**STRENGTH 2: Bootstrap Statistical Analysis** ⭐⭐⭐⭐
- 10,000-resampling bootstrap (§5.5) is appropriate and credible
- Paired evaluation and significance testing rigorous
- CI [+6.5pp, +11.7pp] reasonable given n=260

**CONCERN 1: Small Test Set and Generalization** ⚠️⚠️⚠️
- Only 260 test claims. FEVER has 19,998. Power limited.
- Per-domain results (§6.4) show accuracy range 79.2%-80.1%, but small N per domain (50-54 claims):
  - Networks: 52 claims → CI width ±8.6pp (huge uncertainty)
  - Claimed "excellent cross-domain transfer" but small subsets don't support this
- **Recommendation**: Expand to 500+ test claims or acknowledge limitations more explicitly

**CONCERN 2: Baseline Fairness Unclear** ⚠️⚠️⚠️
- Retrained FEVER, SciFact on CSClaimBench training split (524 claims)
- But tuning parity unclear:
  - Did you perform hyperparameter search on all baselines equally?
  - Same architectural constraints (model size, depth)?
  - Same data augmentation strategy?
- **Recommendation**: Add short baseline methodology table explaining tuning parity

**CONCERN 3: ECE Methodology Clarity** ⚠️⚠️
- ECE_correctness definition is non-standard. Most ECE literature assumes multiclass probability simplex
- Your approach (binary correctness event) is valid but needs clearer positioning
- **Recommendation**: Add brief explanation in §3.5 or §4.3 why you chose binary event vs. multiclass. Reference Braga et al. 2024 on selective prediction ECE definitions

**TECHNICAL ISSUE: Synthetic vs. Real Communication** ✓ RESOLVED
- **Previous version confusing**: Main results jumbled synthetic baseline table with real CSClaimBench results
- **New version clear**: §5.1 now explicitly states CSClaimBench is primary; synthetic moved to Appendix D
- This was CRITICAL concern; now resolved. Paper is vastly improved on this axis.

**MINOR ISSUE**: Component Importance Interpretation
- Learned weights: [0.18, 0.35, 0.10, 0.15, 0.10, 0.12]
- Claim: S₂ (Entailment) "dominant" with 35%
- But in ablation (§6.1), removing S₂ costs only -8.1pp. Is 35% weight justified?
- **Recommendation**: Compare learned weights vs. empirical ablation importance. If they differ, explain discrepancy (e.g., weights capture correlation structure, not individual contribution)

### Questions for Authors

1. Did you use the same validation set for temperature scaling and component weight learning? If yes, leakage risk. If no, why not jointly optimize?

2. Why not include Monte Carlo Dropout (ensemble of BART at test time) as a baseline? That's standard UQ approach you briefly mention but don't thoroughly evaluate.

3. Per-domain generalization (79.7% avg, tight variance): Is this because you trained on balanced 5-domain split? What happens if you train only on Networks domain?

### Recommendation
**ACCEPT with MINOR REVISIONS**

**Strengths**:
- ✅ Rigorous calibration work with appropriate statistical testing
- ✅ Pedagogical framing novel for fact verification (good contribution)
- ✅ Reproducibility infrastructure exemplary
- ✅ Synthetic vs real conflict now resolved in revised version

**Weaknesses**:
- ⚠️ Small test set limits generalization claims
- ⚠️ Baseline fairness methodology incomplete
- ⚠️ ECE_correctness definition needs clarity for broader community

**Minor Revisions Requested**:
1. Add reliability diagram to §5.1 (visual ECE validation)
2. Clarify ECE_correctness definition vs. standard ECE
3. Add baseline methodology table (tuning parity)
4. Address small test set limitation in discussion

**Estimated Revision Time**: 1-2 weeks (mostly adding figures + 2-3 paragraphs)

### Questions for Editor
- Is 260 test claims sufficient for IEEE Access? (vs. IEEE TPAMI standard of 500+)
- Should we require multi-domain evaluation for educational AI papers?

---

## Reviewer 2: Prof. Michael Rodriguez (ML Methods & Transfer Skeptic)
**Affiliation**: CMU, Machine Learning Bias & Robustness Lab  
**Expertise**: Transfer learning, domain shift, fair representation** **Review Tone**: Skeptical but fair, wants to see evidence for strong claims

### Summary
This paper is technically competent but makes overstated novelty claims and under-addresses domain specificity. The system is optimized heavily for CS education; generalization to other domains is completely unvalidated. The "hybrid workflow" framing is interesting but not empirically tested (no user study). I would recommend ACCEPT or ACCEPT WITH REVISIONS depending on how seriously authors take these concerns.

### Detailed Comments

**STRENGTH 1: Multi-Component Ensemble Design** ⭐⭐⭐⭐
- 6-component architecture (§3) is well-motivated
- Learned logistic regression weights more principled than pre-set weighting
- Ensemble naturally prevents overconfidence (good insight)

**STRENGTH 2: Optimization Layer** ⭐⭐⭐
- 8 optimization models achieving 18× speedup (§6.6) is impressive engineering
- Cache, pre-screening, ranking all sensible
- BUT §2 should clearly separate CORE METHOD NOVELTY from ENGINEERING OPTIMIZATION
  - Core novelty: Ensemble + calibration + selective prediction
  - Engineering novelty: Optimization layer (useful but not core contribution)
  - Current paper conflates the two, obscuring what's actually novel

**CONCERN 1: Overclaimed Cross-Domain Generalization** ⚠️⚠️⚠️⚠️
- Paper claims: "Per-domain accuracy range: 79.2%-80.1% (tight ±0.45pp variance)" (§6.4)
- This is WITHIN-DOMAIN variation (5 CS subdomains), not cross-domain
  - Networks, Databases, Algorithms, OS, Distributed Systems = all CS education
  - This is not "domain transfer"; it's subdomain variation
- **What's NOT tested**:
  - History claims (very different entity types, reasoning patterns)
  - Biology claims (lexicon, concept overlap completely different)
  - Medicine (liability, regulatory implications)
  - Law (precedent-based reasoning fundamentally different)
- **Problem**: Reader might interpret "domain generalization" as cross-field robustness; it's not
- **Recommendation**: Rename to "Cross-Subdomain Robustness" or add explicit caveat: "All domains in CS education; generalization to other fields untested"

**CONCERN 2: No User Study for Educational Impact** ⚠️⚠️⚠️⚠️
- §7.3 "Educational Integration: From Calibration to Pedagogy"
- Completely hypothetical workflow:
  - "High confidence (>0.85): Provide supporting evidence..."
  - "Medium confidence (0.60-0.85): Flag for instructor..."
  - NO EVIDENCE these interventions improve learning outcomes
- This is 300 words of speculation without a single learning study
- **Recommendation**: Either (a) cut this section and move to future work, OR (b) conduct small pilot RCT measuring learning gains. Current framing overstates evidence.

**CONCERN 3: Baseline Calibration Comparisons Incomplete** ⚠️⚠️⚠️
- §5.4 compares to Max-Softmax, Entropy, MC Dropout
- BUT missing modern temperature scaling +"Platt scaling"
- Missing Dirichlet calibration (recent ICLR 2024 approach)
- Why these older methods? "Monte Carlo Dropout +400% latency" dismisses ensemble approach, but modern approximations exist (Mixup, etc.)
- **Recommendation**: Add 1-2 modern baselines or justify omissions

**CONCERN 4: FEVER Retraining Details Missing** ⚠️⚠️
- You retrain FEVER on CSClaimBench training set
- But downstream, you evaluate on CSClaimBench test set
- Without full hyperparameter search log, it's unclear if you found FEVER's best possible accuracy
- Could FEVER achieve 80%+ with more tuning?
- **Recommendation**: Report FEVER hyperparameter search space (learning rate, batch size, epochs, architecture) for reproducibility

**CONCERN 5: Selective Prediction Claims** ⚠️⚠️
- AUC-RC 0.9102 claimed as "excellent" (§5.2)
- But baseline_rag_nli achieves 0.70, Smart Notes 0.71 → only +1 point improvement
- Confidence intervals: [0.8864, 0.9287] wide relative to difference from baselines
- **Interpretation question**: Is AUC-RC 0.9102 considered excellent in UQ literature?
  - In outlier detection: 0.91 = very good
  - In selective prediction: depends on task
- **Recommendation**: Contextualize AUC-RC result by citing how it compares to best-known results in selective prediction papers

**FACTUAL CHECK**: Claimed "5.8× better accuracy transfer"
- Quote from abstract: "5.8× better accuracy transfer"
- Math check: Cross-domain avg 79.7% vs. FEVER 68.5% = 1.16× improvement (NOT 5.8×)
- **This is WRONG** or unit mismatch
- **Recommendation**: CHECK and CORRECT this multiplier claim immediately

### Questions for Authors

1. Why no comparison to uncertainty from model ensembles (e.g., Snapshot Ensembles, SG-MCMC)? These would have similar latency to your component ensemble.

2. Did you perform ablation of calibration VERSUS ablation of multi-sourcing consensus? Would simpler calibration (single signal + temp scaling) achieve comparable ECE?

3. For educational deployment: Have you conducted ANY study with real students using the system? Even a small 30-student pilot would strengthen your pedagogical claims.

4. Conformal prediction comparison (§8.3): You estimate CP achieves ~0.82 AUC-RC. Did you actually implement and measure this? Seems like a guess.

### Minor Issues

- **Language**: "31× better calibration consistency" (§6.4) - what's the unit? ECE is not inversely linear
- **Citation**: "Cambridge standard" (§5.6) - which Cambridge paper exactly?
- **Figure quality**: No figures in appendix D (Synthetic results) explaining why synthetic differs from real. Would help interpretation

### Recommendation
**ACCEPT WITH MAJOR REVISIONS**

**Strengths**:
- ✅ Technically sound ensemble + calibration method
- ✅ Good statistical rigor (bootstrap, CIs)
- ✅ Fast optimization layer impressive engineering

**Weaknesses**:
- ❌ Overstated domain generalization claims (only CS subdomains,not cross-field)
- ❌ Pedagogical framing not empirically supported (no user study)
- ❌ Factual error in literature ("5.8× transfer" multiplier incorrect)
- ⚠️ Baseline comparisons incomplete (missing modern calibration methods)

**Major Revisions Required**:
1. Fix overclaimed "5.8× transfer" statement
2. Clarify "domain generalization" → "CS subdomain robustness"
3. Reframe educational integration as hypothesis (not claim) OR include pilot RCT
4. Add modern calibration baselines OR justify omissions
5. FEVER retraining hyperparameters must be disclosed

**Estimated Revision Time**: 2-3 weeks (major claim revisions + potential additional experiments)

### Verdict: Paper has merit but MUST correct overclaiming. With revisions, acceptable for IEEE Access.

---

## Reviewer 3: Dr. Priya Patel (Reproducibility & Open Science Champion)
**Affiliation**: University of Edinburgh, AI Reproducibility Initiative  
**Expertise**: Determinism, computational reproducibility, open-source standards  
**Review Tone**: Enthusiastic about reproducibility, wants to set standard for field

### Summary
**Excellent reproducibility work.** This paper exemplifies reproducibility standards the field needs more of. Deterministic label outputs verified across 3 GPUs, extensive documentation, open-source release planned. This alone makes the paper valuable for the community. Some technical documentation could be clearer, but overall: strong work.

### Detailed Comments

**STRENGTH 1: Reproducibility Infrastructure** ⭐⭐⭐⭐⭐
- §Appendix D + `docs/REPRODUCIBILITY.md` exemplary
- Deterministic execution verified:
  - 3 trials × 3 GPUs (A100, V100, RTX 4090) = 9 runs
  - ALL produce bit-for-bit identical label predictions
  - This is RARE and valuable
- Scripts provided (`reproduce_all.sh` / `.ps1`)
- Pinned dependencies (`requirements-lock.txt`)
- Configuration management (VerificationConfig, no magic constants)

**STRENGTH 2: Open-Source Commitment** ⭐⭐⭐⭐⭐
- MIT License planned, CC-BY-4.0 data
- GitHub release ready (mentioned)
- Artifact checksumming documented (§Appendix D)
- This is the standard the field should adopt

**STRENGTH 3: Statistical Rigor** ⭐⭐⭐⭐
- Bootstrap methodology documented
- Confidence intervals computed properly
- Cross-domain evaluation reported
- Hardware-specific variance quantified

**TECHNICAL REQUEST 1: Seed Documentation** ⚠️
- Paper states GLOBAL_RANDOM_SEED=42 (good)
- But WHERE is this set? Need to point readers to exact file/line:
  - `src/config/verification_config.py` line ___?
  - `.env` default?
  - `pytest.ini`?
- **Recommendation**: Add 1-line pointer in Appendix D: "Seed set in src/config/verification_config.py line XXX"

**TECHNICAL REQUEST 2: Floating-Point Precision Statement** ⚠️
- Paper says "deterministic outputs verified"
- Question: Does "identical" mean:
  - Exactly identical (bit-for-bit) labels + integer counts?
  - Or identical to N decimal places (e.g., 81.23% ± 0.0001)?
- IEEE/ACM standard: Report "byte-identical" or "numerically equivalent within ε=1e-6"
- **Recommendation**: Add precision statement: "Label predictions identical across all runs; floating-point scalars equivalent to 10 decimal places (ε < 1e-9 in logit space)"

**TECHNICAL REQUEST 3: GPU Determinism Caveat** ⚠️⚠️
- Paper claims cross-GPU determinism
- BUT: cuDNN + PyTorch determinism is environment-specific:
  - NVIDIA driver version matters
  - cuDNN version matters
  - Specific models' nondeterminism (some operations not deterministic on GPU, only CPU)
- **Question**: Have you tested on DIFFERENT driver versions or cuDNN versions?
- **Recommendation**: Add caveat in Appendix D: "Determinism verified on A100/V100/RTX 4090 with cuDNN 8.9.1, PyTorch 2.0.1, NVIDIA driver 535.x. Other configurations untested."

**DOCUMENTATION REQUEST 1: File Structure Clarity** ⚠️
- Appendix C mentions `outputs/paper/ablations/` structure
- But unclear:
  - Is ablation output from `src/evaluation/ablation.py`?
  - Does it overwrite previous runs or append timestamps?
  - What if user runs twice—does data persist or reset?
- **Recommendation**: Add to `docs/REPRODUCIBILITY.md`: "run_history.json timestamps all executions; re-running does NOT overwrite prior results"

**DOCUMENTATION REQUEST 2: Validation/Test Split** ⚠️⚠️
- Most critical: How is train/val/test determined and FIXED across runs?
- Paper states: "Random stratified split maintaining domain and label distribution"
- But is the split deterministic (same split all runs) or re-randomized each run?
- **This matters**: If split is re-randomized, results are not truly reproducible
- **Recommendation**: Clarify in §4.1: "Train/val/test split is deterministic, saved to data/splits/fixed_splits.pkl, and reused across all runs. Random seed controls initial generation; split is frozen thereafter."

**INTERPRETATION ISSUE: Synthetic vs. Real Confusion Resolved** ✓
- **Previous version issue**: Section 5.7 presented synthetic results without clear labeling
- **New version fixed**: Appendix D clearly states "Engineering validation, not primary benchmark"
- This demonstrates good responsiveness to reviewer concerns

### Questions for Authors

1. Does your determinism approach handle variable-length sequences (e.g., evidence documents different lengths)? How do you ensure consistent padding/batching across runs?

2. Have you tested reproducibility with different versions of numpy, scipy? (These can cause floating-point variance)

3. For the artifact submission: Will you provide a containerized version (Docker/Singularity) to ensure reproducibility even if dependencies change?

### Recommendations for Strengthening Reproducibility

1. **Optional but valuable**: Provide Dockerfile / environment.yml for complete reproducibility
2. **Consider**: Upload pre-computed results (`outputs/benchmark_results/experiment_log.json`) to public repo so readers can verify without running
3. **Consider**: Add GitHub Actions CI/CD that re-runs evaluation on every commit (ensures ongoing reproducibility)

### Minor Issues

- Line 2147: "Cross-GPU tested (A100, V100, RTX 4090)" - what driver versions?
- Line 2099: "Determinism verified: 9 independent runs" - over what time period? Days/weeks apart?
- Appendix D: Code snippets use `python src/evaluation/runner.py` but don't show actual file paths relative to repo root

### Recommendation
**ACCEPT** (highest confidence review I have)

**Strengths**:
- ✅ Exemplary reproducibility infrastructure
- ✅ Deterministic verification across hardware platforms
- ✅ Open-source commitment aligned with IEEE standards
- ✅ Statistical rigor and confidence intervals
- ✅ Clear documentation suitable for tutorial

**Minor Weaknesses**:
- ⚠️ Some technical precision statements needed (seeds, floating-point tolerance)
- ⚠️ GPU determinism caveats should be explicit

**Minor Revisions Suggested**:
1. Add seed file pointer (exact line of code)
2. Add precision statement for floating-point equivalence
3. Add GPU environment caveat (cuDNN version, driver)
4. Clarify train/val/test split determinism in §4.1

**Estimated Revision Time**: 30 min (just documentation clarifications)

### Recommendation to Editor
**This paper should be accepted for reproducibility contribution alone**, independent of methodological novelty. It sets standards that other fact verification papers should follow. Consider for "Reproduced Results" badge or special reproducibility track if IEEE Access offers it.

---

## Meta-Analysis: Consensus Across Reviewers

| Criterion | R1 (Calibration Expert) | R2 (ML Skeptic) | R3 (Repro Champion) | **Consensus** |
|-----------|---|---|---|---|
| **Technical Correctness** | ✅ Solid | ✅ Correct | ✅ Strong | **Accept** |
| **Novelty** | ⭐⭐⭐⭐ (good calibration work) | ⭐⭐⭐ (incremental) | ⭐⭐⭐⭐ (reproducibility) | **Moderate novelty** |
| **Clarity** | ⚠️ ECE definition | ✅ Clear methodology | ✅ Excellent docs | **Good** |
| **Experimental Rigor** | ⚠️ Small test set | ⚠️ Unfair baselines? | ✅ Rigorous | **Adequate** |
| **Claims Accuracy** | ✅ Honest | ❌ Overclaimed domains | ✅ Careful | **Fix needed** |
| **Reproducibility** | ✅ Good | ✅ Good | ⭐⭐⭐⭐⭐ Exemplary | **Excellent** |
| **Pedagogical Value** | ⚠️ Untested | ❌ Speculative | ✅ Potential | **Future work** |

### Overall Decision Simulation

| Reviewer | Score | Justification |
|----------|-------|---|
| **R1 (Calibration)** | 8/10 (Accept Minor Rev) | Good calibration work; small test set concern; needs ECE clarity |
| **R2 (ML Skeptic)** | 7/10 (Accept Major Rev) | Sound method but overclaimed domains + pedagogical claims unsupported |
| **R3 (Repro)** | 10/10 (Accept) | Sets reproducibility standard; exemplary open science |
| **Average** | 8.3/10 | **ACCEPT with Revisions** |

**Expected Outcome**: 
- If authors address R2's concerns (fix "5.8×" claim, clarify CS-only domain scope, tone down pedagogy)
- And provide R1 reliability diagrams + ECE clarification + baseline fairness table
- And add R3's precision/seed documentation
- → **Acceptance probability: 85-90%** for IEEE Access

**If NOT revised**: Remains marginal (50-50 depending on associate editor's judgment)

---

## Key Factors for Acceptance

### ✅ Will HELP acceptance:
1. ✅ Reproducibility infrastructure is exemplary (R3 enthusiastic)
2. ✅ Calibration work is rigorous (R1 will cite favorably)
3. ✅ Statistical testing credible
4. ✅ Open-source release planned (IEEE Access values this)
5. ✅ Pedagogical framing novel (even if not empirically tested)
6. ✅ Synthetic/real conflict now resolved

### ⚠️ RISKS to acceptance:
1. ⚠️ Small test set (260 claims) vs. FEVER (19.9K)
2. ⚠️ Domain specificity to CS only (not mentioned as limitation)
3. ⚠️ Pedagogical claims unsupported by user study
4. ⚠️ Baseline fairness not clearly documented
5. ⚠️ Overclaimed multipliers ("5.8×") need correction
6. ⚠️ Missing modern calibration baselines

### 🔴 MUST FIX for acceptance:
1. 🔴 Correct factual error ("5.8× transfer" multiplier)
2. 🔴 Clarify "domain generalization" → "CS subdomain robustness"
3. 🔴 Either add user study pilot OR tone down pedagogical claims to "potential"

---

## Suggested Revision Strategy

**Tier 1 (Required, 1-2 weeks)**:
- [ ] Fix "5.8×" factual error
- [ ] Clarify CS-only domain scope
- [ ] Add FEVER retraining hyperparameter details
- [ ] Tone down pedagogical framing

**Tier 2 (Recommended, 2-3 weeks)**:
- [ ] Add reliability diagram to §5.1
- [ ] Clarify ECE_correctness vs. standard ECE
- [ ] Baseline fairness methodology table
- [ ] Update test set size from 260 to 500+ (if

possible)

**Tier 3 (Nice-to-have, 3-4 weeks)**:
- [ ] Add small RCT pilot (5-10 students, measure learning gains)
- [ ] Include modern calibration baselines
- [ ] Docker containerization for reproducibility

**Realistic Path**: Implement Tier 1 + partial Tier 2, resubmit in 3 weeks → High probability acceptance (80%+)

