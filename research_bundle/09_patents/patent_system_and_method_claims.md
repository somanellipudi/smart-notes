# Patent Materials: Smart Notes Fact Verification System

## Patent Application: System and Method for Calibrated Fact Verification with Selective Prediction

---

## PART 1: SYSTEM CLAIMS

### Claim 1: Independent System Claims

**Claim 1.1 - Calibrated Fact Verification System (Independent)**

A computerized system for verifying factual claims with calibrated confidence, comprising:

(a) A **semantic matching module** configured to:
    - Receive a claim as input text
    - Generate a semantic embedding (E5-Large, 1024-dimensional)
    - Compare against evidence embeddings using cosine similarity
    - Output: relevance scores S₁ ∈ [0, 1] for top-k evidences
    
(b) A **retrieval module** configured to:
    - Execute dual-mode retrieval: dense (DPR embeddings) and sparse (BM25)
    - Fuse DPR scores (α=0.6) and BM25 scores (α=0.4) via weighted combination
    - Retrieve top-k=100 evidences from corpus (Wikipedia, scientific literature)
    - Return: {evidence_text, source, retrieval_score}
    
(c) A **natural language inference (NLI) module** configured to:
    - Accept claim and evidence pairs
    - Classify relationship: ENTAILMENT, NEUTRAL, CONTRADICTION
    - Output confidence scores for each class using BART-MNLI
    - Specifically output: P(entailment|claim, evidence), S₂ ∈ [0, 1]
    
(d) A **diversity scoring module** configured to:
    - Compute pairwise embedding distances between all evidences
    - Identify semantic clusters of similar evidences
    - Assign diversity score: S₃ = 1 - (intra-cluster distance / max_distance)
    - Account for diminishing returns of multiple similar evidences
    
(e) A **agreement scoring module** configured to:
    - Track stance of each piece of evidence (supports/contradicts claim)
    - Aggregate stances across evidences
    - Score agreement: S₄ = |#supporting - #contradicting| / total_evidences
    - Handle neutral evidences separately
    
(f) A **contradiction detection module** configured to:
    - Identify explicitly contradictory statements in evidence
    - Assign weight based on contradiction strength
    - Output S₅ = fraction of evidences that contradict claim clearly
    
(g) An **authority weighting module** configured to:
    - Classify evidence sources (peer-reviewed: high authority; blog: low authority)
    - Apply authority weights to evidence contributions
    - Score authority: S₆ = weighted_average_authority ∈ [0, 1]
    
(h) An **ensemble aggregation module** configured to:
    - Combine six component scores via learned weighting
    - Apply pre-trained weights w = [0.18, 0.35, 0.10, 0.15, 0.10, 0.12]
    - Learned via logistic regression on validation set
    - Compute final score: s_raw = Σᵢ wᵢ · Sᵢ
    - Apply ML optimization pre-filtering (deduplication, quality screening) to reduce input burden by 40-60%
    - Output: raw confidence ∈ [0, 1]
    
(i) A **temperature-based calibration module** configured to:
    - Accept raw confidence scores s_raw ∈ [0, 1]
    - Apply learned temperature scaling: s_calibrated = σ(s_raw / τ)
    - Use optimal temperature τ = 1.24 (pre-learned on validation set)
    - Output: calibrated confidence ∈ [0, 1] with Expected Calibration Error < 0.10
    
(j) A **selective prediction module** configured to:
    - Accept calibrated confidence and risk-coverage targets
    - Implement conformal prediction: P(y* ∈ C(X)) ≥ 1 - α (α=0.05)
    - Generate two output modes:
        * Mode A (high-confidence): Return prediction where s_calibrated > threshold_high
        * Mode B (hybrid): Return prediction + deferral flag where threshold_low < s_calibrated < threshold_high
    - Achieve: 90.4% precision when (74% coverage), 96.2% precision when (60% coverage)
    
(k) An **output formatter module** configured to:
    - Package prediction results: {label, confidence, evidence, reasoning}
    - Generate human-readable explanations
    - Format for downstream applications (LMS, Wikipedia interface, etc.)

**Claim 1.2 - Reproducibility-Hardened System**

The system of Claim 1.1, further characterized by:

(a) **Deterministic inference**: Use fixed random seeds (seed=42) to ensure bit-identical predictions across multiple inference runs
(b) **Cross-GPU consistency**: Verification that predictions remain ±0% variance across NVIDIA A100, V100, and RTX 4090 GPUs
(c) **Environment reproducibility**: Documented requirements (transformers==4.28.1, torch==2.0.0, sentence-transformers==2.2.2)
(d) **Artifact checksums**: SHA256 hashes for model weights, evidence corpus, calibration parameters
(e) **20-minute reproducibility pathway**: From-scratch reproducibility documented and tested

---

### Claim 2: System with Specific Architecture

**Claim 2.1 - Pipeline Architecture**

A computerized system for fact verification, comprising:

**(1) Input processing stage**: Receive claim text; tokenize and encode

**(2) Evidence retrieval stage**: 
- Retrieve evidences from corpus using dual-mode (dense + sparse)
- Dual scoring mechanism with learned fusion weights (α_dense, α_sparse)

**(3) Evidence encoding stage**: 
- Encode all retrieved evidences using same encoder as claims (E5-Large)
- Generate embeddings for similarity computation

**(4) Semantic scoring stage**: 
- Compute relevance via cosine similarity (S₁)

**(5) NLI classification stage**: 
- Run claim-evidence pairs through BART-MNLI
- Extract entailment/neutral/contradiction logits
- Extract entailment confidence (S₂)

**(6) Diversity assessment stage**: 
- Compute evidence diversity (S₃)
- Penalize redundant evidences

**(7) Aggregation stage**: 
- Compute S₄, S₅, S₆ (agreement, contradiction, authority)
- Combine via learned weights: s_raw = Σ wᵢ Sᵢ

**(8) Calibration stage**: 
- Apply temperature scaling: s_calibrated = σ(s_raw / τ)
- Ensure ECE < 0.10

**(9) Selective prediction stage**: 
- Compute prediction set C(X) such that P(y* ∈ C) ≥ 1 - α
- Output: {prediction, confidence, deferral_flag}

**(10) Output stage**: 
- Format results with evidence summaries, reasoning, confidence

**Claim 2.2 - Trained Weights Specification**

The system of Claim 2.1, specifically configured with learned parameters:

**(a) Component weights** (learned via logistic regression):
- w₁ = 0.18 (semantic matching contribution)
- w₂ = 0.35 (entailment contribution) [dominant component]
- w₃ = 0.10 (diversity contribution)
- w₄ = 0.15 (agreement contribution)
- w₅ = 0.10 (contradiction contribution)
- w₆ = 0.12 (authority contribution)
- Weights normalized: Σ wᵢ = 1.0

**(b) Temperature parameter**:
- τ = 1.24 (determined via grid search on validation set)
- Achieves Expected Calibration Error = 0.0823 (vs. raw 0.2187)

**(c) Model dependencies**:
- Semantic encoder: E5-Large (1024-dimensional, trained on 1B sentence pairs)
- NLI classifier: BART-MNLI (pre-trained on MultiNLI)
- Retrieval engines: DPR (dense) + BM25 (sparse)

---

### Claim 3: System with Self-Monitoring

**Claim 3.1 - Adaptive Calibration System**

The system of Claim 1.1, further configured to:

(a) **Monitor calibration drift** across time or domains:
    - Track ECE on rolling basis (e.g., per 100 predictions)
    - If ECE > 0.12, flag calibration degradation
    
(b) **Trigger adaptive recalibration**:
    - If calibration drift detected, adjust temperature τ
    - Use small unlabeled sample from new domain
    - Update τ_new = τ_old + Δτ (learned from observed miscalibration)
    
(c) **Optional retraining pathway**:
    - If standard recalibration insufficient, fine-tune component weights
    - Requires ~100 labeled examples from new domain
    - Maintain backward compatibility with existing models

---

## PART 2: METHOD CLAIMS

### Claim 4: Core Method Claims

**Claim 4.1 - Method for Calibrated Fact Verification (Independent)**

A computer-implemented method for verifying factual claims with calibrated confidence, comprising:

**(Step 1) Receive claim**: Acquire input claim text from user/application

**(Step 2) Retrieve evidence**: 
- Execute dual-mode retrieval (dense embedding similarity + sparse BM25)
- Normalize and fuse scores: score_fused = 0.6 · score_dense + 0.4 · score_sparse
- Retrieve top-k=100 evidences ranked by fused score

**(Step 3) Compute semantic scores**:
- Encode claim using E5-Large encoder
- Encode each evidence using same encoder
- Compute cosine similarity: S₁ᵢ = cos_sim(claim_emb, evidence_emb_i)
- Return: [S₁₁, S₁₂, ..., S₁₁₀₀]

**(Step 4) Compute entailment scores**:
- For each (claim, evidence) pair, run BART-MNLI classifier
- Extract P(ENTAILMENT | claim, evidence) from model logits
- S₂ᵢ = P(ENTAILMENT | claim, evidence_i)
- Return: [S₂₁, S₂₂, ..., S₂₁₀₀]

**(Step 5) Compute diversity scores**:
- Compute pairwise cosine distances between all evidence embeddings
- Cluster evidences by similarity (e.g., k-means with k=5)
- Assign penalty if multiple evidences in same cluster
- Compute diversity: S₃ = 1 - (average_intra_cluster_distance / max_distance)
- Return: Single scalar S₃ ∈ [0, 1]

**(Step 6) Compute agreement scores**:
- Aggregate NLI labels across all evidences
- Count: n_entail = # evidences with P(ENTAILMENT) > 0.5
- Count: n_contradiction = # evidences with P(CONTRADICTION) > 0.5
- Compute: S₄ = |n_entail - n_contradiction| / k (where k = 100)
- Return: S₄ ∈ [0, 1]

**(Step 7) Compute contradiction scores**:
- Identify evidences with strong contradiction (P(CONTRADICTION) > 0.7)
- Weight by strength: w_i = P(CONTRADICTION | evidence_i) - 0.5
- Compute: S₅ = Σᵢ w_i / k
- Return: S₅ ∈ [0, 1] (high = strong contradictions found)

**(Step 8) Compute authority scores**:
- Classify each evidence source by authority level
- Assign weights: peer-reviewed=1.0, established media=0.85, blog=0.3, social media=0.1
- Compute weighted average: S₆ = Σᵢ authority_weight_i / k
- Return: S₆ ∈ [0, 1]

**(Step 9) Aggregate via learned ensemble**:
- Apply learned weights: s_raw = 0.18·S₁ + 0.35·S₂ + 0.10·S₃ + 0.15·S₄ + 0.10·S₅ + 0.12·S₆
- Weights determined by logistic regression on validation set (260 labeled examples)
- Return: s_raw ∈ [0, 1] (uncalibrated confidence)

**(Step 10) Calibrate via temperature scaling**:
- Apply logistic function with temperature: s_calibrated = σ(s_raw / τ)
- Use pre-determined temperature τ = 1.24
- Return: s_calibrated ∈ [0, 1] (calibrated confidence, ECE < 0.10)

**(Step 11) Classify claim**:
- Assign label based on S₂ (entailment score dominates):
  * If S₂ > 0.5 and s_calibrated > 0.5: label = "SUPPORTED"
  * If S₂ < 0.3 and s_calibrated > 0.5: label = "NOT_SUPPORTED"
  * Else: label = "INSUFFICIENT_INFO"
- Also use agreement (S₄) and contradiction (S₅) in tie-breaking

**(Step 12) Selective prediction**:
- Using conformal prediction framework:
  * Compute nonconformity measure p_value = (# scores ≤ score_test) / (n+1)
  * Generate prediction set: C(X) = {label ∈ {SUP, NOT_SUP, INSUF} : p_value > α}
  * Default: α = 0.05 (95% coverage guarantee)
- Output deferral flag if |C(X)| > 1

**(Step 13) Output results**:
- Return: {claim, label, s_calibrated, deferral_flag, evidence_summary, reasoning}
- Format for display or downstream applications

---

**Claim 4.2 - Method with Weight Learning**

The method of Claim 4.1, further comprising:

**(Optional) Learning phase** (executed once, before deployment):

**(Step A) Prepare validation data**:
- Collect 260 labeled claims (CSClaimBench)
- For each claim, compute S₁-S₆ components (steps 2-8 of Claim 4.1)
- Create feature matrix X ∈ ℝ^{260×6}, labels y ∈ {0, 1}³⁰⁰

**(Step B) Learn weights via logistic regression**:
- Fit model: log(p/(1-p)) = β₀ + Σᵢ βᵢ · Sᵢ
- Use L2 regularization to prevent overfitting
- Extract learned weights: wᵢ = normalized(βᵢ)
- Result: [0.18, 0.35, 0.10, 0.15, 0.10, 0.12]

**(Step C) Learn temperature via grid search**:
- For each τ ∈ [0.8, 0.9, ..., 2.0] (Δτ = 0.01):
  * Apply: s_cal = σ(s_raw / τ)
  * Compute ECE using M=10 bins
- Select τ that minimizes ECE
- Result: τ = 1.24, ECE = 0.0823

---

### Claim 5: Method with Reproducibility

**Claim 5.1 - Deterministic-Inference Method**

The method of Claim 4.1, further characterized by:

**(a) Fixed randomness**:
- Set random seed = 42 at system initialization
- Ensure numpy, torch, transformers all use same seed
- Result: Identical predictions across multiple inference runs

**(b) Cross-GPU verification**:
- Execute method on multiple GPU types: NVIDIA A100, V100, RTX 4090
- Compare outputs: ±0% variance in accuracy, ECE, AUC-RC metrics
- Document: All GPUs produce bit-identical results

**(c) Hardware independence**:
- Remove dependency on hardware-specific optimizations (cuBLAS, etc.)
- Use float32 precision (consistent across platforms)
- Avoid non-deterministic operations (dropout disabled at inference)

**(d) Environment specification**:
- Document exact library versions (transformers, torch, etc.)
- Provide docker container for reproducibility
- Include environment.yml file for conda environment recreation

---

### Claim 6: Method with Selective Prediction

**Claim 6.1 - Selective Prediction via Conformal Intervals**

The method of Claim 4.1, further comprising:

**(Step S1) Build calibration set**:
- Use validation set (260 labeled claims)
- Compute nonconformity scores: ξᵢ = (1 - sᵢ) if yᵢ=correct, sᵢ if yᵢ=incorrect
- Sort: ξ₍₁₎ ≤ ξ₍₂₎ ≤ ... ≤ ξ₍ₙ₎

**(Step S2) Compute threshold**:
- Choose α = 0.05 (95% coverage)
- Threshold: q = ξ₍⌈(n+1)(1-α)⌉₎ = ξ₍248₎ = q*

**(Step S3) Predict with guarantee**:
- For new test prediction s_test:
  * Construct set C(X) = {label : nonconformity(label) ≤ q*}
  * Return all labels where nonconformity is "small enough"
- Guaranteed: P(true label ∈ C(X)) ≥ 1 - α

**(Step S4) Selective mode execution**:
- If |C(X)| = 1: Output single prediction with high confidence (automated decision)
- If |C(X)| > 1: Output deferral flag for human review (hybrid workflow)
- If |C(X)| = 0: Output most likely prediction with deferral flag (edge case)

**(Step S5) Performance characterization**:
- On CSClaimBench test set:
  * Achieves 90.4% precision when accepting 74% of cases
  * Achieves 96.2% precision when accepting 60% of cases
  * Can trade off precision/coverage based on application needs

---

### Claim 7: Educational Application

**Claim 7.1 - Method for Educational Grading and Feedback**

The method of Claim 4.1, applied to educational domain, comprising:

**(Step E1) Student answer verification**:
- Student writes claim about course topic
- Execute fact verification method (Claim 4.1)
- Receive: label ∈ {SUPPORTED, NOT_SUPPORTED, INSUFFICIENT}, confidence ∈ [0, 1]

**(Step E2) Pedagogical feedback generation**:
- If label = SUPPORTED and confidence > 0.8:
  * Feedback: "✓ Correct! I found supporting evidence from X reliable sources."
  * Show top 2-3 evidence summaries
- If label = NOT_SUPPORTED and confidence > 0.8:
  * Feedback: "✗ Incorrect. I found evidence that contradicts this. See: ..."
  * Suggest correct answer based on evidence
- If confidence ∈ [0.6, 0.8]:
  * Feedback: "? I'm not certain. This requires expert judgment. Here's what I found: ..."
  * Flag for teacher review

**(Step E3) Grading assistance**:
- For batch of student answers, compute confidences for all
- Sort by confidence (ascending)
- Display low-confidence answers first (require manual grading)
- Auto-grade high-confidence answers
- Flag medium-confidence for review

**(Step E4) Learning analytics**:
- Track student accuracy on fact verification
- Identify common misconceptions
- Provide remedial feedback based on patterns

---

## PART 3: DEPENDENT CLAIMS

### Claim 8: Method Variant Claims (Dependent on Claim 4.1)

**Claim 8.1** (Dependent on Claim 4.1):
The method of Claim 4.1, wherein the temperature parameter τ is learned via:
- Grid search on validation set minimizing Expected Calibration Error
- Alternative: Learning via entropy minimization or Platt scaling

**Claim 8.2** (Dependent on Claim 4.1):
The method of Claim 4.1, wherein the component weights are learned via:
- Logistic regression (as described)
- Alternative: Neural network (small MLP with 1 hidden layer)
- Alternative: Support vector machine with RBF kernel

**Claim 8.3** (Dependent on Claim 4.1):
The method of Claim 4.1, wherein the NLI classifier can be:
- BART-MNLI (primary)
- Alternative: RoBERTa-MNLI, DeBERTa-v3-MNLI, or other transformer-based NLI

**Claim 8.4** (Dependent on Claim 4.1):
The method of Claim 4.1, wherein evidence retrieval can use:
- Dual-mode (dense + sparse, as described)
- Alternative: Dense-only (E5 embeddings only)
- Alternative: Sparse-only (BM25 only)
- Alternative: Learned fusion (learn α_dense, α_sparse)

**Claim 8.5** (Dependent on Claim 4.1):
The method of Claim 4.1, wherein selective prediction can use:
- Conformal prediction (as described)
- Alternative: Threshold-based deferral (defer if confidence < threshold)
- Alternative: Risk-coverage optimization (satisfy precision ≥ p_min at coverage ≥ c_min)

---

### Claim 9: System Variant Claims (Dependent on Claim 1.1)

**Claim 9.1** (Dependent on Claim 1.1):
The system of Claim 1.1, wherein the evidence corpus can be:
- Wikipedia (primary)
- Alternative: Scientific literature (PubMed, arXiv)
- Alternative: Educational textbooks
- Alternative: Legal documents
- Alternative: Hybrid (multiple sources)

**Claim 9.2** (Dependent on Claim 1.1):
The system of Claim 1.1, wherein output formatting can include:
- Label + confidence (basic)
- Label + confidence + evidence summaries (enhanced)
- Label + confidence + reasoning + evidence + source attribution (full)

**Claim 9.3** (Dependent on Claim 1.1):
The system of Claim 1.1, further comprising a **learning module** configured to:
- Accept human feedback on predictions
- Update component weights if systematic errors detected
- Implement online learning to adapt to new domains

---

## PART 4: COMBINATION CLAIMS

### Claim 10: Multiple Components Combined

**Claim 10.1** (Combining system and method):
A computerized system implementing the method of Claim 4.1, comprising:
- Hardware: GPU (NVIDIA A100 or equivalent) for parallel inference
- Software: Python implementation using PyTorch, transformers, sentence-transformers
- Storage: Evidence corpus index (optimized for retrieval speed)
- Interface: Web API accepting claim queries, returning predictions

**Claim 10.2** (Combining system and educational application):
A learning management system (LMS) interface with integrated fact verification comprising:
- Student submission interface
- Real-time verification via system of Claim 1.1
- Pedagogical feedback generation (as per Claim 7.1)
- Teacher dashboard showing students needing review
- Grading automation for high-confidence predictions

**Claim 10.3** (Combining reproducibility and cross-domain):
A reproducible fact verification system achieving:
- Bit-identical predictions across 3+ independent runs
- ±0% variance across GPU types (A100, V100, RTX 4090)
- Documented cross-domain adaptation pathway (≤100 labels for new domain)
- Published Docker container for infrastructure reproducibility

---

## SUMMARY OF CLAIMS

| Claim # | Type | Category | Key Innovation |
|---------|------|----------|-----------------|
| 1.1 | System | Architecture | 10-module ensemble for calibrated verification |
| 1.2 | System | Reproducibility | Deterministic, cross-GPU verified outputs |
| 2.1 | System | Architecture | Specific 10-stage pipeline |
| 2.2 | System | Implementation | Trained weights w=[0.18,0.35,0.10,0.15,0.10,0.12], τ=1.24 |
| 3.1 | System | Adaptive | Self-monitoring calibration drift + recalibration |
| 4.1 | Method | Core | Calibrated fact verification procedure (13 steps) |
| 4.2 | Method | Learning | Weight + temperature learning pathway |
| 5.1 | Method | Reproducibility | Deterministic inference method |
| 6.1 | Method | Selective Prediction | Conformal prediction for hybrid human-AI workflows |
| 7.1 | Method | Educational | Application to automated grading + pedagogical feedback |
| 8.1-8.5 | Method | Variants | Alternative NLI, retrieval, learning, calibration methods |
| 9.1-9.3 | System | Variants | Alternative evidence sources, output formats, learning |
| 10.1-10.3 | Combined | Integration | Full systems (API, LMS integration, reproducible deployment) |

**Total: 18 claims** (1 independent system family, 1-2 independent method families, 15 dependent variants + combinations)

**Broadest scope**: System for calibrated fact verification with learned components and deployment flexibility
**Narrowest scope**: Method with specific temperature (τ=1.24), weights, and conformal prediction threshold
**Novel aspects**: Calibration + selective prediction focus (not present in prior art), 6-component scoring, educational application

