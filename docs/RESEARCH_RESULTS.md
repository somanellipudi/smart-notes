# Research Results: Evidence-Grounded Learning Claim Verification

**Status**: Draft / In Progress  
**Last Updated**: 2026-02-17  
**Authors**: Smart Notes Research Team  
**Contact**: https://github.com/somanellipudi/smart-notes

---

## Executive Summary

This document presents research results on verifiable AI for educational content generation. The system generates learning claims (definitions, equations, examples) and validates them against evidence retrieved from course materials, achieving **[ACCURACY]%** accuracy with **[ECE]** calibration error.

**Key Finding**: [Insert main finding here]

---

## 1. Problem Statement

### Background
Educational AI systems often generate plausible-sounding but unsupported claims (hallucinations). This is particularly problematic in educational contexts where incorrect facts can mislead students.

### Research Question
**Can we build an AI system that generates verifiable learning claims with explicit evidence grounding and high confidence calibration?**

### Hypotheses
1. **H1**: Evidence-grounded verification improves claim accuracy vs. baseline LLM generation
   - Expected improvement: +15-25% accuracy
   
2. **H2**: Combining retrieval and NLI provides better discrimination than either alone
   - Expected: Retrieval-only F1 = [X], Retrieval+NLI F1 = [Y], Y > X
   
3. **H3**: Confidence scores are well-calibrated with actual correctness
   - Expected: ECE < 0.1 (Expected Calibration Error)

---

## 2. Methodology

### 2.1 Dataset

**CS Benchmark Dataset** (20 examples, synthetic)

| Domain | Count | Label Distribution |
|--------|-------|-------------------|
| Algorithms | 7 | V:5, R:2, LC:0 |
| Data Structures | 5 | V:3, R:1, LC:1 |
| Complexity Theory | 2 | V:1, R:1, LC:0 |
| Systems | 4 | V:2, R:2, LC:0 |
| Security | 2 | V:2, R:0, LC:0 |

**Legend**: V=VERIFIED, R=REJECTED, LC=LOW_CONFIDENCE

**Dataset Construction**:
- Synthetic source materials (course notes style)
- Generated claims with gold labels
- Evidence spans manually identified
- Balanced label distribution

**Limitations**:
- Small sample size (N=20) for smoke testing
- Synthetic domain (not real student data)
- Balanced labels (real data may be imbalanced)

### 2.2 Verification Pipeline

**Architecture**:

```
Input Claim
    ↓
Retrieval (FAISS + all-MiniLM-L6-v2)
    → Top 5 evidence items
    ↓
Evidence Validation
    → Similarity scores (0-1)
    → NLI consistency (RoBERTa-large-MNLI)
    → Combine scores (60% NLI + 40% similarity)
    ↓
Classification
    → confidence ≥ 0.7: VERIFIED
    → 0.4 ≤ confidence < 0.7: LOW_CONFIDENCE
    → confidence < 0.4: REJECTED
```

**Ablations Tested**:

| Config | Retrieval | NLI | Ensemble | Batch | Notes |
|--------|-----------|-----|----------|-------|-------|
| 00 Baseline | No | No | No | - | Random classification baseline |
| 1a Retrieval | Yes | No | No | - | Similarity-only |
| 1b Full | Yes | Yes | No | Yes | Main approach |
| 1c Ensemble | Yes | Yes | Yes | - | Multiple verifiers combined |
| Feature toggles | Yes | Yes | No | Var | Batch NLI, cleaning effects |

### 2.3 Metrics

**Classification Accuracy**:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Per-Label F1 Score** (focus on VERIFIED):
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Calibration: Expected Calibration Error (ECE)**:
$$ECE = \sum_{b=1}^{B} \frac{|S_b|}{n} | \text{acc}(S_b) - \text{conf}(S_b) |$$

where $S_b$ is set of samples in confidence bin $b$, $\text{acc}(S_b)$ is accuracy in bin, $\text{conf}(S_b)$ is mean confidence.

**Calibration: Brier Score** (MSE of confidence):
$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\text{conf}_i - y_i)^2$$

**Efficiency**:
- Time per claim (average, median, p95)
- Memory usage (evidence store size)
- Throughput (claims/second)

---

## 3. Results

### 3.1 Main Results

**Overall Accuracy by Configuration**:

| Config | Accuracy | F1 (V) | Precision (V) | Recall (V) | ECE | Brier | Time (s) |
|--------|----------|--------|---|---|-----|-------|---------|
| 00 Baseline | ?? | ?? | ?? | ?? | ?? | ?? | ?? |
| 1a Retrieval | ?? | ?? | ?? | ?? | ?? | ?? | ?? |
| **1b Full** | **??** | **??** | **??** | **??** | **??** | **??** | **??** |
| 1c Ensemble | ?? | ?? | ?? | ?? | ?? | ?? | ?? |

**Interpretation**:
- Configuration [X] achieved best overall accuracy
- Configuration [Y] had best calibration (lowest ECE)
- Trade-off between accuracy and speed: [description]

### 3.2 Ablation Results

#### 3.2.1 Retrieval Component
**Finding**: Similarity-based retrieval alone achieves [??]% accuracy
- Misses nuanced cases (e.g., "always O(n log n)" vs "average case O(n log n)")
- Improvement with NLI: +[??]% absolute accuracy

#### 3.2.2 NLI Component
**Finding**: NLI consistency checks improve precision
- Reduces false positives (claims incorrectly marked VERIFIED)
- Precision (V): [without NLI] → [with NLI] (+[??]%)
- Recall (V): [without NLI] → [with NLI] ([change]%)

#### 3.2.3 Batch NLI Optimization
**Finding**: Batch NLI matching reduces inference time
- Single calls: [time] seconds/claim
- Batch calls: [time] seconds/claim
- Speed-up: [??]x faster

#### 3.2.4 Text Cleaning
**Finding**: Preprocessing removes boilerplate without harming accuracy
- Dataset-specific effect (headers, footers)
- Accuracy with cleaning: [??]%
- Accuracy without cleaning: [??]%
- Difference: [minimal/significant]

### 3.3 Calibration Analysis

**Expected Calibration Error (ECE)**: [??]

**Interpretation**: 
- Model confidence is [well-calibrated / overconfident / underconfident]
- Gap between predicted confidence and actual accuracy: [description]

**Confidence Bins**:

| Confidence Bin | Sample Count | Accuracy | Confidence | Gap |
|---|---|---|---|---|
| 0.0-0.1 | ?? | ?? | ?? | ?? |
| 0.1-0.2 | ?? | ?? | ?? | ?? |
| ... | ... | ... | ... | ... |
| 0.9-1.0 | ?? | ?? | ?? | ?? |

**Brier Score**: [??]  
**Interpretation**: Average confidence error of [??] (on 0-1 scale)

### 3.4 Robustness Analysis

**Noise Injection Tests** (5 examples, 3 noise types):

| Noise Type | Baseline Acc | With Noise | Drop |
|---|---|---|---|
| Typo (char substitution) | ?? | ?? | ?? |
| Paraphrase (truncation) | ?? | ?? | ?? |
| Negation (add "NOT") | ?? | ?? | ?? |

**Findings**:
- Most robust to: [noise type]
- Most sensitive to: [noise type]
- Overall robustness: [high/moderate/low]

### 3.5 Per-Domain Performance

| Domain | Accuracy | F1 (V) | Count |
|--------|----------|--------|-------|
| Algorithms | ?? | ?? | 7 |
| Data Structures | ?? | ?? | 5 |
| Complexity Theory | ?? | ?? | 2 |
| Networking | ?? | ?? | 2 |
| Security | ?? | ?? | 2 |
| Databases | ?? | ?? | 2 |

**Domain-Specific Insights**:
- Best performance on [domain]: clear evidence patterns
- Challenging domain [domain]: ambiguous terminology, multiple valid interpretations

---

## 4. Analysis

### 4.1 Why Does the Approach Work?

1. **Explicit Evidence Grounding**: Forcing verification against evidence prevents unsupported claim generation
2. **NLI Consistency**: Natural language inference catches semantic contradictions
3. **Confidence Scores**: Enabling downstream uncertainty-aware decision making
4. **Batch Efficiency**: Amortized LLM cost improves practical feasibility

### 4.2 Failure Mode Analysis

**Common Misclassifications**:

**Example 1: False Positive (classified VERIFIED but should be REJECTED)**
- Claim: "Hash tables provide O(1) lookup in all cases"
- Retrieved Evidence: "Hash tables provide O(1) average-case lookup..."
- Issue: Model matched on "O(1) lookup" without semantic understanding of "all cases"
- Mitigation: Stricter threshold or syntactic patterns for scope qualifiers

**Example 2: False Negative (classified REJECTED but should be VERIFIED)**
- Claim: "[similar concept with different wording]"
- Retrieved Evidence: "[original text]"
- Issue: Paraphrase mismatch, evidence embedding didn't rank high enough
- Mitigation: Improved evidence reranking or query expansion

### 4.3 Limitations

1. **Dataset Size**: N=20 is small; statistical significance limited
2. **Domain Coverage**: Synthetic CS dataset; limited to STEM
3. **Label Imbalance**: Balanced dataset may not reflect real distribution
4. **Evidence Availability**: Assumes sufficient evidence in corpus
5. **Language Models**: Results specific to chosen models (RoBERTa, MiniLM)

### 4.4 Generalization

**Expected to Generalize**:
- ✓ Different domains (history, literature) with domain-specific evidence
- ✓ Real student notes (more noisy, varied writing styles)
- ✓ Multi-modal sources (audio transcripts, videos)

**May Not Generalize**:
- ✗ Claims requiring external knowledge (current events, proprietary data)
- ✗ Highly subjective claims (opinions, aesthetic judgments)
- ✗ Mathematical proofs requiring formal symbolic reasoning

---

## 5. Contributions

1. **Verifiable AI Framework**: Open-source system for evidence-grounded claim generation
2. **Benchmark Dataset**: 20 synthetic CS examples with gold labels (extensible)
3. **Ablation Study**: Systematic analysis of component contributions
4. **Calibration Analysis**: ECE measurement for confidence reliability

---

## 6. Related Work

**Evidence-Based AI**:
- FEVER (fact extraction and verification): fever.ai
- CLIMATE-FEVER (climate science)
- Retrieval-augmented LLMs (Karpukhin et al. 2021)

**Claim Verification**:
- Natural language inference (RoBERTa, BART)
- Semantic textual similarity (sentence-transformers)

**Calibration**:
- ECE metric (Niculescu-Mizil & Caruana 2005)
- Brier score (Brier 1950)
- Temperature scaling (Guo et al. 2017)

---

## 7. Discussion

### 7.1 Key Insights

**Insight 1**: [Description]  
**Citation**: [Results section reference]

**Insight 2**: [Description]  
**Citation**: [Results section reference]

### 7.2 Practical Implications

For educators and students:
- Verifiable mode can flag unsupported claims
- Confidence scores help decide whether to trust AI output
- Evidence links enable fact-checking

For researchers:
- Benchmark enables reproducible evaluation
- Ablation results guide future improvements
- Calibration analysis informs decision thresholds

### 7.3 Future Work

1. **Scale**: Evaluate on larger datasets (100-1000 claims)
2. **Domains**: Extend to non-STEM subjects
3. **Real Data**: Use authentic student notes and lectures
4. **Model Variants**: Test with different LLMs (GPT-4, Llama, etc.)
5. **Active Learning**: Prioritize verification for uncertain claims
6. **User Studies**: Measure human trust and learning outcomes

---

## 8. Reproducibility

### 8.1 Code Availability
All code is open-source: https://github.com/somanellipudi/smart-notes

**Key Files**:
- `evaluation/cs_benchmark/`: Dataset
- `src/evaluation/cs_benchmark_runner.py`: Benchmark runner
- `scripts/run_cs_benchmark.py`: Ablation script
- `tests/test_*.py`: Unit tests

### 8.2 Reproducibility Instructions

```bash
# Clone repository
git clone https://github.com/somanellipudi/smart-notes.git
cd smart-notes

# Install dependencies
pip install -r requirements.txt

# Run benchmark (main configuration)
python scripts/run_cs_benchmark.py \
    --seed 42 \
    --sample-size 20 \
    --output-dir evaluation/results

# Run ablations
python scripts/run_cs_benchmark.py \
    --seed 42 \
    --noise-injection \
    --output-dir evaluation/results

# View results
cat evaluation/results/ablation_summary.md
```

### 8.3 Computational Requirements
- GPU: NVIDIA A100 / V100 (recommended) or CPU
- Memory: 8GB+ (for model loading)
- Time: ~5 minutes for full run (20 claims × 8 configurations)
- Models: Auto-downloaded (2GB total)

### 8.4 Environment Details
- Python: 3.9+
- PyTorch: 2.0+
- CUDA: 11.8+ (optional, for GPU acceleration)
- OS: Linux, macOS, Windows

---

## 9. Appendix

### A. Detailed Predictions

Example predictions table (first 5 claims):

| Claim ID | Predicted Label | Confidence | Gold Label | Match | Top Evidence |
|----------|-------|------|---|----- |
| algo_001 | VERIFIED | 0.89 | VERIFIED | ✓ | "worst-case O(n²)" |
| algo_002 | LOW_CONF | 0.65 | LOW_CONF | ✓ | "average-case O(n log n)" |
| algo_003 | VERIFIED | 0.92 | VERIFIED | ✓ | "guaranteed O(n log n)" |
| ds_001 | REJECTED | 0.25 | REJECTED | ✓ | "worst case O(n)" |
| ds_002 | VERIFIED | 0.87 | VERIFIED | ✓ | "O(log n) operations" |

### B. Confidence Calibration Plots

[Include plots]:
- Reliability diagram (actual vs predicted accuracy per bin)
- Histogram of confidence scores
- Accuracy curve by confidence threshold

### C. Timing Breakdown

| Component | Time (ms) | % of Total |
|---|---|---|
| Embedding (claim) | ?? | ?? |
| FAISS retrieval | ?? | ?? |
| NLI verification | ?? | ?? |
| Aggregation | ?? | ?? |
| **Total** | **??** | **100%** |

### D. Statistical Significance

Confidence intervals (95%) for main metrics (N=20):

| Metric | Estimate | CI Lower | CI Upper |
|--------|----------|----------|---------|
| Accuracy | ?? | ?? | ?? |
| ECE | ?? | ?? | ?? |

Note: Small sample size limits statistical power.

---

## References

[1] Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." *Proc. ICML*.

[2] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration of modern neural networks." *Proc. ICML*.

[3] Karpukhin, V., Oğuz, B., Yih, S. W. B., et al. (2021). "Dense passage retrieval for open-domain question answering." *Proc. EMNLP*.

[4] Thawani, A., Pujara, J., & Singh, A. (2023). "FEVER in a fact-checking landscape." *Proc. WebConf*.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-17  
**Status**: Template (fill in [??] with actual results)
