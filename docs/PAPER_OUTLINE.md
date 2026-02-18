# Paper Outline: Verifiable AI for Educational Content

**Working Title**: *Grounding Educational AI: Evidence-Based Claim Verification and Confidence Calibration*

**Conference Target**: ACL 2026, AAAI 2026, or ICML 2026 (Generative AI + Education tracks)

**Paper Type**: Research article (8-10 pages)

---

## 0. Metadata

- **Authors**: [Team members]
- **Affiliation**: Smart Notes Research Team
- **Contact**: [contact@smartnotes.ai]
- **Code**: https://github.com/somanellipudi/smart-notes
- **Data**: evaluation/cs_benchmark/ (20 synthetic examples)

---

## 1. Title & Abstract

### 1.1 Title Options

1. **Grounding Educational AI: Evidence-Based Claim Verification and Confidence Calibration** (primary)
2. "Verifiable Learning: Reducing Hallucinations in Educational AI Through Evidence-Grounded Verification"
3. "From Hallucinations to Evidence: A Framework for Trustworthy AI-Generated Study Guides"

### 1.2 Abstract (150-200 words)

```
Large language models can generate plausible-sounding but unsupported claims 
when tasked with educational content generation. This paper proposes a verifiable AI 
framework that grounds learning claims in retrieved evidence and validates consistency 
using natural language inference. 

We introduce (1) a benchmark dataset of 20 annotated CS learning claims with gold labels, 
(2) a verification pipeline combining retrieval (FAISS + embedding-based similarity) 
and NLI consistency checks (RoBERTa-large-MNLI), and (3) a calibration analysis measuring 
confidence reliability (ECE < 0.1).

Our ablation study reveals that NLI consistency checking improves F1 by [??]% over 
retrieval-only baseline, while batch processing reduces inference time by [??]x. 
We achieve [??]% accuracy with well-calibrated confidence scores, enabling downstream 
uncertainty-aware applications.

The verifiable mode filters unsupported claims from the final output, reducing hallucination 
risk while maintaining educational value. Code, benchmark, and reproducibility materials 
are open-sourced at github.com/somanellipudi/smart-notes.

Keywords: educational AI, claim verification, hallucination detection, evidence grounding, 
calibration, NLI, retrieval-augmented LLMs
```

---

## 2. Introduction (1-1.5 pages)

### 2.1 Hook
Start with a concrete example:

> "A student using an AI study guide learns that 'Hash tables provide O(1) lookup in all cases.' 
> Later, they fail a test question about hash table worst-case complexity. The AI made a 
> plausible-sounding simplification that is technically incorrect. How can we prevent this?"

### 2.2 Problem Statement

- **AI Hallucination in Education**: LLMs generate unsupported facts (Maynez et al., 2021)
- **Consequences**: Misinformation, confidence in wrong concepts, poor learning outcomes
- **Root Cause**: LLMs optimize for fluency, not truth; no explicit verification mechanism

### 2.3 Related Work (Brief)

**Existing Approaches**:
1. Fact verification (FEVER, CLIMATE-FEVER) - focused on encyclopedic facts, not education
2. Retrieval-augmented LLMs (Karpukhin et al., 2021) - improves grounding but doesn't verify
3. Uncertainty quantification (Kuleshov et al., 2021) - estimates confidence but doesn't correct

**Gap**: No end-to-end framework for educational AI that:
- Generates claims from learning objectives
- Grounds in evidence (course materials, textbooks)
- Validates consistency (semantic, logical)
- Calibrates confidence for decision-making

### 2.4 Contribution Statement

We propose **Smart Notes Verifiable Mode**, an open-source framework addressing this gap:

1. **Verifiable Framework**: Full pipeline from content ingestion → claim generation → verification
2. **Benchmark Dataset**: Annotated CS claims (20 examples, extensible to other domains)
3. **Empirical Analysis**: Ablation study quantifying component contributions
4. **Calibration Study**: Measuring and improving confidence alignment

**Expected Impact**:
- Teachers/students: Deployable system for fact-checking AI-generated content
- Researchers: Benchmark and open-source codebase for reproducibility
- Education AI: Shifts paradigm from fluency-optimized to verification-aware generation

---

## 3. Related Work (1 page)

### 3.1 Fact Verification & Claim Checking
- FEVER, CLIMATE-FEVER: domain-specific benchmarks
- Natural language inference models (RoBERTa, BART)
- Multi-hop reasoning approaches

### 3.2 Retrieval-Augmented Generation
- Dense passage retrieval (DPR, Karpukhin et al. 2021)
- Language models with retrieval (REALM, FiD)
- Comparison: RAG improves grounding but doesn't verify claims

### 3.3 Hallucination in LLMs
- Categorization of hallucination types (Maynez et al., 2021)
- Mitigation strategies: prompting, decoding, fine-tuning
- Confidence estimation: temperature scaling, uncertainty quantification

### 3.4 Calibration & Uncertainty
- Expected Calibration Error (ECE) metric (Niculescu-Mizil & Caruana, 2005)
- Brier score for probabilistic forecasts
- Confidence in NLP: task-specific calibration studies

### 3.5 Educational AI
- LLMs for tutoring: personalization, diagnostic models
- AI-generated explanations: clarity, correctness
- Student misconception detection
- **Gap in literature**: Integrating verification into educational content generation

---

## 4. Method (2 pages)

### 4.1 System Overview

**Architecture Diagram**:
```
Input Content (PDF, audio, URLs, text)
    ↓ [Preprocessing]
Evidence Store (FAISS index + embeddings)
    ↓
Claim Generation (Baseline LLM)
    ↓ [Claim Extractor]
Learning Claims
    ↓ [Verification Pipeline]
Retrieval: Top-k evidence via FAISS
    ↓ (similarity filtering)
NLI Consistency: RoBERTa-large-MNLI
    ↓ (score aggregation)
Confidence Aggregation
    ↓ (thresholding)
Classification: VERIFIED / LOW_CONFIDENCE / REJECTED
    ↓
Filtered Output + Metadata
```

### 4.2 Evidence Store Construction

**Input**: Course materials (PDFs, transcripts, web articles)

**Process**:
1. Text extraction & cleaning (remove boilerplate, OCR fallback)
2. Chunking: fixed-size (200-char) with 50-char overlap
3. Embedding: sentence-transformers all-MiniLM-L6-v2 (384-dim)
4. Indexing: FAISS IndexFlatIP (inner product, deterministic)

**Output**: Searchable evidence store with metadata (source, span, timestamp)

### 4.3 Claim Verification Pipeline

#### 4.3.1 Evidence Retrieval

**Query**: Claim text (embedded with same model)  
**Search**: FAISS top-k=5, threshold τ=0.2 (cosine similarity)  
**Output**: List of EvidenceItem(snippet, similarity, source)

#### 4.3.2 Consistency Validation (NLI)

**Baseline NLI Model**: RoBERTa-large-MNLI (345M parameters)

**Batch Processing Optimization**:
- Group claims by batch size (default: 8)
- Single forward pass instead of individual verification calls
- Reduces per-claim inference time by [??]x

**Individual Verification** (if batch disabled):
```python
result = nli_verifier.verify(claim_text, evidence_snippet)
# Output: {label: ENTAILMENT/CONTRADICTION/NEUTRAL, entailment_prob: 0-1}
```

#### 4.3.3 Confidence Aggregation

Combine similarity and NLI consistency:
$$\text{confidence} = 0.6 \cdot \text{entailment\_prob} + 0.4 \cdot \text{mean\_similarity}$$

**Thresholding**:
- confidence ≥ 0.7 → VERIFIED
- 0.4 ≤ confidence < 0.7 → LOW_CONFIDENCE
- confidence < 0.4 → REJECTED

### 4.4 Batch NLI Verification

**Algorithm**: Process multiple (claim, evidence) pairs together

**Benefits**:
1. Amortization: share attention computation
2. GPU utilization: better batch efficiency
3. Speed: [??]% faster than sequential

**Equivalence Guarantee**: Same per-pair scores as batch-size-1 (validated empirically)

### 4.5 Dataset

**CS Benchmark Dataset** (evaluation/cs_benchmark_dataset.jsonl):

**Schema**:
```json
{
  "doc_id": "algo_001",
  "domain_topic": "algorithms.sorting",
  "source_text": "...",
  "generated_claim": "...",
  "gold_label": "VERIFIED|LOW_CONFIDENCE|REJECTED",
  "evidence_span": "..."
}
```

**Coverage**: 20 claims across 8 CS domains
- **VERIFIED** (11/20): Fully supported by evidence
- **LOW_CONFIDENCE** (4/20): Partially supported or ambiguous
- **REJECTED** (5/20): Contradicted or unsupported

**Domains**: Sorting, data structures, complexity, networking, security, databases, ML, compilers

**Construction Method**: Synthetic (non-copyright) course materials + manually curated claims + gold labels

### 4.6 Evaluation Metrics

#### 4.6.1 Classification Metrics
- **Accuracy**: Overall correctness
- **F1 Score** (per label): Harmonic mean of precision/recall
- **Precision/Recall**: Type-specific performance

#### 4.6.2 Calibration Metrics
- **ECE (Expected Calibration Error)**: Gap between confidence and accuracy across bins
  - Formula: $ECE = \sum_{b=1}^{B} \frac{|S_b|}{n} | \text{acc}(S_b) - \text{conf}(S_b) |$
  - Interpretation: Lower is better; perfect calibration = ECE of 0
  - Goal: ECE < 0.1

- **Brier Score**: Mean squared error of confidence predictions
  - Formula: $\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\text{conf}_i - y_i)^2$
  - Range: [0, 0.25] for binary classification
  - Goal: Minimize

#### 4.6.3 Efficiency Metrics
- **Time per claim**: Average, median, p95
- **Memory usage**: Evidence store size
- **Throughput**: Claims per second

### 4.7 Experiments

#### 4.7.1 Ablation Study

**Configurations**:

| Abbr | Retrieval | NLI | Ensemble | Description |
|------|-----------|-----|----------|---|
| **00** | ✗ | ✗ | ✗ | Baseline: random classification |
| **1a** | ✓ | ✗ | ✗ | Retrieval-only (similarity) |
| **1b** | ✓ | ✓ | ✗ | Full verification (main) |
| **1c** | ✓ | ✓ | ✓ | Ensemble verifier |
| +Features | ✓ | ✓ | ✗ | Toggles: cleaning, batch NLI, etc. |

**Hypothesis Testing**:
- H1: 1b > 00 (evidence grounding helps)
- H2: 1b > 1a (NLI improves over retrieval-only)
- H3: Confidence well-calibrated (ECE < 0.1)

#### 4.7.2 Robustness Testing

**Noise Injection** (5 examples, 3 types):
1. **Typo**: Character substitution
2. **Paraphrase**: Truncation or reordering
3. **Negation**: Add "NOT" to beginning

**Metric**: Accuracy degradation with noisy input

---

## 5. Results (1.5 pages)

### 5.1 Main Results

**Table 1: Main Results**

| Config | Acc | F1(V) | P(V) | R(V) | ECE | Brier | Time (s) |
|--------|-----|-------|------|------|-----|--------|---------|
| 00 Baseline | ?? | ?? | ?? | ?? | ?? | ?? | ?? |
| 1a Retrieval | ?? | ?? | ?? | ?? | ?? | ?? | ?? |
| **1b Full** | **??** | **??** | **??** | **??** | **??** | **??** | **??** |
| 1c Ensemble | ?? | ?? | ?? | ?? | ?? | ?? | ?? |

**Key Findings**:
- Configuration 1b achieves [accuracy]% accuracy
- NLI improves F1 by [??]% absolute (1b vs 1a)
- ECE of [??] indicates [well-calibrated / slightly overconfident]
- Average inference time: [??] seconds per claim

### 5.2 Ablation Results

**Figure: Ablation Study**

[Plot showing accuracy/ECE for each configuration]

**Interpretation**:
- **Retrieval component**: Necessary but insufficient (1a < 1b)
- **NLI consistency**: Adds [??]% accuracy, improves precision
- **Batch optimization**: [??]x speedup with same accuracy
- **Ensemble**: Marginal improvement (+[??]%) at higher cost

### 5.3 Calibration Analysis

**Figure: Reliability Diagram**

[Reliability diagram: confidence bins vs actual accuracy]

**ECE Decomposition**:
- Well-calibrated bins: [0.0-0.3], [0.7-1.0]
- Under-confident bin [0.4-0.5]: [description]
- Over-confident bin [0.9-1.0]: [description]

**Brier Score**: [??] (normalized: [??]%)

**Recommendation**: Confidence scores suitable for downstream decision-making with threshold tuning

### 5.4 Per-Domain Performance

**Table 2: Domain Breakdown**

| Domain | Acc | F1(V) | N |
|--------|-----|-------|---|
| Algorithms | ?? | ?? | 7 |
| Data Structures | ?? | ?? | 5 |
| Complexity | ?? | ?? | 2 |
| ... | ... | ... | ... |

**Observation**: Algorithm claims have higher accuracy (clear evidence patterns)

### 5.5 Robustness

**Table 3: Noise Injection Results**

| Noise | Baseline | Noisy | Drop |
|-------|----------|-------|------|
| Typo | ?? | ?? | ??% |
| Paraphrase | ?? | ?? | ??% |
| Negation | ?? | ?? | ??% |

**Robustness Score**: [??]% (average accuracy under attack)

---

## 6. Discussion

### 6.1 Interpretation

**Why does the approach work?**

1. **Evidence Grounding** (1a > 00): Forcing explicit evidence prevents unfounded claims
2. **Semantic Validation** (1b > 1a): NLI catches nuanced contradictions (e.g., "always" vs "usually")
3. **Batch Efficiency**: Amortized LLM cost enables practical deployment
4. **Well-Calibration**: Confidence scores reflect actual correctness

### 6.2 Failure Modes

**Common Errors**:

1. **False Positives**: Claims match on surface form but differ semantically
   - Example: "O(1) in all cases" vs "O(1) average case"
   - Fix: Stricter thresholds or syntax-aware patterns

2. **False Negatives**: Paraphrased evidence ranked low initially
   - Example: Different synonyms in claim vs evidence
   - Fix: Query expansion or improved reranking

### 6.3 Limitations

1. **Dataset Scale**: N=20 is small (confidence intervals wide)
2. **Synthetic Data**: Real notes may have different characteristics
3. **Model-Specific**: Results depend on chosen embeddings (MiniLM) and NLI (RoBERTa)
4. **Evidence Assumption**: Requires representative course materials
5. **Scope**: Validated on CS; other domains TBD

### 6.4 Generalization

✓ **Should Generalize To**:
- Different CS topics
- Other STEM domains (math, physics, chemistry)
- Humanities (history, literature) with adapted evidence sources

✗ **May Not Generalize To**:
- Claims requiring external knowledge
- Subjective statements (opinions, aesthetics)
- Future-oriented claims (predictions)

### 6.5 Broader Impacts

**Positive**:
- Increases trust in AI tutoring systems
- Reduces misinformation in student learning
- Enables fact-checking workflow

**Risks**:
- Overreliance on automated verification
- Exclusion of unsupported-but-true claims (edge cases)
- Model bias propagation (inherits biases from NLI model)

**Mitigation**: Human-in-the-loop review for sensitive applications; transparent confidence reporting

---

## 7. Future Work

### 7.1 Immediate (1-3 months)

1. **Larger Benchmark**: Extend to 100+ claims across domains
2. **Human Evaluation**: Assess inter-annotator agreement for gold labels
3. **Model Variants**: Test with newer embeddings (E5, BGE) and NLI models (Llama 2)

### 7.2 Medium Term (3-6 months)

1. **Real Data**: Annotate CS100, Calculus101 course materials
2. **Multi-Modal**: Extend to video lectures (transcripts + timing)
3. **Active Learning**: Prioritize verification for uncertain claims

### 7.3 Long Term (6-12 months)

1. **Production Deployment**: Edge-hosted inference for schools
2. **Domain Extensions**: Adapt to history, literature, social sciences
3. **User Studies**: Measure learning outcomes with verified content

---

## 8. Reproducibility Checklist

- [ ] Code released on GitHub (linked)
- [ ] Dataset with license (CC BY-NC-SA)
- [ ] Pre-trained models specified (all open-source)
- [ ] Hyperparameters documented (seed, batch sizes, thresholds)
- [ ] Hardware requirements listed (8GB RAM, optional GPU)
- [ ] Computational cost reported ([X] GPU-hours or CPU-hours)
- [ ] Results easily reproducible? Yes (all deterministic)
- [ ] Failure cases documented? Yes (Section 6.2)

**Reproducibility Statement**:

```bash
git clone https://github.com/somanellipudi/smart-notes.git
python scripts/run_cs_benchmark.py --seed 42 --sample-size 20
cat evaluation/results/ablation_summary.md
```

Expected runtime: ~5 minutes on CPU; resultsmatch Table 1 within [epsilon].

---

## 9. References

[1] Karpukhin, V., Oğuz, B., Yih, S. W. B., et al. (2021). "Dense passage retrieval for open-domain question answering." *Proc. EMNLP*.

[2] Maynez, J., Narayan, S., Hash, B., & Celikyilmaz, A. (2021). "On faithfulness and factuality in abstractive summarization." *Proc. ACL*.

[3] Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." *Proc. ICML*.

[4] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration of modern neural networks." *Proc. ICML*.

[5] Thawani, A., Pujara, J., & Singh, A. (2023). "FEVER in a fact-checking landscape." *Proc. WebConf*.

---

## 10. Appendix

### A. Ablation Configurations (JSON)

```json
{
  "retrieval_only": { "use_retrieval": true, "use_nli": false },
  "full_pipeline": { "use_retrieval": true, "use_nli": true }
}
```

### B. Example Predictions

| Claim | Predicted | Gold | Evidence |
|-------|-----------|------|----------|
| "Merge sort is O(n²)" | REJECTED | REJECTED | "guaranteed O(n log n)" |
| "Hash O(1) always" | REJECTED | REJECTED | "worst case O(n)" |

### C. Dataset Schema

[Link to cs_benchmark_dataset.jsonl]

---

## 11. Submission Checklist

- [ ] Title and abstract finalized
- [ ] All sections complete with [??] replaced by actual values
- [ ] Figures and tables captioned
- [ ] References in BibTeX format
- [ ] Anonymized for review (if needed)
- [ ] Supplementary materials ready (code, data, results)
- [ ] PDF formatted per conference style (ACL, AAAI, ICML)
- [ ] Word count: [X] pages (target: 8-10)

---

**Status**: Template (fill in \[??\] with actual results from RESEARCH_RESULTS.md)  
**Last Updated**: 2026-02-17
