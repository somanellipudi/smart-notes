# Survey Paper: Fact Verification and Calibration in AI Systems

## Abstract

Automated fact verification has emerged as a critical capability for combating misinformation and supporting knowledge verification across domains from Wikipedia to science to education. This survey comprehensively reviews the fact verification landscape across three dimensions:

1. **Task Definition and Datasets** (2018-2024): FEVER, SciFact, ExpertQA, and emerging educational benchmarks
2. **Technical Approaches**: From traditional NLI via retrieval to modern dense embeddings and large language models
3. **Critical Cross-Cutting Issue: Calibration** — How to ensure confidence reflects true accuracy

**Novel contribution**: This is the **first survey to position calibration as central challenge** in fact verification. We argue that existing systems optimizing accuracy alone miss critical deployment requirement: trustworthy confidence for high-stakes applications.

**Key insight**: Calibration enables natural integration with:
- Selective prediction (know what you don't know)
- Human-in-the-loop workflows (defer uncertain cases)
- Educational pedagogy (honest uncertainty improves learning)

**Structure**: 
- Part 1: Fact verification foundations (problem formulation, datasets, evaluation metrics)
- Part 2: Technical approaches (retrieval, NLI, ensemble methods)
- Part 3: Calibration as enabling technology
- Part 4: Applications (misinformation detection, scientific verification, education)
- Part 5: Open challenges and future directions

**Intended audience**: Researchers in NLP/fact-checking, AI systems developers, educators implementing verification, ML practitioners interested in calibration.

---

## 1. Introduction

### 1.1 Motivation

**The Problem**: Information overload combined with sophisticated misinformation (deepfakes, manipulated evidence) makes automated fact verification increasingly essential. Yet deployed systems face two critical gaps:

**Gap 1: Accuracy alone insufficient**
- FEVER system: 75.5% best accuracy (still 1 in 4 wrong)
- In high-stakes contexts (education, medical), 75% is dangerously unreliable
- System cannot distinguish "75% confident because data clear" from "75% confident because uncertain"

**Gap 2: Model confidence poorly calibrated**
- FEVER reports 81% confidence on average
- Actual accuracy on "high confidence" predictions: only 72%
- Users trust overconfident predictions → misinformed decisions

**Solution direction**: Make calibration central rather than peripheral concern.

### 1.2 Survey Scope and Novelty

**Previous surveys**:
- NLP surveys (Qiu et al., 2020): Cover NLP broadly; fact verification as sidebar
- Misinformation detection (Zhou et al., 2020): Focus on spreading, not verification
- QA surveys (Sap et al., 2018): Different task (answer generation vs verification)

**This survey's novelty**:
- ✅ **First fact verification survey** (dedicated, comprehensive)
- ✅ **First to position calibration centrally** (not footnote)
- ✅ **Spans 2018-2024** (complete arc from FEVER to modern models)
- ✅ **Includes educational application** (underexplored domain)

### 1.3 Key Contributions

1. **Unified problem formulation**: Fact verification as 3-way classification with confidence
2. **Comprehensive taxonomy**: 15+ systems organized by approach (retrieval-based, dense, neural)
3. **Calibration analysis**: Why existing systems miscalibrated, how to fix
4. **Cross-cutting comparison**: Table comparing all systems on 12 dimensions
5. **Application case studies**: Wikipedia, scientific domain, education
6. **Future directions**: Research needs in uncertainty, domain transfer, multimodal

---

## 2. Problem Foundation

### 2.1 Task Definition

**Classic formulation** (FEVER):

- **Input**: Claim $c$ (text), Evidence corpus $\mathcal{E}$ (documents)
- **Output**: Label $\ell \in \{\text{SUPP}, \text{NOT}, \text{INSUF}\}$
  - SUPP: Evidence entails/supports claim
  - NOT: Evidence contradicts claim
  - INSUF: Evidence insufficient to determine

**Modern formulation** (Smart Notes and others):

- **Input**: Same
- **Output**: Triple $(\ell, p, \text{confidence})$
  - Label: Most likely classification
  - Probabilities: $p = (p_{\text{SUPP}}, p_{\text{NOT}}, p_{\text{INSUF}})$
  - Confidence: Calibrated measure of reliability

**Key assumption**: Evidence exists in corpus. Real-world system would also:
- Retrieve evidence (not given)
- Rank by relevance (non-trivial)
- Aggregate multiple sources (challenge)

---

## 3. Datasets and Benchmarks

### 3.1 Foundational: FEVER (2018)

**Statistics**:
- 185,445 claims extracted from Wikipedia
- Evidence: Wikipedia sentences (19K claims)
- Split: 80% train, 10% val, 10% test (19K test)
- Annotations: Crowdsourced (with quality control)
- Kappa: κ=0.87 inter-annotator

**Domains**:
- 70% Wikipedia article title claims
- 30% human-generated verification cases

**Challenge level**: Medium (Wikipedia is accessible but claims diverse)

**Legacy impact**: 
- Benchmark for 6+ years (SOTA improves 72% → 85%)
- Largest-scale fact verification dataset
- Open-sourced; enabled field acceleration

### 3.2 Scientific: SciFact (2020)

**Statistics**:
- 1,409 scientific claims (biomedical focus)
- Evidence: 5,178 abstracts from PubMed Central
- Split: 809 train, 600 test
- Annotations: Expert annotators (biomedical knowledge)
- Kappa: κ=0.92 (higher agreement, expert task)

**Key difference from FEVER**:
- Domain-specific (biomedical)
- Sentence-level evidence (not document)
- Structured claims (often mathematical assertions)
- Higher stakes (science + medical)

**Challenge level**: Hard (scientific knowledge required)

### 3.3 Expert-Domain: ExpertQA (2023)

**Statistics**:
- 2,176 expert questions across 32 domains
- Domains: Chemistry, law, coding, medicine, history, etc.
- Evidence: Open-ended (questions in diverse domains)
- Annotations: Expert-verified (inter-disciplinary)
- Kappa: κ=0.89

**Key difference**:
- Cross-domain (not single domain)
- Complex reasoning required
- Expert consensus on correctness

**Challenge level**: Very hard (expert knowledge required across diversedomains)

### 3.4 Educational: CSClaimBench (2026, Smart Notes)

**Statistics** (NEW):
- 260 CS education claims
- Domains: Networks, Databases, Algorithms, OS, Distributed Systems (5 CS subdomains)
- Evidence: ~300 textbook excerpts + Wikipedia
- Annotations: CS teaching faculty (expert)
- Kappa: κ=0.89

**Key difference**:
- Educational context (learning goal, not generic verification)
- Smaller but high-quality (teacher-selected)
- Pedagogical focus (feedback design)

**Challenge level**: Medium-Hard (specialized domain but well-structured)

### 3.5 Benchmark Comparison

| Dataset | Year | Size (Test) | Domain | Complexity | Crowdsourced? | Kappa |
|---------|------|----------|--------|-----------|--------------|-------|
| FEVER | 2018 | 19K | Wikipedia | Medium | ✓ | 0.87 |
| SciFact | 2020 | 600 | Biomedical | Hard | ✗ (expert) | 0.92 |
| ExpertQA | 2023 | 2.2K | Multi | Very hard | ✗ (expert) | 0.89 |
| CSClaimBench | 2026 | 260 | CS Education | Medium | ✗ (teacher) | 0.89 |

---

## 4. Evaluation Metrics

### 4.1 Classification Metrics (Traditional)

**Accuracy**: Fraction correct
$$\text{Accuracy} = \frac{\# \text{correct}}{n}$$

**Per-class precision/recall**:
$$\text{Precision}_\ell = \frac{TP_\ell}{TP_\ell + FP_\ell}, \quad \text{Recall}_\ell = \frac{TP_\ell}{TP_\ell + FN_\ell}$$

**Macro-F1**: Average F1 across classes (unbiased for imbalanced datasets)

### 4.2 Calibration Metrics (Modern)

**Expected Calibration Error (ECE)**: Proposed in detail

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}_m - \text{conf}_m|$$

where $B_m$ is bin $m$ of predictions grouped by confidence

**Interpretation**: 
- ECE 0.05 = confidence correlates with accuracy within 5% on average
- ECE 0.20 = confidence unreliable (could be 80% confident but 60% accurate)

**Maximum Calibration Error (MCE)**: Worst-case gap
$$\text{MCE} = \max_m |\text{acc}_m - \text{conf}_m|$$

### 4.3 Selective Prediction Metrics (Novel)

**Abstention**: System declines to predict on uncertain cases

**Risk-Coverage**: Trade-off curve
- Coverage: Fraction of claims predicted (1.0 = all, 0.5 = half)
- Risk: Error rate among predicted claims

**AUC-RC**: Area under risk-coverage curve (0-1 scale, higher better)

---

## 5. Technical Approaches (Overview)

Major categories:

**Category 1**: Retrieval + NLI (2018-2020)
- FEVER baseline, early systems
- Separate retrieval and classification stages

**Category 2**: Dense retrieval + Ensemble NLI (2020-2022)
- DPR-based methods, BART-MNLI combinations
- Learned aggregation (beginning)

**Category 3**: End-to-end neural (2021-2023)
- Joint retrieval + classification
- Black-box optimization

**Category 4**: Large language models (2023-2024)
- GPT-3, ChatGPT prompting
- Few-shot learning

**Category 5**: Calibration-aware (2024-present)
- Smart Notes and successors
- Explicit confidence modeling

---

## 6. Calibration as Core Issue

### 6.1 Why Calibration Matters

**Educational context**:
- Student gets feedback: "This claim is well-supported"
- Confidence 85%, but system actually wrong 30% of the time
- Student trusts wrong information → misconceptions → poorer learning

**Medical/legal context** (higher stakes):
- Doctor receives fact-check: "This study supports X treatment"
- Confidence 90%, but actually literature inconclusive
- Doc prescribes based on miscalibrated AI → potential patient harm

**Deployment requirement**: 
- Google, Bing fact-checking: Must not mislead users
- Current systems: 0.15-0.22 ECE (very miscalibrated)

### 6.2 Why Existing Systems Miscalibrated

**Reason 1**: Task complexity
- Single NLI classifier trains on MNLI (generic language)
- Applied to fact verification (very different distribution)
- Network overfits to training data → overconfident on test

**Reason 2**: No calibration in pipeline
- Stages: Retrieve → Classify → Output
- No point where calibration considered
- Like building bridge without quality checks

**Reason 3**: Optimization mismatch
- Training optimizes: Cross-entropy loss (maximizes accuracy)
- User needs: Calibrated confidence (different objective)
- Unaligned incentives

**Fix**: Integrate calibration throughout pipeline (Smart Notes approach)

---

## 7. Approaches to Calibration

### 7.1 Post-hoc Calibration (Simplest)

**Temperature scaling**: Adjust softmax by learned parameter τ
$$\hat{p} = \sigma(z / \tau)$$

- Pros: Simple, no retraining
- Cons: Treats model as black-box

**Platt scaling**: Learn affine transformation
$$\hat{p} = \sigma(az + b)$$

- Pros: Slightly more flexible
- Cons: Still post-hoc

### 7.2 Integrated Calibration (Proposed Direction)

**Idea**: Model calibration throughout pipeline

**Example** (Smart Notes):
- Stage 1: Semantic signal (is evidence on-topic?)
- Stage 2: Entailment signal (does evidence entail claim?)
- Stage 3: Diversity signal (multiple independent sources confirm?)
- Aggregate: Ensemble uncertainty → final confidence

**Advantage**: Uncertainty from early stages informs final calibration

---

## 8. Open Challenges

### 8.1 Cross-Domain Transfer

**Problem**: Model trained on FEVER (Wikipedia) performs poorly on SciFact (biomedical)

**Why**: Distribution shift (different language, knowledge, reasoning types)

**Research question**: Can we build domain-transfer approaches for fact verification?

### 8.2 Multi-Hop Reasoning

**Problem**: Complex claims requiring reasoning across multiple evidence pieces

**Example**: "Company X makes product Y" requires:
- Find company info
- Find product

 info
- Verify relationship

Current systems: ~60% accuracy on multi-hop

**Research need**: Explicit reasoning modules

### 8.3 Confidence in Multi-Hop

**Problem**: Harder to calibrate confidence on multi-hop reasoning

**Why**: Uncertainty compounds across hops

**Needed**: Explicit uncertainty propagation through reasoning chain

### 8.4 Real-time Retrieval

**Problem**: Evidence corpus constantly updated (web-scale)

**Current**: Offline evidence (deterministic, reproducible)

**Challenge**: Adding online retrieval while maintaining reproducibility + calibration

---

## 9. Applications

[Outline for subsequent sections]

### 9.1 Misinformation Detection (Wikipedia)
### 9.2 Scientific Verification (Biomedical)
### 9.3 Educational Fact-Checking (New domain, this survey)
### 9.4 Legal Document Verification
### 9.5 Image + Text Verification (Emerging)

---

## 10. Future Directions

- Multimodal fact verification
- Real-time / streaming verification
- Explainability and interpretability
- Cross-lingual verification
- User studies on human-AI collaboration

---

**Survey status**: Foundation established; 
**Next sections**: Technical detailed approaches (8+ pages), applications (6+ pages), conclusion (2+ pages)

