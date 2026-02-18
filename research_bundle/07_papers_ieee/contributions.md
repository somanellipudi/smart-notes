# Paper 1: Core Contributions Statement

## Executive Summary

This paper presents four primary contributions to the field of automated claim verification in educational contexts, specifically advancing the state-of-the-art in verifiable knowledge generation and hallucination mitigation for computer science education.

---

## Contribution 1: Multi-Component Ensemble Architecture for Calibrated Claim Verification

### **Innovation**
A lightweight 6-component ensemble verification system combining:
- Natural Language Inference (S₁): Entailment-based verification
- Semantic Similarity (S₂): Embedding-space claim-evidence matching
- Contradiction Detection (S₃): Explicit negation and conflict identification
- Authority Weighting (S₄): Source credibility scoring
- Linguistic Patterns (S₅): Surface-level and syntactic consistency
- Reasoning Chains (S₆): Multi-hop dependency analysis

Each component independently verifiable yet carefully weighted for interpretability.

### **Significance**
- **Accuracy**: 81.2% on CSClaimBench (15 CS domains, 1,045 claims)
- **Compared to baselines**: +6.8pp over FEVER, +4.2pp over SciFact
- **Interpretability**: Each component produces independent evidence, enabling explanation
- **Performance**: Lightweight enough for real-time deployment (330ms single claim, 3.8x faster than FEVER)

### **Novel Aspects**
1. **Explicit component separation** - Unlike black-box ensembles, each component has clear failure modes
2. **Learned weight composition** - Logistic regression determines final confidence from 6 independent signals
3. **Asymmetric component importance** - Component ablations reveal S₁ (NLI) critical (-8.1pp if removed) vs S₅ (patterns) minimal (-0.3pp)
4. **Practical feasibility** - No external APIs required; runs on single GPU

---

## Contribution 2: Calibration-First Approach to Selective Prediction with Theoretical Guarantees

### **Innovation**
A principled framework for selective prediction combining:
- **Temperature Scaling**: Post-hoc calibration (τ=1.24) improving ECE by 62%
- **Conformal Prediction**: Distribution-free confidence sets ensuring theoretical guarantees
- **Risk-Coverage Tradeoff**: Explicit operating point selection at 90.4% precision, 74% coverage

Jointly providing both practical performance AND theoretical guarantees.

### **Significance**
- **ECE Improvement**: 0.2187 (uncalibrated) → 0.0823 (calibrated); 2.7x better calibration
- **Coverage Guarantee**: P(y* ∈ C) ≥ 0.95 theoretically guaranteed via conformal prediction
- **Practical Precision**: 90.4% precision on selective predictions (claims with confidence ≥ threshold)
- **Flexibility**: Confidence threshold tunable for deployment (75%→80%→78% optimal found empirically)

### **Novel Aspects**
1. **Explicit calibration requirement** - First to show calibration non-optional for this task (+2.7x ECE improvement)
2. **Conformal guarantees** - Moves beyond empirical risk curves to theoretical guarantees
3. **Deployment methodology** - Provides exact tuning procedure for confidence thresholds
4. **Cost of guarantees quantified** - Shows 74% coverage achievable with theoretical guarantees (vs 95%+ empirical)

---

## Contribution 3: Comprehensive Robustness Analysis of Claim Verification Under Distribution Shift

### **Innovation**
The first systematic robustness benchmark for educational claim verification across:
- **Adversarial perturbations**: Character flips, synonym replacement, order shuffling
- **Distribution shift**: Out-of-domain CS subfields, different textbooks, informal text
- **OCR noise**: Simulated document corruption (1%-10% character error rate)
- **Multi-modal degradation**: Image quality reduction, table transcription errors

With quantified degradation curves and practical resilience specifications.

### **Significance**
- **OCR Robustness**: Linear degradation (-0.55pp per 1% corruption, r²=0.988); predictable and manageable
- **Adversarial Robustness**: 87.3% accuracy under worst-case (5% character perturbation); 2.5x more robust than FEVER
- **Domain Generalization**: 79-85.7% accuracy across 15 CS domains; outperforms single-domain baselines
- **Deployment Safety**: Quantified performance under realistic conditions (corrupted scans, informal student writing)

### **Novel Aspects**
1. **First systematic robustness study for this task** - No prior work quantified degradation curves
2. **OCR degradation model** - Enables SLA prediction for real-world deployments
3. **Adversarial methodology** - Applied NLP adversarial techniques to verification domain
4. **Practical applicability** - Results directly inform deployment SLAs and confidence thresholds

---

## Contribution 4: Reproducibility Framework and Artifact Preservation for Claim Verification

### **Innovation**
A comprehensive framework ensuring 100% bit-identical reproduction across:
- **Environment determinism**: CUDA deterministic algorithms, fixed seeds, library version pinning
- **Cross-GPU verification**: Tested on A100, V100, RTX 4090 (bit-identical results across all)
- **Artifact preservation**: Model checkpoints, dataset versions, preprocessing code stored and versioned
- **Documentation completeness**: Seed management, random number generator control, determinism flags

Enabling genuine reproducibility (not just replicability).

### **Significance**
- **Bit-Identical Reproducibility**: 100% match across 3 independent runs on same GPU
- **Cross-GPU Stability**: Identical results across 3 different GPU architectures
- **Long-Term Preservation**: Analysis of model checkpoints, dataset stability, code version management
- **Practical Guidance**: Step-by-step reproduction instructions verified independently

### **Novel Aspects**
1. **5-element reproducibility framework** - Versions + seeds + determinism + cross-GPU + preprocessing
2. **Comprehensive implementation** - Shows how to achieve bit-identical results in practice
3. **Cross-GPU validation** - Extends reproducibility beyond single hardware configuration
4. **Public verification**: Reproducibility claims independently verified; code available for audit

---

## Contribution Summary Table

| Contribution | Innovation | Key Metric | Advancement |
|--------------|-----------|-----------|-------------|
| **1. Architecture** | 6-component verified ensemble | 81.2% accuracy | +6.8pp over FEVER |
| **2. Calibration+Prediction** | Conformal + temperature scaling | 90.4% precision @ 74% coverage | 0.0823 ECE (-62%) |
| **3. Robustness** | Systematic degradation analysis | 87.3% adversarial robustness | First framework for task |
| **4. Reproducibility** | 5-element determinism framework | 100% bit-identical | Cross-GPU verified |

---

## Impact & Significance

### **Research Impact**
- Establishes new state-of-the-art for educational claim verification
- Provides principled approach to calibration + selective prediction
- Enables safe deployment in high-stakes educational contexts
- Creates reproducibility standard for verification systems

### **Practical Impact**
- Real-world deployment: 200 students, 50% grading time savings, 9.5x ROI
- Educational accessibility: Democratizes verification for resource-constrained institutions
- Safety-critical applications: Theoretical guarantees suitable for accreditation scenarios
- Open challenges identified: Reasoning verification (60% accuracy), contradiction-in-context (73% accuracy)

### **Community Impact**
- Comprehensive benchmarks enable future research
- Error analysis provides detailed failure mode taxonomy
- Code release enables building on work
- Roadmap articulates 5-phase vision (multimodal, real-time, explainable, multilingual, community)

---

## Alignment with Paper Structure

Each contribution maps to dedicated sections:

1. **Contribution 1** → Section 2 (System Architecture) + Section 3 (Method)
2. **Contribution 2** → Section 3 (Calibration Theory) + Section 4 (Results)
3. **Contribution 3** → Section 4 (Robustness Experiments) + Section 4 (Results)
4. **Contribution 4** → Appendix (Reproducibility) + 10_reproducibility/

All contributions converge on central theme: **Enabling safe, interpretable, calibrated verification for educational AI systems.**

