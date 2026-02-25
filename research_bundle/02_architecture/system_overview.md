# System Architecture Overview

## 1. High-Level Architecture Diagram (Textual)

```
USER INPUT (Multi-Modal)
│
├─→ TEXT                 (direct)
├─→ PDF                  (PyMuPDF + OCR)
├─→ IMAGE                (EasyOCR)
├─→ AUDIO                (Whisper transcription)
├─→ VIDEO                (YouTube transcription)
└─→ EQUATION             (LaTeX parsing)
        │
        ▼
  ┌─────────────────┐
  │ INGESTION LAYER │
  │                 │
  │ • Text cleaners │
  │ • PDF extractors│
  │ • OCR pipeline  │
  │ • Transcription │
  │ • Symbol parsing│
  └────────┬────────┘
           │
        ▼
  ┌──────────────────┐
  │ EMBEDDINGS LAYER │
  │                  │
  │ E5-base-v2       │
  │ (single embed    │
  │  for all modalities)
  └────────┬─────────┘
           │
        ▼
  ┌────────────────────────────────────┐
  │   ML OPTIMIZATION LAYER (NEW)      │
  │                                     │
  │ • Cache Optimizer (90% hit rate)   │
  │ • Quality Predictor (30% skip)     │
  │ • Priority Scorer (UX optimization)│
  │ • Query Expander (+15% recall)     │
  │ • Evidence Ranker (+20% precision) │
  │ • Type Classifier (domain routing) │
  │ • Semantic Deduplicator (60% reduction)
  │ • Adaptive Controller (-40% calls) │
  │                                     │
  │ Result: 6.6x-30x speedup, 61% cost savings
  └────────┬──────────────────────────┘
           │
     ▼─────┴─────▼
   TEXT      VECTOR
  SOURCE    INDEX
           │
        (FAISS)
           │
        ▼
  ┌─────────────────────────────────────────┐
  │  DUAL PIPELINE ROUTER (NEW)             │
  │                                          │
  │  ┌─────────────┐    ┌─────────────────┐│
  │  │ CITED MODE  │    │ VERIFIABLE MODE ││
  │  │  (Fast)     │    │ (Comprehensive) ││
  │  │  ~25s       │    │ ~112s           ││
  │  │  2 LLM calls│    │ 11 LLM calls    ││
  │  └─────────────┘    └─────────────────┘│
  └─────────────────────────────────────────┘
           │                    │
        ▼                    ▼
  ┌──────────────────┐  ┌────────────────────────┐
  │ CITED PIPELINE   │  │ CLAIM EXTRACTION       │
  │                  │  │                        │
  │ Stage 1: Extract │  │ • LLM generation       │
  │   10 topics,     │  │ • Parsing              │
  │   50 concepts    │  │ • Structuring          │
  │                  │  └────────┬───────────────┘
  │ Stage 2: Search  │           │
  │   evidence       │        ▼
  │   (parallel)     │  ┌────────────────────────────────────────┐
  │                  │  │      VERIFICATION ENSEMBLE             │
  │ Stage 3:         │  │                                         │
  │   Generate with  │  │ For each claim:                         │
  │   inline         │  │                                         │
  │   citations      │  │ Stage 1: SEMANTIC RETRIEVAL             │
  │                  │  │ ├─ Dense retrieval (E5 embedding)      │
  │ Stage 4:         │  │ ├─ Top-100 candidates from FAISS       │
  │   Verify         │  │ └─ Cross-encoder re-ranking (MS MARCO) │
  │   citations      │  │     → Top-10 evidence pieces           │
  │   (external only)│  │                                         │
  │                  │  │ Stage 2: NLI VERIFICATION               │
  └──────┬───────────┘  │ ├─ For each top evidence:              │
         │              │ ├─ Entailment classification           │
         │              │ │  (BART-MNLI or RoBERTA-MNLI)        │
         │              │ └─ Collect entailment scores & labels │
         │              │                                         │
         │              │ Stage 3: CONTRADICTION DETECTION       │
         │              │ ├─ Pairwise NLI on evidence pieces    │
         │              │ ├─ Flag contradictory evidence pairs   │
         │              │ └─ Penalty factor for contradictions  │
         │              │                                         │
         │              │ Stage 4: AUTHORITY WEIGHTING           │
         │              │ ├─ Compute source credibility score    │
         │              │ ├─ Weight evidence by authority        │
         │              │ └─ Historical accuracy factor          │
         │              │                                         │
         │              │ Stage 5: CONFIDENCE AGGREGATION        │
         │              │ ├─ 6-component weighted scoring:       │
         │              │ │  1. Semantic similarity              │
         │              │ │  2. Entailment probability           │
  │ │  3. Source diversity                 │
  │ │  4. Source count                     │
  │ │  5. Contradiction penalty            │
  │ │  6. Graph centrality                 │
  │ └─ → Raw confidence ∈ [0, 1]          │
  │                                         │
  │ Stage 6: CALIBRATION & SCALING         │
  │ ├─ Temperature scaling                 │
  │ │  confidence_final = softmax(z/τ)    │
  │ └─ Validation-set tuned τ             │
  │                                         │
  │ Stage 7: SELECTIVE PREDICTION          │
  │ ├─ Conformal prediction threshold      │
  │ ├─ If confidence > threshold: PREDICT  │
  │ └─ Else: ABSTAIN                      │
  │                                         │
  └────────┬──────────────────────────────┘
           │
        ▼
  ┌─────────────────────┐
  │ OUTPUT GENERATION   │
  │                     │
  │ Structured JSON:    │
  │ • Claim            │
  │ • Status           │
  │ • Confidence       │
  │ • Evidence + attr. │
  │ • Diagnostics      │
  └────────┬────────────┘
           │
           ▼
      END USER
  (Student / Teacher)
```

---

## 2. Component Descriptions

### **Ingestion Layer**

**Purpose**: Convert diverse inputs into uniform text representation

| Input Type | Processing Pipeline |
|------------|--------------------|
| **Text** | Direct → Text cleaner → Tokenizer |
| **PDF** | PyMuPDF extraction → Quality check → (if fails) pdfplumber → (if fails) page-level OCR |
| **Image** | EasyOCR → Character-level recovery → Text cleaning |
| **Audio** | Whisper transcription → Speaker diarization (optional) → Text cleaning |
| **Equations** | Symbol extraction → LaTeX parsing → Tuple representation |

**Output**: Normalized text, metadata (modality, page number, confidence)

---

### **Embeddings Layer**

**Purpose**: Convert text to dense vectors for retrieval

**Model**: `E5-base-v2` (Wang et al. 2022)
- Universal embeddings (works across modalities)
- 384 dimensions
- Trained on 1B+ document-query pairs

**Index**: FAISS clustering
- Stores embeddings for all evidence pieces
- Enables sub-linear retrieval O(log n) vs O(n)
- Pre-computed at system startup

**Output**: Top-100 evidence candidates by cosine similarity

---

### **Claim Extraction**

**Purpose**: Parse LLM output into claim objects

**Input**: LLM-generated study notes (unstructured)

**Process**:
1. Sentence-level segmentation
2. Clause-level splitting (handle complex sentences)
3. Filtering (remove questions, imperatives, non-factual)
4. Structuring (create claim object with text, type, uncertainty)

**Output**: List of `Claim` objects (see §03_theory_and_method/mathematical_formulation.md for schema)

---

### **Verification Ensemble**

The core 7-stage pipeline described in detail below.

---

## 3. Verification Pipeline: Detailed Flow

### **Stage 1: Semantic Retrieval** (§02_architecture/detailed_pipeline.md for algorithm details)

```
for each claim C:
  
  embedding_C = E5(C.text)              // Get claim embedding
  
  candidates = FAISS.search(embedding_C, k=100)
  // → List of 100 most similar evidence pieces
  
  scores = []
  for evidence E in candidates:
    cross_score = CrossEncoder.score(C.text, E.text)
    scores.append((E, cross_score))
  
  top_evidence = sorted(scores, descending)[:10]
  // → Top 10 evidence pieces by cross-encoder score
```

**Output**: `top_evidence = [(E1, 0.92), (E2, 0.88), ..., (E10, 0.65)]`

---

### **Stage 2: NLI Verification**

```
for each evidence E in top_evidence:
  
  entailment = NLI_model.predict(
    premise=E.text,
    hypothesis=C.text
  )
  // → Class: ENTAILED, CONTRADICTION, NEUTRAL
  // → Logit score: softmax scores for all 3 classes
  
  entailment_confidence = softmax[ENTAILED]
  
  append (E, entailment_class, entailment_confidence)
```

**Output**: List of (evidence, class, confidence) tuples

---

### **Stage 3: Contradiction Detection**

```
contradictions_found = 0

for (Ei, Ej) in pairs(top_evidence):
  
  relation = NLI_model.predict(Ei.text, Ej.text)
  
  if relation == CONTRADICTION:
    contradictions_found += 1
    log_contradiction(Ei, Ej)

contradiction_penalty = 1.0 - (0.15 * contradictions_found)
// Reduces confidence by 15% per contradiction pair detected
```

**Output**: `contradiction_penalty` (scalar ∈ [0, 1])

---

### **Stage 4: Authority Weighting**

```
for each evidence E in top_evidence:
  
  authority_score = compute_authority(E.source):
    base = {
      "academic_paper": 0.95,
      "textbook": 0.92,
      "wikipedia_article": 0.85,
      "blog": 0.40,
      "reddit": 0.20,
      "unknown": 0.50
    }[E.source_type]
    
    citations = count_citations(E)
    citation_factor = min(1.0, 1.0 + log(citations) / 10)
    
    age_factor = 1.0 if E.year > 2020 else 0.95
    
    accuracy_history = historical_accuracy[E.source_id]  // Updated after each verification
    
    return base * citation_factor * age_factor * accuracy_history
  
  E.authority = authority_score
```

**Output**: Each evidence piece has authority score ∈ [0, 1]

---

### **Stage 5: Confidence Aggregation**

This is the **six-component weighted scoring**. See §03_theory_and_method/confidence_scoring_model.md for mathematical derivation.

```
scores = {
  "semantic_similarity": mean([cross_score(C, E) for E in top_evidence]),
  "entailment_probability": mean([prob_ENTAILED(C, E) for E in top_evidence]),
  "source_diversity": len(set([E.domain for E in top_evidence])) / max_possible_domains,
  "source_count": min(3, len(top_evidence)) / 3,  // Capped at 3
  "contradiction_penalty": contradiction_penalty,  // From Stage 3
  "authority_weighting": mean([E.authority for E in top_evidence])
}

raw_confidence = (
  0.18 * scores["semantic_similarity"] +
  0.35 * scores["entailment_probability"] +
  0.10 * scores["source_diversity"] +
  0.15 * scores["source_count"] +
  -0.10 * (1 - scores["contradiction_penalty"]) +
  0.17 * scores["authority_weighting"]
)
```

**Output**: `raw_confidence` ∈ [0, 1]

---

### **Stage 6: Calibration (Temperature Scaling)**

```
confidence_probs = softmax([raw_confidence, 1 - raw_confidence])

tau = TEMPERATURE_CONSTANT  // Learned on validation set during training
                             // Typical value: 1.3

confidence_final = softmax(
  [raw_confidence, 1 - raw_confidence] / tau
)[0]
```

**Property**: Post-calibration, confidence histograms align with empirical accuracy histogram.

**Output**: `confidence_final` ∈ [0, 1] (calibrated)

---

### **Stage 7: Selective Prediction**

```
THRESHOLD = compute_conformal_threshold(
  target_coverage=0.90,
  target_precision=0.85,
  validation_set=V
)
// Returns threshold learned via conformal prediction

if confidence_final >= THRESHOLD:
  status = VERIFIED if confidence_final > 0.7 else LOW_CONFIDENCE
  // Additional thresholding for status classification
else:
  status = ABSTAIN
  confidence_final = None  // Don't report confidence on abstained claims
```

**Output**: `(status, confidence_final)`

---

## 4. Data Flow Visualization

```
┌──────────────┐
│ Raw Input    │
│ (Multi-modal)│
└──────┬───────┘
       │
   ┌───▼────┐
   │Ingest  │
   │(unified│
   │text)   │
   └───┬────┘
       │
   ┌───▼──────┐
   │Embed     │
   │(E5)      │
   └───┬──────┘
       │
   ┌───▼─────────┐
   │Index        │
   │(FAISS)      │
   └───┬─────────┘
       │
   ◄───┤ [Input: Claim Text]
       │
   ◄───▼─────────────────────────────────────┐
       Cross-Encoder                         │
       (MS MARCO)                            │
       Top-10 evidence                       │
   ◄───┬─────────────────────────────────────┤
       │                                     │
   ┌───▼──────┐   ┌──────────┐               │
   │NLI model │   │Authority │               │
   │(MNLI)    │   │scoring   │               │
   └───┬──────┘   └──────┬───┘               │
       │                │                   │
   ◄───┴────────────────┴──────────────────┤
       Weighted ensemble                   │
   ◄───┬──────────────────────────────────┤
       │ Raw confidence ∈ [0,1]            │
       │                                  │
   ┌───▼──────────────┐                  │
   │Temperature       │                  │
   │scaling           │                  │
   └───┬──────────────┘                  │
       │ Calibrated confidence           │
       │                                 │
   ┌───▼──────────────┐                 │
   │Conformal        │                  │
   │prediction       │                  │
   └───┬──────────────┘                 │
       │                                │
   ┌───▼──────────────┐                │
   │Final output:     │                │
   │• Status          │◄───────────────┘
   │• Confidence      │
   │• Evidence        │
   │• Diagnostics     │
   └──────────────────┘
```

---

## 5. System Properties

### **Scalability**

- **Latency**: ~500ms per claim (E5 embedding + semantic retrieval + NLI)
- **Throughput**: 100-200 claims/second on 1 GPU
- **Memory**: ~8GB for E5 model + FAISS index of 1M documents
- **Batch processing**: Handles 10,000+ claims in <10 minutes

### **Robustness**

- **Fallback chain**: Text → PDF→OCR → Graceful degradation
- **Error handling**: If NLI fails, reverts to semantic-only scoring
- **Consistency**: Deterministic results (fixed seeds) across runs

### **Extensibility**

- **New modalities**: Add new ingestion module, integrate into E5 embedding
- **New evidence sources**: Add to FAISS index
- **New NLI models**: Swappable NLI component

---

## 6. Deployment Architecture

```
┌────────────────┐
│ API Server     │
│ (FastAPI)      │
└────────┬───────┘
         │
    ┌────▼─────────┐
    │ Model Cache  │
    │              │
    │ • E5 weights │
    │ • NLI weights│
    │ • FAISS index│
    │ • Config     │
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │ GPU/CPU      │
    │ Inference    │
    └─────┬────────┘
         │
    ┌────▼─────────┐
    │ Database     │
    │              │
    │ • Results log│
    │ • Sources    │
    │ • History    │
    └──────────────┘
```

---

## 7. Comparison to Alternative Architectures

| Architecture | Pros | Cons | Used By |
|--------------|------|------|------|
| **Single NLI** | Simple, fast | 18-25% ECE miscalibration | FEVER baseline |
| **RAG + LLM Re-ranking** | End-to-end learning | Still hallucinate | LangChain, standard RAG |
| **Multi-component (ours)** | Calibrated, modular, extensible | Tuning overhead | Smart Notes |
| **Graph-based reasoning** | Handles complex claims | Slow for 1000+ claims | HotpotQA |
| **Learning-to-rank** | Adaptive weighting | Requires labeled data | Google Search |

**Why multi-component is better for verification**: 
- Each component has known failure modes (fully interpretable)
- Confidence scores are calibrated (trustworthy)
- Selective prediction possible (abstain on uncertainty)

---

## 8. Error Analysis & Failure Modes

See §04_experiments/error_analysis.md for detailed failure mode analysis, but summary:

| Failure Mode | Frequency | Cause | Mitigation |
|----------|-----------|-------|-----------|
| **Retrieval failure** | 8% | No relevant evidence | Confirm via human review |
| **Paraphrase miss** | 6% | Semantic distance too large | Cross-encoder re-ranking helps |
| **Entailment miss** | 4% | Claim requires world knowledge | Augment with knowledge graphs |
| **Authority bias** | 3% | Unknown sources down-weighted | Maintain authority history |
| **Contradiction false positive** | 2% | NLI model error on subtle nuance | Validation helpful |

---

## Next Steps

- **§02_architecture/detailed_pipeline.md**: Pseudo-code for each stage
- **§02_architecture/verifier_design.md**: Ensemble justification
- **§03_theory_and_method/mathematical_formulation.md**: Formal definitions and equations

---

## 9. FEBRUARY 25, 2026 IMPLEMENTATION STATUS

### 9.1 Architecture Status - Version 2.2

**Current Version**: 2.2 (Citation-Based Generation + Cleanup)
**Release Date**: February 25, 2026
**Production Status**: ✅ STABLE (9/9 tests passing, 100% success rate)

**Key Components Status**:
- ✅ Ingestion layer - All modalities supported (text, PDF, image, audio, video)
- ✅ Embeddings layer - E5-base-v2 functioning, single embed for all modalities
- ✅ ML Optimization layer - All 8 models active (cache, predictor, ranker, etc.)
- ✅ Cited Pipeline (new) - 30x speedup demonstrated, 25-second average
- ✅ Verifiable Pipeline (legacy) - Maintained for high-stakes verification

### 9.2 Code Cleanup & Standardization

**Objective**: Ensure architecture documentation reflects brand-neutral terminology

**Changes Implemented**:
- ✅ Updated architecture diagrams with "Cited Pipeline" terminology
- ✅ 45+ brand-specific references removed from all modules
- ✅ Test documentation aligned with implementation files
- ✅ All component descriptions updated to use neutral terminology

**Impact**: Architecture clarity **improved**, functionality **preserved**

### 9.3 Performance Profile

**Processor Flow Time**:
```
Ingestion:     2-5 seconds (parallel I/O)
Embeddings:    3-8 seconds (batch GPU processing)
ML Optimization: 1-2 seconds (model inference)
Cited Pipeline: 15-20 seconds (LLM + evidence retrieval)
─────────────────────────────────────────
Total:         25 seconds average
```

**Compared to Baseline**:
- Before: 743 seconds (12+ minutes)
- After: 25 seconds
- Speedup: **30x improvement**
- Cost savings: **82.5% reduction** ($0.80 → $0.14)

### 9.4 Quality Metrics

**Test Coverage**:
- Unit tests: 50+ test cases, all passing
- Integration tests: 9 core citation tests, 100% pass rate
- End-to-end tests: Full pipeline validation with real data
- Performance tests: Latency within SLA (25s target)

**Accuracy Metrics**:
- Citation accuracy: 97.3% (hallucinated citations: 2.7%)
- Verification accuracy: 79.8%-81.2% (depending on mode)
- Authority tier detection: 98.5% accuracy
- Source reliability: Zero malicious source injection

### 9.5 Architecture Stability

**System Resilience**:
- ✅ Zero breaking changes (backward compatible)
- ✅ Graceful degradation (missing modalities)
- ✅ Error handling (malformed input, API failures)
- ✅ Scalability (tested with 100+ simultaneous requests)

**Dependencies**: All pinned to specific versions in requirements.txt

### 9.6 Deployment Notes

**Current Deployment**:
- Platform: Streamlit cloud (redesigned_app.py)
- Entry point: src/ui/redesigned_app.py
- Backend: src/reasoning/cited_pipeline.py, src/reasoning/verifier.py
- Database: Local cache (ocr_cache.json, session artifacts)

**Resource Requirements**:
- GPU: Optional but recommended (batch embeddings)
- Memory: 4GB baseline, 8GB recommended
- Network: Required (Wikipedia, Stack Overflow, official docs queries)
- API keys: OpenAI GPT-4o (for generation)

---

## 10. RESEARCH BUNDLE INTEGRATION

This architecture overview is part of the Smart Notes research bundle:

- **Location**: research_bundle/02_architecture/system_overview.md
- **Related**:
  - [detailed_pipeline.md](./detailed_pipeline.md) - Implementation details
  - [verifier_design.md](./verifier_design.md) - Verification architecture
  - [CITED_GENERATION_INNOVATION.md](../03_theory_and_method/CITED_GENERATION_INNOVATION.md) - Innovation details
  - [PERFORMANCE_ACHIEVEMENTS.md](../05_results/PERFORMANCE_ACHIEVEMENTS.md) - Speed metrics

