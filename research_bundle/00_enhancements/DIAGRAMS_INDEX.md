# Diagram and Architecture Index

**Purpose**: Quick reference for all system diagrams and flowcharts
**Last Updated**: February 2026
**Status**: All diagrams updated to reflect ML optimization layer and dual pipeline architecture

---

## Overview Diagrams

### 1. System Architecture Overview (High-Level)
**File**: [02_architecture/system_overview.md](02_architecture/system_overview.md) (Section 1)
**Diagram**: Multi-modal ingestion → ML optimization layer → Dual pipeline router → Dual modes (Cited/Verifiable) → Output

**Key Features**:
- ✅ Shows ML optimization layer (8 models)
- ✅ Dual pipeline router (Cited ~25s vs. Verifiable ~112s)
- ✅ Cited pipeline stages 1-4
- ✅ Verifiable pipeline stages 1-7
- ✅ Quality diagnostics and recommendation feedback

**Use Case**: High-level overview for executives/researchers

---

### 2. Detailed Pipeline Architecture
**File**: [02_architecture/detailed_pipeline.md](02_architecture/detailed_pipeline.md) (Executive Summary)
**Diagram**: Input Claim → Stage 1-7 → Output with ML optimization layer overlay

**Key Features**:
- ✅ ML optimization layer (8 models) as pre-processing stage
- ✅ 7-stage verification pipeline (semantic, retrieval, NLI, diversity, aggregation, calibration, selective prediction)
- ✅ Component scoring with weights [0.18, 0.35, 0.10, 0.15, 0.10, 0.12]
- ✅ Temperature scaling (τ=1.24)
- ✅ Selective prediction threshold

**Use Case**: Technical reference for engineers

---

### 3. Ingestion Pipeline
**File**: [02_architecture/ingestion_pipeline.md](02_architecture/ingestion_pipeline.md) (Executive Summary)
**Diagram**: 7 input types → Preprocessing → Canonical representation → ML optimization → Dual pipeline router

**Key Features**:
- ✅ Text, PDF, OCR, Speech-to-text, Audio, Handwritten, Semi-structured inputs
- ✅ ML optimization layer before dual router
- ✅ Dual pipeline routing (Cited vs. Verifiable)
- ✅ Output quality metrics

**Use Case**: User-facing documentation, deployment guides

---

## Performance Diagrams

### 4. Performance Optimization Timeline
**File**: [05_results/PERFORMANCE_ACHIEVEMENTS.md](05_results/PERFORMANCE_ACHIEVEMENTS.md) (Section 2-3)
**Diagrams**:
- Phase timeline: Baseline (743s) → Parallel (112s) → ML Opt (30-40s) → Cited (25s)
- Bottleneck evolution chart
- Cost reduction breakdown ($0.80 → $0.31 → $0.14)

**Key Features**:
- ✅ 30x total speedup breakdown
- ✅ Per-component timing
- ✅ Cost analysis across phases
- ✅ Speedup factors (6.6x, 2.8x-3.7x, 4.5x)

**Use Case**: Project management, ROI analysis

---

## Cited Generation Diagrams

### 5. Cited Generation Pipeline
**File**: [03_theory_and_method/CITED_GENERATION_INNOVATION.md](03_theory_and_method/CITED_GENERATION_INNOVATION.md) (Section 2.1)
**Diagram**: 4-stage pipeline (Extract → Search → Generate → Verify)

**Key Features**:
- ✅ Stage 1: Topic extraction (3-5s, 50 concepts max)
- ✅ Stage 2: Parallel evidence search (2-3s, multiple sources)
- ✅ Stage 3: Cited generation (15-20s, 8000 token budget)
- ✅ Stage 4: Citation verification (1-2s, external-only policy)
- ✅ Authority tiers (Tier 1/2/3 with color badges)

**Use Case**: Fast educational note generation with citations

---

## ML Optimization Architecture Diagrams

### 6. ML Optimization Layer
**File**: [03_theory_and_method/ML_OPTIMIZATION_ARCHITECTURE.md](03_theory_and_method/ML_OPTIMIZATION_ARCHITECTURE.md) (Section 2.1)
**Related**: [ML_ALGORITHMS_EXPLAINED.md](03_theory_and_method/ML_ALGORITHMS_EXPLAINED.md) - Complete algorithm reference
**Diagram**: 8-model ensemble with parallel execution

**Key Features**:
- ✅ Cache optimizer (semantic dedup)
- ✅ Quality predictor (pre-screen)
- ✅ Priority scorer (UX optimization)
- ✅ Query expander (search diversity)
- ✅ Evidence ranker (relevance)
- ✅ Type classifier (domain routing)
- ✅ Semantic deduplicator (clustering)
- ✅ Adaptive controller (dynamic depth)

**Use Case**: Performance optimization, ML architecture reference

---

## Verification Scoring Diagrams

### 7. 6-Component Confidence Scoring
**File**: [07_papers_ieee/ieee_methodology_and_results.md](07_papers_ieee/ieee_methodology_and_results.md) (Section 3.2)
**Equations**: Mathematical formulation of S₁-S₆ components

**Key Features**:
- ✅ S₁: Semantic relevance (weight=0.18)
- ✅ S₂: Entailment strength (weight=0.35, **dominant**)
- ✅ S₃: Evidence diversity (weight=0.10)
- ✅ S₄: Agreement/consensus (weight=0.15)
- ✅ S₅: Contradiction signal (weight=0.10)
- ✅ S₆: Source authority (weight=0.12, **external-only Tier 1-3**)

**Use Case**: Research papers, technical documentation

---

## IEEE Paper Diagrams

### 8. IEEE Paper System Overview
**File**: [07_papers_ieee/ieee_methodology_and_results.md](07_papers_ieee/ieee_methodology_and_results.md) (Section 3.1)
**Diagram**: Dual-mode architecture with ML optimization preprocessing

**Key Features**:
- ✅ ML optimization layer (pre-filtering, 40-60% reduction)
- ✅ Cited mode (2 LLM calls, ~25s)
- ✅ Verifiable mode (11 LLM calls, ~112s, 7 stages)
- ✅ Output with quality metrics

**Use Case**: IEEE paper submission, academic publications

---

## Patent Diagrams

### 9. Patent System Architecture
**File**: [09_patents/patent_system_and_method_claims.md](09_patents/patent_system_and_method_claims.md) (Claim 2.1)
**Diagram**: 10-stage pipeline (input → ML optimization → 8 verification stages → output)

**Key Features**:
- ✅ ML optimization stage (0)
- ✅ Input processing (1)
- ✅ Evidence retrieval (2)
- ✅ Evidence encoding (3)
- ✅ Semantic scoring (4)
- ✅ NLI classification (5)
- ✅ Diversity assessment (6)
- ✅ Aggregation (7)
- ✅ Calibration (8)
- ✅ Selective prediction (9)
- ✅ Output (10)

**Use Case**: Patent applications, IP protection

---

## Comparison Diagrams

### 10. Cited vs. Verifiable Mode Comparison
**File**: [03_theory_and_method/CITED_GENERATION_INNOVATION.md](03_theory_and_method/CITED_GENERATION_INNOVATION.md) (Section 3)
**Table**: Performance, quality, cost comparison

| Metric | Verifiable | Cited | Winner |
|--------|-----------|-------|--------|
| Speed | 112s | 25s | Cited (4.5x) |
| LLM calls | 11 | 2 | Cited (5.5x) |
| Cost | $0.31 | $0.14 | Cited (55%) |
| Accuracy | 81.2% | 79.8% | Verifiable (+1.4%) |
| Content | 3.2 pages | 4.1 pages | Cited (+28%) |

**Use Case**: Feature comparison, mode selection guidance

---

## Authority Tier Diagrams

### 11. Authority Tiers and Verification Policy
**File**: [03_theory_and_method/CITED_GENERATION_INNOVATION.md](03_theory_and_method/CITED_GENERATION_INNOVATION.md) (Section 6)
**Diagram**: Tier hierarchy with badges and weights

**Tier 1 (Official)** - Authority weight 1.0 🏆
- Official documentation
- PEPs, RFCs, standards
- Academic databases

**Tier 2 (Community)** - Authority weight 0.8 ✅
- Stack Overflow
- GeeksforGeeks
- Official blogs

**Tier 3 (Educational)** - Authority weight 0.6 📖
- Wikipedia
- Khan Academy
- Tutorial sites

**Tier 0 (Rejected)** - Not allowed ❌
- User input/class notes
- Social media
- Unverified forums

**Use Case**: Source evaluation, citation verification

---

## Quality Diagnostics Diagrams

### 12. Quality Diagnostics Report
**File**: [03_theory_and_method/CITED_GENERATION_INNOVATION.md](03_theory_and_method/CITED_GENERATION_INNOVATION.md) (Section 5)
**Example Output**:
```
📊 Content Quality Analysis:
✅ Extracted: 28 concepts from 8 topics
✅ Evidence found: 156 sources (5.6 per concept avg)
✅ Coverage: 82% (23/28 concepts with sources)
✅ Generated: 4.2 pages with 94 citations

💡 Recommendations:
✅ Extraction quality: Excellent
✅ Evidence coverage: Strong (target: 70%+)
⚠️ Content density: Could be improved
```

**Use Case**: User feedback, quality assurance

---

## Flowcharts by Use Case

### Educational Use (Cited Mode)
```
Student uploads notes
    ↓ [Ingestion]
Canonicalize text
    ↓ [ML Optimization]
Check cache (90% hit rate)
    ↓ [Cited Pipeline]
Extract topics → Search evidence → Generate with citations → Verify citations
    ↓ [Output]
Display enriched notes with authority badges + quality report
    ↓ [Feedback]
Show extraction count, coverage %, recommendations
```

### High-Stakes Verification (Verifiable Mode)
```
Teacher/researcher inputs claim
    ↓ [Ingestion]
Canonicalize text
    ↓ [ML Optimization]
Pre-screen quality, deduplicate, prioritize
    ↓ [Verifiable Pipeline, 7 Stages]
Semantic → Retrieval → NLI → Diversity → Aggregation → Calibration → Selective Pred.
    ↓ [Output]
Prediction + Calibrated Confidence + Evidence + Reasoning
    ↓ [Decision]
If confidence > threshold: Auto-verify
Else: Flag for human review (hybrid workflow)
```

---

## Performance Metrics Diagrams

### 13. Speedup Breakdown
**File**: [05_results/PERFORMANCE_ACHIEVEMENTS.md](05_results/PERFORMANCE_ACHIEVEMENTS.md)

**Total Speedup**: 30x (743s → 25s)

| Phase | Speedup | Cumulative | Technique |
|-------|---------|-----------|-----------|
| Baseline | 1.0x | 1.0x | Sequential processing |
| Parallelization | 6.6x | 6.6x | asyncio evidence retrieval |
| ML Optimization | 2.8x-3.7x | 18.5x-24.8x | 8 models, cache, adaptive depth |
| Cited Generation | 4.5x | 30x | LLM reduction (11→2), ~25s typical |

---

## Document Cross-References

### By Diagram Type
- **Pipeline diagrams**: system_overview.md, detailed_pipeline.md, ingestion_pipeline.md
- **Performance diagrams**: PERFORMANCE_ACHIEVEMENTS.md
- **ML architecture diagrams**: ML_OPTIMIZATION_ARCHITECTURE.md
- **Cited generation diagrams**: CITED_GENERATION_INNOVATION.md
- **Academic diagrams**: ieee_methodology_and_results.md
- **Patent diagrams**: patent_system_and_method_claims.md

### By Audience
- **Executives**: System overview, performance achievements, cost analysis
- **Researchers**: IEEE paper diagrams, technical approach, ablation studies
- **Engineers**: Detailed pipeline, API specifications, ML layer architecture
- **Users**: Ingestion pipeline, output examples, quality diagnostics
- **Lawyers**: Patent diagrams, system/method claims, novelty positioning

---

## Diagram Updates (February 2026)

✅ **All diagrams updated to reflect**:
1. ML optimization layer (8 models) as preprocessing stage
2. Dual pipeline router (Cited mode ~25s vs. Verifiable mode ~112s)
3. Citation verification with authority tiers (Tier 1-3, external-only policy)
4. Performance improvements (30x speedup, 61% cost reduction)
5. Cited generation innovation (user's breakthrough idea)
6. Quality diagnostics (extraction tracking, coverage monitoring, recommendations)

---

## Future Diagram Needs

### Planned (Next 3 months)
- [ ] Streaming generation diagram (real-time content display)
- [ ] Edge deployment architecture (on-device models)
- [ ] Multi-modal cited generation (images + citations)
- [ ] Personalized cache diagram (user-specific optimization)

### Research Directions
- [ ] Hybrid mode diagram (combine cited + verifiable)
- [ ] Multi-hop citation chains (reasoning verification)
- [ ] Active learning workflow
- [ ] Causal vs. correlational claim classification

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial 7-stage verification system |
| 2.0 | Feb 2026 | +ML optimization layer (8 models) |
| 2.1 | Feb 2026 | +Dual pipeline (Cited+Verifiable), +Cited generation innovation |
| 3.0 | Feb 2026 | Current - All diagrams updated, comprehensive index created |

---

**End of Diagram Index**

For detailed specifications of any diagram, see the referenced file.
