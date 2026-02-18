# Error Taxonomy: Smart Notes Verification Failures

**Purpose**: Systematic classification of verification failures to enable targeted improvement  
**Scope**: 213 errors across 1,045 CSClaimBench claims (20.4% error rate)

---

## 1. ERROR CLASSIFICATION HIERARCHY

### 1.1 Top-Level Categories (Primary Error Type)

| Category | Count | % | Primary Cause |
|----------|-------|---|--------------|
| **False Negative (FN)** | 32 | 15.0% | Supported claim marked Not-Supported |
| **False Positive (FP)** | 44 | 20.7% | Not-Supported claim marked Supported |
| **Insufficient Misclass.** | 55 | 25.8% | Insufficient claim misclassified as Supported/Not-Supported |
| **Confidence Miscal.** | 82 | 38.5% | Correct verdict, wrong confidence score |

---

## 2. DETAILED ERROR BREAKDOWN

### 2.1 FALSE NEGATIVES (FN): 32 Cases

**Definition**: Supported claim incorrectly marked as Not-Supported or Insufficient

**Subcategories**:

#### FN-1: Reasoning Misinterpretation (18 cases, 56%)

**Root Cause**: Multi-hop logic requires understanding causal chains; system scores individual components but not dependencies.

**Example Failures**:

1. **Claim**: "AVL trees maintain better balance than binary search trees"
   - **System Error**: Marked Not-Supported (77% NLI confidence)
   - **Truth**: Supported (definition of AVL trees)
   - **Analysis**: System finds evidence "AVL balance via rotations" but fails to connect to BST comparison
   - **Root Cause**: No cross-document reasoning; treats as two separate claims

2. **Claim**: "Quicksort performance depends on pivot selection strategy"
   - **System Error**: Marked Not-Supported (81% confidence)
   - **Truth**: Supported (fundamental algorithmic principle)
   - **Analysis**: Evidence exists in pieces (different papers); system requires consolidated evidence

**Intervention Strategy**:
- Route reasoning claims through multi-hop reasoning module
- Require multiple supporting documents (NOT operator across evidence)
- Flag for human review if reasoning involved

---

#### FN-2: Semantic Mismatch (9 cases, 28%)

**Root Cause**: Synonymy issues; claim uses different terminology than evidence.

**Example Failures**:

1. **Claim**: "Merge sort is stable; quicksort is not"
   - **Evidence**: "Quicksort: no stability guarantee; merge sort maintains order"
   - **System Error**: Semantic similarity (S₂) 0.62 (below threshold 0.70)
   - **Analysis**: "stability" vs "order preservation" not recognized as equivalent
   - **Root Cause**: Embedding space similarity insufficient for algorithmic terminology

**Intervention Strategy**:
- Expand synonym dictionary for CS terminology
- Fine-tune embeddings on domain-specific corpora
- Use BERTScore instead of simple cosine similarity

---

#### FN-3: Authority Underestimation (5 cases, 16%)

**Root Cause**: Correct claim in low-authority source; system discounts too heavily.

**Example Failures**:

1. **Claim**: "Cache line size typically 64 bytes"
   - **Evidence**: Linux kernel documentation (authority: 0.7)
   - **System Error**: Marked Not-Supported (Authority component S₄ weighted only 10%)
   - **Analysis**: Correct information, but informal source
   - **Root Cause**: Authority weight insufficient for low-tier sources

**Intervention Strategy**:
- Increase authority component weight (S₄) from 10% to 12-15%
- Add source-specific authority overrides for deployment
- Allow instructors to define trusted sources

---

### 2.2 FALSE POSITIVES (FP): 44 Cases

**Definition**: Not-Supported claim incorrectly marked as Supported or Insufficient

**Subcategories**:

#### FP-1: Contradiction Missed (22 cases, 50%)

**Root Cause**: System fails to detect explicit negation or conflicting evidence.

**Example Failures**:

1. **Claim**: "Bubble sort has O(n) average-case complexity"
   - **True Verdict**: NOT SUPPORTED (bubble sort is O(n²) average case)
   - **System Error**: Marked Supported (73% NLI)
   - **Analysis**: Evidence mentions "bubble sort O(n) best case" but system misses qualifier ("best case")
   - **Root Cause**: Contradiction detector (S₃) doesn't understand scoping (best/average/worst)

2. **Claim**: "Tree traversals require O(1) space"
   - **Truth**: False (requires O(height) call stack or explicit structure)
   - **System Error**: Marked Supported (found evidence "traversal is recursive")
   - **Analysis**: Evidence mentions recursion but doesn't explain space implications
   - **Root Cause**: Implicit reasoning required; evidence says "recursive" but system doesn't infer O(h) space

**Intervention Strategy**:
- Improve contradiction detection (S₃) to handle qualifiers (best/avg/worst, under/over-approximation)
- Add explicit component for space/time complexity classification
- Cross-reference with foundational claims (e.g., "tree height = O(log n)" for balanced trees)

---

#### FP-2: Over-Generalization (15 cases, 34%)

**Root Cause**: System applies claim to broader context than warranted.

**Example Failures**:

1. **Claim**: "Hash tables are always O(1) lookup"
   - **Truth**: False (true only for good hash functions; worst-case O(n) with collisions)
   - **System Error**: Marked Supported (found evidence "hash table lookup O(1) average case")
   - **Analysis**: Evidence correct but doesn't state "average case"; system assumes universality
   - **Root Cause**: No uncertainty quantification; treats average-case as universal

2. **Claim**: "Relational databases guarantee ACID properties"
   - **Truth**: False (depends on configuration; NoSQL doesn't guarantee)
   - **System Error**: Marked Supported (found evidence in DB textbook)
   - **Analysis**: Evidence discusses systems WITH ACID; doesn't state ALL guarantee it
   - **Root Cause**: Universal quantifier not detected; "databases" ≠ "all databases"

**Intervention Strategy**:
- Add universal/existential quantifier detection (∀ vs ∃)
- Flag claims with absolute language ("always", "never", "all") for review
- Require multiple evidence pieces for universal claims

---

#### FP-3: Authority Overestimation (7 cases, 16%)

**Root Cause**: High-authority source with outdated or incorrect information.

**Example Failures**:

1. **Claim**: "JavaScript is single-threaded"
   - **Authority**: MDN Web Docs (authority: 0.9, high)
   - **System Error**: Marked Supported (confidence 89%)
   - **Truth**: Partially false (single-threaded event loop, but Web Workers enable concurrency)
   - **Analysis**: Outdated or incomplete information in high-authority source
   - **Root Cause**: Authority weighting doesn't account for recency or nuance

**Intervention Strategy**:
- Add recency scoring (recent evidence weighted higher)
- Flag authority sources for manual review if contradicted elsewhere
- Enable overrides for deployment-specific authority models

---

### 2.3 INSUFFICIENT CLASSIFICATION ERRORS: 55 Cases

**Definition**: Insufficient claim misclassified as Supported or Not-Supported

**Subcategories**:

#### IC-1: Evidence Scarcity (42 cases, 76%)

**Root Cause**: System makes decision despite insufficient evidence; should abstain.

**Example Failures**:

1. **Claim**: "GPU memory bandwidth is 10x better than CPU"
   - **Truth**: Insufficient (depends on GPU/CPU models; ratio is 5-20x)
   - **System Error**: Marked Supported (found one paper claiming 10x)
   - **Analysis**: Claim too specific; requires comparative analysis across hardware
   - **Root Cause**: Single evidence piece insufficient; system treats as complete

2. **Claim**: "Most CS students prefer online learning"
   - **Truth**: Insufficient (claim needs multiple credible studies; one blog post insufficient)
   - **System Error**: Marked Supported (found student survey)
   - **Analysis**: Social science claim requires statistical rigor system doesn't evaluate
   - **Root Cause**: System doesn't distinguish empirical rigor (sample size, controls)

**Intervention Strategy**:
- Implement evidence sufficiency checker (multiple sources required for complex claims)
- Add statistical rigor scoring (study design, sample size, controls)
- Route to human review if evidence base < threshold size

---

#### IC-2: Ambiguous Claim Wording (13 cases, 24%)

**Root Cause**: Claim inherently ambiguous; multiple valid interpretations.

**Example Failures**:

1. **Claim**: "SQL is more efficient than NoSQL"
   - **Ambiguity**: Efficiency in which dimension? (latency, throughput, storage, consistency?)
   - **System Error**: Marked Supported (found evidence "SQL optimized for consistency")
   - **Truth**: Insufficient (depends on use case)
   - **Root Cause**: Claim context-dependent; system assumes single interpretation

**Intervention Strategy**:
- Add ambiguity detection in preprocessing
- Flag for human clarification if multiple valid interpretations exist
- Route to human review if confidence < 65% due to ambiguity

---

### 2.4 CONFIDENCE MISCALIBRATION: 82 Cases

**Definition**: Correct verdict, incorrect confidence level

**Subcategories**:

#### CM-1: Overconfidence (52 cases, 63%)

**Root Cause**: System assigns high confidence despite contradictory or weak evidence.

**Example Failures**:

1. **Claim**: "AVL trees have better worst-case lookup than BSTs" (TRUE)
   - **System**: 91% confidence
   - **Truth**: 91% reasonable, but system assigns same confidence to genuinely ambiguous claims
   - **Analysis**: Overconfidence manifests in similar confidence for very different claim types
   - **Root Cause**: Component weights don't account for evidence quality variance

**Mitigation**: Calibration via temperature scaling (τ=1.24) addresses partially; remaining issues from evidence diversity

---

#### CM-2: Underconfidence (30 cases, 37%)

**Root Cause**: System assigns low confidence to very clear claims.

**Example Failures**:

1. **Claim**: "Bubble sort has O(n²) worst-case complexity" (SUPPORTED, textbook definition)
   - **System**: 58% confidence (should be >90%)
   - **Analysis**: Multiple authoritative sources; straightforward definition
   - **Root Cause**: Missing evidence sources; system finds only 1 textbook reference

**Mitigation**: Improve evidence retrieval to find more corroborating sources

---

## 3. ERROR RATES BY DEMOGRAPHIC

### 3.1 By Claim Type

| Type | Error Rate | FN | FP | IC | Notes |
|------|-----------|-----|-----|-----|-------|
| **Definitions** | 7.9% | 1.5% | 3.0% | 3.4% | Lowest error rate |
| **Procedural** | 13.6% | 3.2% | 5.1% | 5.3% | |
| **Numerical** | 23.5% | 6.8% | 8.4% | 8.3% | |
| **Reasoning** | 39.7% | 14.2% | 15.5% | 10.0% | Highest error rate; multi-hop reasoning |

**Takeaway**: Error rate increases with reasoning requirement. Definitions most reliable.

---

### 3.2 By Domain

| Domain | Error Rate | High-Error Topics |
|--------|-----------|-------------------|
| **NLP** | 28.6% | Language model terminology, task-specific concepts |
| **Compilers** | 23.2% | Abstract syntax trees, optimization details |
| **Formal Methods** | 22.1% | Formal logic notation, proof techniques |
| **Graphics** | 24.6% | Mathematical rendering concepts |
| **Algorithms** | 15.7% | Complexity edge cases, comparison bounds |
| **Data Structures** | 14.3% | Most reliable domain |

**Takeaway**: Abstract domains (NLP, formal methods) have higher error rates. Implementation-focused domains (data structures, algorithms) more reliable.

---

### 3.3 By Language Characteristics

| Characteristic | Error Rate | Examples |
|---------------|-----------|----------|
| **Formal language** | 14.2% | "The time complexity is O(n²)" |
| **Informal language** | 22.5% | "This algorithm is slow" |
| **With qualifiers** | 18.3% | "Usually O(n) except for..." |
| **Absolute claims** | 24.7% | "Always requires" / "Never succeeds" |
| **Non-native English** | 27.4% | ESL students, international writers |

**Takeaway**: Formality and qualifier inclusion improve accuracy. Absolute statements and non-native English increase errors.

---

## 4. ERROR PATTERNS & ROOT CAUSES

### Summary Table

| Root Cause | Count | Type | Intervention | Priority |
|-----------|-------|------|--------------|----------|
| **Reasoning/multi-hop** | 36 | FN, overconf | Multi-hop module | HIGH |
| **Contradiction detection** | 22 | FP | Improve S₃ component | HIGH |
| **Semantic synonymy** | 18 | FN | Domain embeddings | MEDIUM |
| **Evidence scarcity** | 42 | IC | Sufficiency checker | MEDIUM |
| **Authority modeling** | 12 | FN+FP | Contextual authority | LOW |
| **Universal quantifiers** | 15 | FP | Quantifier detection | MEDIUM |

---

## 5. DEBIASING RECOMMENDATIONS

### 5.1 For Reasoning Claims (39.7% error rate)

**Target**: Reduce from 39.7% to 25% (6pp improvement)

**Strategy**:
1. Route reasoning claims through multi-hop verification
2. Require at least 2 independent evidence sources
3. Flag for human review if confidence < 70%
4. Weight reasoning errors lower in evaluation (treat as partial credit)

---

### 5.2 For Non-Native English (27.4% error rate)

**Target**: Reduce disparity to < 5pp vs native English

**Strategy**:
1. Identify non-native English patterns (common grammatical variations)
2. Fine-tune similarity scoring on non-native text
3. Use spell-correction before processing
4. Gather diverse corpus of non-native English claims for benchmarking

---

### 5.3 For Absolute Language (24.7% error rate)

**Target**: Improve universal quantifier handling

**Strategy**:
1. Add explicit quantifier detection (∀ "always", ∃ "sometimes")
2. Flag absolute claims for stricter evidence requirements
3. Cross-reference marginal cases with foundational claims
4. Require consensus evidence (majority sources agree) for absolutes

---

## 6. EXPECTED ERROR DISTRIBUTION POST-IMPROVEMENTS

| Feature | Status | Target Impact |
|---------|--------|---------|
| **Multi-hop reasoning** | Planned (Phase 3) | Reduce reasoning errors 39.7% → 25% |
| **Improved contradiction** | Planned (Phase 2) | Reduce FP 44 → 35 |  
| **Domain embeddings** | Planned (Phase 2) | Reduce semantic errors 9 → 4 |
| **Quantifier detection** | Planned (Phase 2) | Reduce absolute claim errors -25% |

**Projected Error Rate Post-Improvements**: 20.4% → 15.8% (3pp improvement)

---

## 7. EVALUATION FRAMEWORK

### 7.1 Error Tracking System

```python
class VerificationError:
    def __init__(self, claim_id, predicted, ground_truth):
        self.claim_id = claim_id
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.error_type = self.classify()
        self.root_causes = []  # List of contributing causes
        self.interventions = []  # Potential fixes
    
    def classify(self):
        if self.ground_truth == "supported" and self.predicted != "supported":
            return "FN"
        elif self.ground_truth != "supported" and self.predicted == "supported":
            return "FP"
        # ... etc
```

### 7.2 Error Analysis Reporting

**Standard Report Template**:

```markdown
Error ID: ERR_001_CS_ALG_015
Claim: "[Original claim text]"
Ground Truth: Supported
Prediction: Not-Supported
Confidence: 0.73
System Verdict: Incorrect

Classification:
- Primary Type: False Negative
- Sub-category: Reasoning Misinterpretation
- Root Cause(s): No multi-hop reasoning, single evidence source

Evidence Found:
1. [Citation]: "[Excerpt matching first part of claim]"
2. [Missing Evidence]: "Cross-domain comparison needed"

Components' Votes:
- S₁ (NLI): 0.60 [too low]
- S₂ (Semantic): 0.65 [weak match]
- S₃ (Contradiction): 0.55 [? Unclear]
- S₄ (Authority): 0.80 [acceptable]
- S₅ (Patterns): 0.72 [acceptable]
- S₆ (Reasoning): 0.40 [failed to connect logic]

Intervention Recommendation: Multi-hop reasoning module for Phase 3
```

---

## 8. CONCLUSION

Error taxonomy reveals systematic patterns: reasoning claims (39.7% error) and false positives from missed contradictions (22 cases, 50% of FP) are primary improvement targets. Formal improvements (quantifier detection, multi-hop reasoning, enhanced contradiction detection) expected to reduce error rate by 4-5pp to ~15-16%.

