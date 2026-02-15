"""
Diagnostic Guide: Debugging Mass Rejection in Smart Notes Verification

OVERVIEW:
=========

The Smart Notes verification pipeline is currently rejecting ~100% of claims.
This guide explains how to use the diagnostic tools to identify the root cause.

QUICK START:
============

1. Enable diagnostics via environment variables:

   export DEBUG_VERIFICATION=true
   export DEBUG_RETRIEVAL_HEALTH=true
   export DEBUG_NLI_DISTRIBUTION=true
   export DEBUG_CHUNKING=true

2. Run a test session:

   python app.py

3. Inspect the output:
   - Console logs show per-claim debugging information
   - outputs/debug_session_report_*.json contains detailed metrics

ROOT CAUSE DIAGNOSIS:
====================

The diagnostic system checks four components:

A. RETRIEVAL HEALTH
   - Do we retrieve evidence passages?
   - Are similarity scores high enough?
   
   Diagnosis:
   - If max_similarity < 0.45 for most claims → chunking or embedding issue
   - If num_candidates = 0 → sources not indexed properly
   - If avg_similarity < 0.30 → evidence too dissimilar from claims

B. ENTAILMENT CONFIDENCE
   - Do NLI models think evidence entails the claims?
   
   Diagnosis:
   - If mean_entailment < 0.40 across session → NLI alignment problem
   - If mean_entailment = 0.5-0.6 but threshold = 0.60 → threshold too strict
   - Symptom: "NLI mostly predicting NEUTRAL – possible chunking mismatch"

C. SOURCE INDEPENDENCE
   - Do we have sufficient independent sources?
   
   Diagnosis:
   - If avg_independent_sources < 2.0 → MIN_SUPPORTING_SOURCES=2 is blocking
   - Check if same material is counted multiple times vs. truly independent

D. CONTRADICTION DETECTION
   - Is the NLI model finding spurious contradictions?
   
   Diagnosis:
   - If avg_max_contradiction > 0.30 frequently → too sensitive to conflicts
   - Check if MAX_CONTRADICTION_PROB threshold should be raised

---

REMEDIATION STRATEGIES:
=======================

Strategy 1: RELAXED_VERIFICATION_MODE
Purpose: Determine if thresholds are too strict
Steps:
  1. Set: export RELAXED_VERIFICATION_MODE=true
  2. This overrides:
     - MIN_ENTAILMENT_PROB: 0.60 → 0.50
     - MIN_SUPPORTING_SOURCES: 2 → 1
     - MAX_CONTRADICTION_PROB: 0.30 → 0.50
  3. Re-run and observe rejection rate
  4. If rejection drops significantly → calibrate thresholds

Strategy 2: DEBUG_RETRIEVAL_HEALTH
Purpose: Identify retrieval bottlenecks
Steps:
  1. Set: export DEBUG_RETRIEVAL_HEALTH=true
  2. Look for warnings: "Retrieval weak for claim X"
  3. Inspect similarity scores - should be 0.6-0.95 for good hits
  4. If low, investigate:
     - Embedding model quality (e5-base-v2)
     - Chunking strategy (chunk size, overlap)
     - Source material preprocessing

Strategy 3: DEBUG_NLI_DISTRIBUTION
Purpose: Identify NLI misalignment
Steps:
  1. Set: export DEBUG_NLI_DISTRIBUTION=true
  2. Look for warnings about NEUTRAL predictions
  3. If mean_entailment < 0.40:
     - Check if [claim, evidence] pairs are well-aligned
     - Verify BART-MNLI model checkpoint loaded correctly
     - Consider alternative NLI models (e.g., RoBERTa)

Strategy 4: Incrementally Raise Thresholds
Purpose: Find the right balance without relaxation
Steps:
  1. Start with current thresholds in debug mode
  2. Print session summary showing rejection reasons
  3. Identify dominant failure mode (e.g., "LOW_ENTAILMENT": 300 cases)
  4. If dominant is LOW_ENTAILMENT:
     - Lower MIN_ENTAILMENT_PROB by 0.05 (0.60 → 0.55)
     - Re-test
     - Iterate until rejection rate is reasonable (20-40%)
  5. Do the same for other threshold parameters

---

INTERPRETING DEBUG OUTPUT:
==========================

Per-Claim Debug Output:
-----------------------
```
CLAIM DEBUG
ID: abc123def456
TEXT: "Velocity is the rate of change of position..."
Retrieved passages: 5
Top similarities: [0.71, 0.65, 0.60, 0.55, 0.48]
Top entailments: [0.72, 0.48, 0.31, 0.18, 0.08]
Top contradictions: [0.05, 0.02, 0.01, 0.01, 0.00]
Independent sources: 1
Decision: LOW_CONFIDENCE
Reason: INSUFFICIENT_SOURCES
```

Interpretation:
- Good retrieval (5 passages with 0.71 max similarity)
- Good entailment (0.72 prob)
- But only 1 source → rejected due to MIN_SUPPORTING_SOURCES=2
- Action: Check if source_id counting is correct, or lower MIN_SUPPORTING_SOURCES


Session Diagnostics Summary:
----------------------------
```
SESSION DIAGNOSTICS
Total claims: 243
Verified: 0
Low confidence: 12
Rejected: 231

Average max similarity: 0.52
Average max entailment: 0.41
Average contradiction: 0.07
Average independent sources: 1.1

Top rejection reasons:
NO_EVIDENCE: 120
LOW_ENTAILMENT: 80
INSUFFICIENT_SOURCES: 31
```

Interpretation:
- 50% of claims have NO_EVIDENCE (retrieval failed)
- 33% have LOW_ENTAILMENT (NLI not finding support)
- 13% have INSUFFICIENT_SOURCES (only 1 source found)
- Action priorities:
  1. Fix retrieval (120 / 243 = 49% loss)
  2. Then fix entailment (80 / 243 = 33% loss)
  3. Then source counting (31 / 243 = 13% loss)

---

JSON DEBUG REPORT:
==================

File: outputs/debug_session_report_SESSION_ID.json

Structure:
{
  "session_id": "session_abc123",
  "mode": "RELAXED",
  "summary": {
    "total_claims": 243,
    "verified": 12,
    "low_confidence": 45,
    "rejected": 186,
    "avg_similarity": 0.52,
    "avg_entailment": 0.41,
    "avg_contradiction": 0.07,
    "avg_sources": 1.1
  },
  "rejection_reasons": {
    "NO_EVIDENCE": 120,
    "LOW_ENTAILMENT": 80,
    "INSUFFICIENT_SOURCES": 31
  },
  "config_thresholds": {
    "min_entailment_prob": 0.50,
    "min_supporting_sources": 1,
    "max_contradiction_prob": 0.50
  }
}

Use this to:
- Compare before/after threshold changes
- Track improvement across iterations
- Document final tuning parameters

---

COMMON FAILURE PATTERNS:
=======================

Pattern 1: All NO_EVIDENCE (50%+ rejection)
Cause: Retrieval not working
Action:
  1. Check semantic_retriever.py – is FAISS index built?
  2. Verify embeddings are loaded (e5-base-v2)
  3. Check source chunking (chunk_size, chunk_overlap)
  4. Debug output: "Retrieval weak for claim X"

Pattern 2: All LOW_ENTAILMENT (33%+ rejection)
Cause: NLI model not confident, or [claim, evidence] misalignment
Action:
  1. Check BART-MNLI model loading
  2. Look at Debug output for "NLI mostly predicting NEUTRAL"
  3. May need to lower MIN_ENTAILMENT_PROB to 0.50-0.55
  4. Or: evidence chunks poorly align with claim text

Pattern 3: All INSUFFICIENT_SOURCES (20%+ rejection)
Cause: MIN_SUPPORTING_SOURCES = 2 but only 1 source found
Action:
  1. Check how many independent sources exist in material
  2. If only 1 source: set MIN_SUPPORTING_SOURCES = 1
  3. Or: fix source_id counting (may be overcounting or undercounting)

Pattern 4: Mixed Failures
Cause: Multiple issues compounding
Action:
  1. Fix largest contributor first (prioritize by %)
  2. Re-test incrementally
  3. Use RELAXED_VERIFICATION_MODE as temporary sanity check
  4. Then calibrate individual thresholds

---

STEP-BY-STEP DEBUGGING WORKFLOW:
=================================

1. Run with full diagnostics enabled:
   export DEBUG_VERIFICATION=true
   export DEBUG_RETRIEVAL_HEALTH=true
   export DEBUG_NLI_DISTRIBUTION=true
   export DEBUG_CHUNKING=true
   python app.py

2. Collect session report:
   JSON saved to outputs/debug_session_report_*.json

3. Read session summary to console:
   Check: avg_similarity, avg_entailment, avg_sources

4. Identify dominant failure (rejection_reason_distribution):
   If top_reason = "NO_EVIDENCE":
     → Focus on retrieval debugging
   Else if top_reason = "LOW_ENTAILMENT":
     → Focus on NLI calibration
   Else if top_reason = "INSUFFICIENT_SOURCES":
     → Check source independence counting

5. Try RELAXED_VERIFICATION_MODE:
   export RELAXED_VERIFICATION_MODE=true
   python app.py
   If rejection drops >50%:
     → Thresholds are too strict
     → Calibrate incrementally
   Else:
     → Core components (retrieval/NLI) need fixing

6. Make targeted fix:
   - Adjust single threshold
   - OR fix retrieval/NLI component
   - Re-test with diagnostics

7. Measure improvement:
   Compare rejection_reason_distribution before/after
   Continue until rejection rate is acceptable (20-40%)

---

EXPECTED METRICS FOR HEALTHY SYSTEM:
====================================

Healthy Verification Rate Profile:
- Verified: 50-70%
- Low Confidence: 15-25%
- Rejected: 10-30%

Healthy Diagnostic Metrics:
- avg_max_similarity: 0.65-0.85
- avg_max_entailment: 0.65-0.80
- avg_max_contradiction: <0.15
- avg_independent_sources: 2.0+

If your metrics differ significantly, adjust per Pattern matching.

---

DISABLING DIAGNOSTICS (Production):
===================================

Once calibrated, disable debug logs:
  export DEBUG_VERIFICATION=false
  export DEBUG_RETRIEVAL_HEALTH=false
  export DEBUG_NLI_DISTRIBUTION=false

This removes per-claim printing but keeps thresholds fixed.

"""
