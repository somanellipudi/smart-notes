# Verification Diagnostics: Quick Start

## Problem
Smart Notes verification pipeline rejecting ~100% of claims. Need systematic debugging.

## Solution
Added comprehensive diagnostic instrumentation (no core logic changes).

---

## Quick Start

### Enable Debug Mode
```bash
export DEBUG_VERIFICATION=true
python app.py
```

View per-claim debugging output + session summary automatically.

### Try Relaxed Mode (Test if thresholds too strict)
```bash
export RELAXED_VERIFICATION_MODE=true
python app.py
```

Lowers thresholds:
- MIN_ENTAILMENT: 0.60 → 0.50
- MIN_SOURCES: 2 → 1
- MAX_CONTRADICTION: 0.30 → 0.50

If rejection drops significantly → thresholds were too strict.

---

## What You Get

### 1. Per-Claim Debug Output
```
CLAIM DEBUG
ID: abc123
TEXT: "Velocity is rate of change..."
Retrieved passages: 5
Top similarities: [0.71, 0.65, 0.60]
Top entailments: [0.72, 0.48, 0.31]
Independent sources: 1
Decision: LOW_CONFIDENCE
Reason: INSUFFICIENT_SOURCES
```

### 2. Session Summary
```
SESSION DIAGNOSTICS
Total claims: 243
Verified: 12 (5%)
Rejected: 186 (76%)

Avg max similarity: 0.52
Avg max entailment: 0.41

Top Rejection Reasons:
  NO_EVIDENCE: 120
  LOW_ENTAILMENT: 80
  INSUFFICIENT_SOURCES: 31
```

### 3. JSON Debug Report
Saved to: `outputs/debug_session_report_*.json`

---

## Diagnostic Features

| Flag | Purpose |
|------|---------|
| `DEBUG_VERIFICATION` | Per-claim details + session summary |
| `DEBUG_RETRIEVAL_HEALTH` | Warn if similarity < 0.45 |
| `DEBUG_NLI_DISTRIBUTION` | Warn if entailment < 0.40 |
| `DEBUG_CHUNKING` | Validate chunk sizes (50-800 tokens) |
| `RELAXED_VERIFICATION_MODE` | Test with looser thresholds |

---

## Interpreting Results

### Pattern: "NO_EVIDENCE" dominates (50%+)
**Cause**: Retrieval not working  
**Action**: Enable `DEBUG_RETRIEVAL_HEALTH`, check FAISS indexing

### Pattern: "LOW_ENTAILMENT" dominates (33%+)
**Cause**: NLI not confident, or misalignment  
**Action**: Enable `DEBUG_NLI_DISTRIBUTION`, may need to lower MIN_ENTAILMENT_PROB

### Pattern: "INSUFFICIENT_SOURCES" dominates (20%+)
**Cause**: Only 1 source found, but MIN_SUPPORTING_SOURCES=2  
**Action**: Check source independence counting, or lower to MIN_SOURCES=1

---

## Files

- **diagnostics.py**: Core instrumentation class
- **DIAGNOSTIC_GUIDE.md**: Complete debugging workflow (450 lines)
- **DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md**: Technical details (500 lines)
- **test_verification_not_all_rejected.py**: Sanity check tests

---

## Expected Healthy Metrics

| Metric | Healthy Range |
|--------|---------------|
| Verification Rate | 50-70% |
| Rejection Rate | 10-30% |
| Avg Similarity | 0.65-0.85 |
| Avg Entailment | 0.65-0.80 |
| Avg Sources | 2.0+ |

---

## Next Steps

1. Run with `DEBUG_VERIFICATION=true`
2. Check session summary rejection reasons
3. Try `RELAXED_VERIFICATION_MODE=true` to test thresholds
4. Apply fixes based on dominant failure pattern
5. Re-test and iterate

See **DIAGNOSTIC_GUIDE.md** for detailed troubleshooting strategies.

---

## Testing

```bash
python -m pytest tests/test_verification_not_all_rejected.py -v
```

✓ All tests passing (16/16)
