# AI Verification Session Report

## Session Information

- **Session ID**: `session_20260218_190323`
- **Timestamp**: 2026-02-18T19:08:17.672251
- **Version**: 1.0.0
- **Random Seed**: 42

### Models Used
- **Language Model**: gpt-4
- **Embedding Model**: intfloat/e5-base-v2
- **NLI Model**: cross-encoder/qnli

### Inputs

## Ingestion Statistics

- **Total Pages**: 0
- **Pages OCR'd**: 0
- **Headers Removed**: 0
- **Footers Removed**: 0
- **Watermarks Removed**: 0

### Extraction
- **Total Chunks**: 0
- **Avg Chunk Size**: 512 chars
- **Methods**: 

## Verification Results

### Claim Status Distribution
- **Verified**: 17/31 (54.8%)
- **Low Confidence**: 0/31 (0.0%)
- **Rejected**: 14/31 (45.2%)

### Overall Metrics
- **Average Confidence**: 0.44

### Top Rejection Reasons
- INSUFFICIENT_CONFIDENCE: 3 claims
- NO_EVIDENCE: 0 claims
- LOW_SIMILARITY: 0 claims
- INSUFFICIENT_SOURCES: 0 claims
- LOW_CONSISTENCY: 0 claims

## What to Trust / What Not to Trust

### ✅ You Can Trust Claims That Are:
- **Verified with high confidence** (status = VERIFIED, confidence > 0.8)
- **Supported by multiple evidence sources** from authoritative materials
- **Not contradicted** by any retrieved evidence
- **Common in domain literature** (not edge cases or disputed claims)

### ⚠️ Use With Caution:
- **Low-confidence claims** that lack multiple supporting sources
- **Claims from specialized/technical domains** that may have nuanced context
- **Claims where model confidence is 0.5-0.8** (middle range)

### ❌ Do Not Trust:
- **Rejected claims** - evidence contradicted or insufficient
- **Claims with zero supporting evidence** from course materials
- **Highly technical definitions** without expert review
- **Edge case or controversial claims** marked as low confidence

### 🔍 How to Verify Further:
1. Check the evidence citations (linked to page numbers and snippets below)
2. Cross-reference low-confidence claims with original texts
3. For critical material, consult with instructor or textbook
4. Review the confidence score and calibration metrics

## Verified Claims Table

### ✅ Verified Claims

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| A stack is a data structure that follows the principle of Last In, First Out (LI... | 0.89 | 2 | N/A |
| A Last In - First Out (LIFO) data structure is a way to organize information, si... | 0.85 | 2 | N/A |
| "First In - Last Out (FILO) refers to a method in which the last item added to a... | 0.83 | 2 | N/A |
| The "Top of the Stack" refers to the last item added in a stack, which is the fi... | 0.84 | 2 | N/A |
| Push Operation... | 0.86 | 2 | N/A |
| Pop Operation... | 0.85 | 2 | N/A |
| Inserting an element into a stack is known as pushing. This operation adds a new... | 0.87 | 2 | N/A |
| Deletion from a stack refers to removing the last item added (top element) using... | 0.86 | 2 | N/A |
| You have a stack with the elements [2,4,6,1,8]. You want to know if the stack is... | 0.81 | 2 | N/A |
| You have a stack with the elements [2,4,6,1,8]. You want to know if the stack is... | 0.81 | 2 | N/A |
_... and 7 more_

### ⚠️ Low-Confidence Claims

_None_

### ❌ Rejected Claims

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| Example: You have a stack with the elements [2,4,6,1,8]. Yo...... | 0.00 | 2 | N/A |
| Example: You have a stack with the elements [2,4,6,1,8]. Yo...... | 0.00 | 2 | N/A |
| Example: You have a stack with the elements [2,4,6,1,8]. Yo...... | 0.00 | 2 | N/A |
| What is a stack in data structures?... | 0.00 | 2 | N/A |
| What does LIFO mean in the context of stacks?... | 0.00 | 2 | N/A |
| What is the top of the stack?... | 0.00 | 2 | N/A |
| Why are stacks considered LIFO structures?... | 0.00 | 2 | N/A |
| Can you explain the process of insertion in a stack?... | 0.00 | 2 | N/A |
| How is deletion performed in a stack?... | 0.00 | 2 | N/A |
| What is the significance of the top of the stack?... | 0.00 | 2 | N/A |
_... and 4 more_
