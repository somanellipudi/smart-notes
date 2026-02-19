# AI Verification Session Report

## Session Information

- **Session ID**: `session_20260218_195307`
- **Timestamp**: 2026-02-18T19:59:10.641210
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
| A stack is a data structure that follows the principle of "Last In, First Out" (... | 0.89 | 2 | N/A |
| A Last In - First Out (LIFO) data structure is a way to store information in a s... | 0.85 | 2 | N/A |
| First In - Last Out (FILO) is a principle that organizes data in a stack, a type... | 0.83 | 2 | N/A |
| The "Top of the Stack" refers to the last item added in a stack, which is the fi... | 0.84 | 2 | N/A |
| Push Operation... | 0.86 | 2 | N/A |
| Pop Operation... | 0.85 | 2 | N/A |
| Stack operations are actions performed on a stack, a data structure that follows... | 0.89 | 2 | N/A |
| A linear data structure is a type of organization for storing information sequen... | 0.85 | 2 | N/A |
| You have a stack with the elements [2, 4, 6]. Perform a push operation to add th... | 0.85 | 2 | N/A |
| You have a stack with the elements [2, 4, 6, 1]. Perform a pop operation.... | 0.86 | 2 | N/A |
_... and 7 more_

### ⚠️ Low-Confidence Claims

_None_

### ❌ Rejected Claims

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| Example: Explain the principle of Last In - First Out (LIFO...... | 0.00 | 2 | N/A |
| Example: Explain the principle of First In - Last Out (FILO...... | 0.00 | 2 | N/A |
| Example: What is the top of the stack?...... | 0.00 | 2 | N/A |
| What is a stack in data structures?... | 0.00 | 2 | N/A |
| What does Last In - First Out (LIFO) mean?... | 0.00 | 2 | N/A |
| What is the top of the stack?... | 0.00 | 2 | N/A |
| What is the difference between a stack... | 0.00 | 2 | N/A |
| a queue?... | 0.00 | 2 | N/A |
| Why are stacks called Last In - First Out (LIFO) structures?... | 0.00 | 2 | N/A |
| What are some real-world examples of stacks?... | 0.00 | 2 | N/A |
_... and 4 more_
