# AI Verification Session Report

## Session Information

- **Session ID**: `session_20260218_214456`
- **Timestamp**: 2026-02-18T21:49:17.167020
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
- **Avg Chunk Size**: N/A chars
- **Methods**: N/A

## Verification Results

### Claim Status Distribution
- **Verified**: 14/26 (53.8%)
- **Low Confidence**: 0/26 (0.0%)
- **Rejected**: 12/26 (46.2%)

### Overall Metrics
- **Average Confidence**: 0.41

### Top Rejection Reasons
- RejectionReason.DISALLOWED_CLAIM_TYPE: 12 claims (100.0%)

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
| Data Structures are specialized formats used to organize, store, and manage data... | 0.83 | 2 | Yes |
| A stack is a data container that follows the rule "Last In, First Out" (LIFO). I... | 0.87 | 2 | Yes |
| The Principle of Stack in computer science means that the last item added to a s... | 0.87 | 2 | Yes |
| Push Operation... | 0.86 | 2 | Yes |
| Pop Operation... | 0.85 | 2 | Yes |
| The "Top of the Stack" refers to the last item added in a stack, which is the fi... | 0.84 | 2 | Yes |
| Push and Pop operations are the same.... | 0.85 | 2 | Yes |
| What is a push operation in a stack?... | 0.90 | 2 | Yes |
| What is a pop operation in a stack?... | 0.90 | 2 | Yes |
| What happens if we try to pop an element from an empty stack?... | 0.84 | 2 | Yes |
_... and 4 more_

### ⚠️ Low-Confidence Claims

_None_

### ❓ Questions Answered

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| What is a stack in data structures?... | 0.00 | 2 | Yes |
| What does LIFO mean in the context of stacks?... | 0.00 | 2 | Yes |
| What does FILO mean in the context of stacks?... | 0.00 | 2 | Yes |
| What is the top of the stack?... | 0.00 | 2 | Yes |
| Can you explain the principle of LIFO with an example?... | 0.00 | 2 | Yes |
| What are the applications of stacks in computer science?... | 0.00 | 2 | Yes |
| How can we implement a stack in programming?... | 0.00 | 2 | Yes |
| Can a stack be used to reverse a word or sentence?... | 0.00 | 2 | Yes |
| What is the difference between a stack... | 0.00 | 2 | Yes |
| a queue?... | 0.00 | 2 | Yes |
_... and 2 more_
