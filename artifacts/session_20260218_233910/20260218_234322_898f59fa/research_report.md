# AI Verification Session Report

## Session Information

- **Session ID**: `session_20260218_233910`
- **Timestamp**: 2026-02-18T23:43:22.620788
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
- **Verified**: 13/25 (52.0%)
- **Low Confidence**: 0/25 (0.0%)
- **Rejected**: 12/25 (48.0%)

### Overall Metrics
- **Average Confidence**: 0.44

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
| A stack is a type of data structure that operates on the principle of Last In, F... | 0.89 | 2 | Yes |
| The Principle of Last In - First Out (LIFO) is a rule for managing data structur... | 0.86 | 2 | Yes |
| Push Operation in Stacks... | 0.90 | 2 | Yes |
| Pop Operation in Stacks... | 0.90 | 2 | Yes |
| You have a stack with the elements [2,4,6,1,8]. Perform a push operation to add ... | 0.85 | 2 | Yes |
| Push operation removes an element from the stack... | 0.90 | 2 | Yes |
| Elements can be added or removed from either end of the stack... | 0.86 | 2 | Yes |
| What is a push operation in a stack?... | 0.90 | 2 | Yes |
| What is a pop operation in a stack?... | 0.90 | 2 | Yes |
| What happens if you try to pop an element from an empty stack?... | 0.84 | 2 | Yes |
_... and 3 more_

### ⚠️ Low-Confidence Claims

_None_

### ❓ Questions Answered

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| What is a stack in data structures?... | 0.00 | 2 | Yes |
| What does LIFO mean in the context of stacks?... | 0.00 | 2 | Yes |
| Why is a stack considered a LIFO structure?... | 0.00 | 2 | Yes |
| Can you add... | 0.00 | 2 | Yes |
| remove elements from the bottom of a stack?... | 0.00 | 2 | Yes |
| How does a stack differ from a queue?... | 0.00 | 2 | Yes |
| Why are stacks important in programming?... | 0.00 | 2 | Yes |
| Can you give an example of where stacks are used in real-world applications?... | 0.00 | 2 | Yes |
| What is the difference between a static stack... | 0.00 | 2 | Yes |
| a dynamic stack?... | 0.00 | 2 | Yes |
_... and 2 more_
