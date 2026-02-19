# AI Verification Session Report

## Session Information

- **Session ID**: `session_20260219_103830`
- **Timestamp**: 2026-02-19T10:44:22.496174
- **Version**: 1.0.0
- **Random Seed**: 42

### Models Used
- **Language Model**: gpt-4
- **Embedding Model**: intfloat/e5-base-v2
- **NLI Model**: cross-encoder/qnli

### Inputs

## Ingestion Statistics

### Overall Extraction

- **Total Chunks (All Sources)**: 0
- **Avg Chunk Size**: N/A

## Verification Results

### Claim Status Distribution
- **Verified**: 17/33 (51.5%)
- **Low Confidence**: 0/33 (0.0%)
- **Rejected**: 16/33 (48.5%)

### Overall Metrics
- **Average Confidence**: 0.40

### Top Rejection Reasons
- RejectionReason.DISALLOWED_CLAIM_TYPE: 11 claims (68.8%)
- RejectionReason.INSUFFICIENT_CONFIDENCE: 5 claims (31.2%)

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
| A stack is a type of data structure that follows the principle of "Last In, Firs... | 0.89 | 2 | Yes |
| A Last In - First Out (LIFO) data structure is a way to store information in a s... | 0.85 | 2 | Yes |
| First In - Last Out (FILO) is a principle that prioritizes the order in which it... | 0.83 | 2 | Yes |
| The "Top of the Stack" refers to the last item added in a stack, which is the fi... | 0.84 | 2 | Yes |
| Push Operation... | 0.86 | 2 | Yes |
| Pop Operation... | 0.85 | 2 | Yes |
| A linear data structure is a type of organization for storing information sequen... | 0.85 | 2 | Yes |
| A Data Item is a single piece of information stored in a computer system, such a... | 0.78 | 2 | Yes |
| Push and pop operations are the same... | 0.87 | 2 | Yes |
| What is a push operation in a stack?... | 0.90 | 2 | Yes |
_... and 7 more_

### ⚠️ Low-Confidence Claims

_None_

### ❌ Rejected Claims

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| Example: What is a stack in data structures?...... | 0.00 | 2 | Yes |
| Example: What is the process of adding an element in the st...... | 0.00 | 2 | Yes |
| Example: What operation removes an element from the stack?...... | 0.00 | 2 | Yes |
| Example: What is the principle that a stack follows?...... | 0.00 | 2 | Yes |
| Example: Where are the addition and deletion of data items ...... | 0.00 | 2 | Yes |

### ❓ Questions Answered

| Claim | Confidence | Evidence | Citation |
|-------|-----------|----------|----------|
| What is a stack in data structures?... | 0.00 | 2 | Yes |
| What does LIFO mean in the context of stacks?... | 0.00 | 2 | Yes |
| What is the top of the stack?... | 0.00 | 2 | Yes |
| Why is a stack considered a linear data structure?... | 0.00 | 2 | Yes |
| What is the difference between LIFO... | 0.00 | 2 | Yes |
| FILO in the context of stacks?... | 0.00 | 2 | Yes |
| What types of problems can be solved using stacks?... | 0.00 | 2 | Yes |
| What is the difference between a stack... | 0.00 | 2 | Yes |
| a queue?... | 0.00 | 2 | Yes |
| Can we implement a stack using an array?... | 0.00 | 2 | Yes |
_... and 1 more_
