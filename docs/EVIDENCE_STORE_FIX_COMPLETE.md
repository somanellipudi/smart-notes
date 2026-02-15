## Evidence Store Integration - Root Cause Fix Complete

### Problem Identified
The verification pipeline was yielding **~100% REJECTED** status for any input. 

**Root Cause**: The evidence store was empty or not properly indexed before verification began. The pipeline was running verification logic without any indexed evidence to retrieve from, causing all claims to be marked as LOW_CONFIDENCE or REJECTED due to NO_EVIDENCE.

### Solution Implemented

#### 1. **Evidence Store Module** (`src/retrieval/evidence_store.py` - 277 lines)
- Created centralized `EvidenceStore` class with FAISS indexing for efficient semantic search
- Features:
  - `Evidence` dataclass for individual evidence chunks with metadata
  - FAISS IndexFlatL2 for L2 distance (normalized for cosine similarity)
  - Fallback brute-force search if FAISS unavailable
  - Statistics tracking: num_chunks, total_chars, num_sources, faiss_index_size
  - Hard validation: `validate()` returns (bool, error_message), used by pipeline to prevent silent failures

#### 2. **Evidence Builder Module** (`src/retrieval/evidence_builder.py` - 350 lines)
- Created `build_session_evidence_store()` function that:
  1. Validates input text length (warns <500 chars, errors <100 chars)
  2. Chunks input with overlapping windows (500 chars, 50 char overlap)
  3. Adds external context as separate source if provided
  4. Includes equations as evidence items
  5. Ingests URLs (YouTube + articles) if enabled
  6. Returns (store, stats) WITHOUT building index (caller adds embeddings)

#### 3. **Pipeline Integration** (Modified `src/reasoning/verifiable_pipeline.py`)
- **Step 0.5: Build Evidence Store (NEW, mandatory step)**
  - Called AFTER URL ingestion (Step 0) and BEFORE baseline generation (Step 1)
  - Builds evidence store from all inputs (combined_content, external_context, equations, urls)
  - Computes embeddings using semantic_retriever's encoder (384-dim vectors)
  - Builds FAISS index with all evidence
  - **Hard validation**: raises RuntimeError with clear message if validation fails
  - Prevents any verification from proceeding without indexed evidence

- **Step 3: Updated Claim Retrieval (Modified)**
  - Changed from keyword matching on raw text chunks to FAISS semantic search
  - Encodes each claim as query embedding
  - Searches evidence_store with `search(query_embedding, top_k=10, min_similarity=0.3)`
  - Retrieves only semantically similar evidence chunks
  - Maintains same decision policy application

#### 4. **Diagnostics Integration** (Modified `src/verification/diagnostics.py`)
- Added `evidence_stats` field to `VerificationDiagnostics.__init__()`
- Updated `save_debug_report()` to include evidence section:
  ```json
  {
    "evidence": {
      "input_chars": 2500,
      "num_chunks": 8,
      "total_chars": 2500,
      "num_sources": 1,
      "faiss_index_size": 8
    },
    ...
  }
  ```
- Provides visibility into evidence ingestion and indexing

#### 5. **Test Coverage** (Created `tests/test_evidence_store_integration.py`)
9 comprehensive tests covering:
- ✅ Evidence store contains chunks from input
- ✅ FAISS index is properly built
- ✅ Store validation passes for valid inputs
- ✅ Claims verified with matching evidence (prevents ~100% rejection)
- ✅ Errors raised for empty input
- ✅ Errors raised for insufficient input
- ✅ Validation fails for empty stores
- ✅ Statistics include all required metrics
- ✅ External context counted as separate source

**All 9 tests PASS**

### How the Fix Works

#### Before Fix (Broken):
```
Input → URL Ingest → Baseline LLM → Extract Claims → Retrieve Evidence (empty!) 
         → NO evidence found → Apply Policy → "NO_EVIDENCE" → REJECTED (100%)
```

#### After Fix (Correct):
```
Input → URL Ingest → [NEW] Build Evidence Store:
  1. Chunk input text (500-char overlap chunks)
  2. Add external context
  3. Add equations
  4. Ingest URLs
  5. Compute embeddings (encoder.encode)
  6. Build FAISS index
  7. VALIDATE before proceeding (hard error if fails)
  
→ Baseline LLM → Extract Claims → Retrieve Evidence (FAISS search)
  → Find semantically similar chunks → Apply Policy 
  → VERIFIED/LOW_CONFIDENCE/REJECTED (based on actual evidence match quality)
```

### Configuration Changes
- Added `MIN_INPUT_CHARS_FOR_VERIFICATION = 500` to config.py
- Evidence store building honors `ENABLE_URL_SOURCES` setting
- Chunk size configurable (default: 500 chars, overlap: 50 chars)

### Error Handling

**Hard Validation**: Pipeline will NOT proceed if:
1. Evidence store has 0 chunks
2. Total evidence chars < MIN_INPUT_CHARS_FOR_VERIFICATION
3. FAISS index not built

**Error Message Example**:
```
RuntimeError: Evidence store validation failed: Evidence store has 0 chunks. 
Cannot run verification.

Verification cannot proceed without valid evidence. Please ensure:
1. Input text is at least 500 characters
2. Input contains substantive content
3. If using URLs, at least one URL successfully ingested
```

### Testing the Fix
1. Run evidence store tests: `pytest tests/test_evidence_store_integration.py -v`
2. Run full test suite: `pytest tests/ -v`
3. Test with real input: Use Streamlit UI or CLI with `input_chars > 500`

### Next Steps Verified
✅ Evidence store module complete and tested
✅ Evidence builder module complete and tested  
✅ Pipeline Step 0.5 (evidence building) integrated
✅ Claim retrieval updated to use FAISS search
✅ Diagnostics updated to track evidence stats
✅ All 9 integration tests pass
✅ Clear error messages for debugging

### Impact
- **Fixes 100% rejection issue**: Evidence is now always indexed before verification
- **Improves search quality**: FAISS semantic search vs keyword matching
- **Better observability**: Debug reports show evidence stats
- **Prevents silent failures**: Hard validation raises clear errors
- **Maintainability**: Explicit, mandatory evidence step in pipeline
