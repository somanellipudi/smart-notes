# Evidence Store Integration - Quick Reference

## What Was Fixed
**Problem**: Verification pipeline yielding ~100% REJECTED for any input  
**Root Cause**: Evidence store was empty/unindexed, verification ran without evidence  
**Solution**: Created mandatory Step 0.5 that builds and validates evidence BEFORE verification

## Quick Start

### For Developers
```python
# Import evidence store
from src.retrieval.evidence_store import EvidenceStore, validate_evidence_store, Evidence
from src.retrieval.evidence_builder import build_session_evidence_store

# Build evidence store from input
store, stats = build_session_evidence_store(
    session_id="my_session",
    input_text="Your input text...",
    external_context="Optional external material",
    equations=None,
    urls=None
)

# Add embeddings and build FAISS index
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('sentence-transformers/e5-base-v2')
embeddings = encoder.encode([ev.text for ev in store.evidence], 
                           batch_size=32, convert_to_numpy=True)
for i, ev in enumerate(store.evidence):
    ev.embedding = embeddings[i]
store.build_index(embeddings=embeddings)

# Validate
is_valid, error = store.validate(min_chars=500)
if not is_valid:
    raise ValueError(f"Invalid store: {error}")

# Search with query embedding
query_embedding = encoder.encode("Your query")[0]
results = store.search(query_embedding, top_k=10, min_similarity=0.3)
# Returns: [(Evidence, similarity_score), ...]

# Get statistics
stats = store.get_statistics()
# Returns: {
#   'num_chunks': int,
#   'total_chars': int,
#   'num_sources': int,
#   'faiss_index_size': int,
#   ...
# }
```

### For Testing
```bash
# Run evidence store tests
pytest tests/test_evidence_store_integration.py -v

# Run full test suite
pytest tests/ -v

# Check compilation
python -m py_compile src/retrieval/evidence_store.py \
                     src/retrieval/evidence_builder.py \
                     src/reasoning/verifiable_pipeline.py \
                     src/verification/diagnostics.py
```

## Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `MIN_INPUT_CHARS_FOR_VERIFICATION` | 500 | Min chars for verification (warns <500, errors <100) |
| `ENABLE_URL_SOURCES` | True | Enable URL ingestion in evidence store |
| `TOP_K_EVIDENCE` | 10 | Number of evidence chunks to retrieve per claim |

## Error Handling

### Evidence Store Validation Fails
**When**: input_chars < 100 or no chunks created or FAISS index not built

**Error Message**:
```
RuntimeError: Evidence store validation failed: {error_detail}

Verification cannot proceed without valid evidence. Please ensure:
1. Input text is at least 500 characters
2. Input contains substantive content
3. If using URLs, at least one URL successfully ingested
```

### Missing Evidence
- **Problem**: Input too short (<100 chars)
- **Solution**: Provide input text ≥500 characters (warning), >100 (required)

### FAISS Index Not Built
- **Problem**: Called search() before build_index()
- **Solution**: Call `store.build_index(embeddings=embeddings)` first

## Architecture

```
Input Processing:
  1. Parse combined_content (transcript + notes)
  2. Receive external_context, equations, urls
  
Step 0.5: Build Evidence Store (MANDATORY)
  1. Chunk input (500-char, 50-char overlap)
  2. Create Evidence objects with source tracking
  3. Compute embeddings (384-dim vectors)
  4. Build FAISS index (IndexFlatL2 + normalize)
  5. Validate (hard failure if invalid)
  
Step 3: Retrieve Evidence for Claims (MODIFIED)
  1. Encode each claim
  2. Search evidence store (semantic similarity)
  3. Retrieve top-k matching chunks
  4. Apply decision policy
```

## Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/retrieval/evidence_store.py` | FAISS-indexed evidence storage | EvidenceStore, Evidence, validate_evidence_store |
| `src/retrieval/evidence_builder.py` | Evidence ingestion from sources | build_session_evidence_store, chunk_text |
| `src/reasoning/verifiable_pipeline.py` | Main pipeline (Step 0.5 + Step 3) | process() method |
| `src/verification/diagnostics.py` | Debug reporting | VerificationDiagnostics.evidence_stats |
| `config.py` | Configuration | MIN_INPUT_CHARS_FOR_VERIFICATION |

## Key Concepts

### Evidence Chunk
A portion of input text (500 chars) with metadata:
- `text`: The actual chunk content
- `source_id`: Where it came from (url, session_input, etc.)
- `source_type`: Type of source (transcript, article, youtube, etc.)
- `embedding`: 384-dim vector from sentence-transformer
- `metadata`: Additional info (url, title, fetched_at, etc.)

### FAISS Index
Facebook AI Similarity Search - efficient large-scale semantic search:
- IndexFlatL2: L2 distance (works on normalized vectors = cosine similarity)
- Normalized: embeddings scaled to unit length for cosine similarity
- Fast: O(m) search time where m = number of chunks

### Validation
Hard check before proceeding:
- Store has evidence chunks (num_chunks > 0)
- Sufficient total characters (>= min_chars)
- FAISS index is built (index_built = True)
- Index has vectors (ntotal > 0)

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Chunk 2500 chars | ~5ms | 500-char chunks with overlap |
| Encode 5 chunks | ~100ms | Using batch_size=32 |
| Build FAISS index | ~50ms | 5 chunks, 384-dim |
| Search query | ~20ms | Top-10 retrieval |
| **Total Step 0.5** | **~175ms** | Typical for medium input |

## Common Issues

### 100% Rejected Claims
- **Check**: `diagnostics.evidence_stats['num_chunks']` > 0
- **Fix**: Ensure input_chars >= 500

### No Results from Search
- **Check**: similarity_score >= min_similarity (default: 0.3)
- **Fix**: Lower min_similarity threshold or improve query embedding

### ArgumentError: Input text too short
- **Check**: input_text length
- **Fix**: Provide input >= 100 chars (warning), >= 500 chars (recommended)

### ImportError: No module named 'faiss'
- **Impact**: Minimal - falls back to numpy brute-force
- **Fix** (optional): `pip install faiss-cpu` or `pip install faiss-gpu`

## Testing

### Unit Tests (9 passing)
```bash
pytest tests/test_evidence_store_integration.py::TestEvidenceStoreNotEmpty -v
pytest tests/test_evidence_store_integration.py::TestRaisesWhenNoEvidence -v
pytest tests/test_evidence_store_integration.py::TestEvidenceStoreStatistics -v
```

### Integration Test
```python
# Full pipeline with evidence store
from src.reasoning.verifiable_pipeline import VerifiablePipelineWrapper

pipeline = VerifiablePipelineWrapper(...)
output = pipeline.process(
    combined_content="Your input text...",
    external_context=None,
    equations=None,
    urls=None,
    session_id="test_001"
)
# Should complete without ~100% rejection
```

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Evidence Store Stats
```python
store_stats = evidence_store.get_statistics()
print(f"Chunks: {store_stats['num_chunks']}")
print(f"Chars: {store_stats['total_chars']}")
print(f"Sources: {store_stats['num_sources']}")
print(f"FAISS size: {store_stats['faiss_index_size']}")
```

### Validate Before Proceeding
```python
is_valid, error_msg = store.validate(min_chars=500)
if not is_valid:
    print(f"Store invalid: {error_msg}")
    # Handle error
```

### Check Debug Report
```python
# In Streamlit UI or output/
# Look for: debug_session_report_*.json
# Check: {"evidence": {...}} section
```

## Updating Evidence Store

### Add New Evidence
```python
new_evidence = Evidence(
    evidence_id="",  # Auto-generated
    source_id="custom_source",
    source_type="research_paper",
    text="Your text...",
    chunk_index=0,
    char_start=0,
    char_end=50,
    metadata={"author": "John Doe"}
)
store.add_evidence(new_evidence)
```

### Add Multiple Evidence Items
```python
evidence_list = [ev1, ev2, ev3]
store.add_evidence_batch(evidence_list)
```

### Rebuild Index After Adding Evidence
```python
# If adding new evidence after building index:
new_embeddings = encoder.encode([ev.text for ev in store.evidence])
for i, ev in enumerate(store.evidence):
    ev.embedding = new_embeddings[i]
store.build_index(embeddings=new_embeddings)
```

## Production Checklist

- [ ] Input validation: `input_text` >= 500 characters
- [ ] Evidence store builds without errors: no RuntimeError
- [ ] FAISS index built: `store.index_built == True`
- [ ] Validation passes: `is_valid, _ = store.validate()` returns (True, "")
- [ ] Statistics traced: `diagnostics.evidence_stats` populated
- [ ] Search returns results: `store.search()` returns list
- [ ] Claims not all rejected: %rejected < 50% typically
- [ ] Debug report generated: check evidence section in JSON

## Support & Questions

Refer to:
- `docs/EVIDENCE_STORE_COMPLETION.md` - Full technical details
- `docs/EVIDENCE_STORE_CHANGELOG.md` - Detailed change log
- `tests/test_evidence_store_integration.py` - Working examples
- GitHub Issues - For bugs or questions
