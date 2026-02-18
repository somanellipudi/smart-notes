# Evidence Artifact Persistence

**Status**: Implemented (2026-02-17)  
**Purpose**: Deterministic, reproducible evidence retrieval with persistent caching

---

## Overview

The Evidence Artifact Store provides content-addressable storage for:
- **Sources**: Original documents/transcripts with metadata
- **Spans**: Text chunks with position information
- **Embeddings**: Dense vectors (384-dim) for retrieval
- **Run Metadata**: Config snapshots, seeds, model versions, git commits

### Key Features

1. **Deterministic IDs**: SHA256 hashing ensures stable identifiers
   - `source_id = SHA256(source_type + origin + normalized_text)`
   - `span_id = SHA256(source_id + start + end + normalized_text)`
   - `run_id = timestamp + short_hash`

2. **Content-Addressable Caching**: Same input + same model → cache hit
   - Avoids re-embedding identical content
   - Speeds up repeated verification runs

3. **Reproducible Retrieval**: Global seed control for:
   - NumPy random operations
   - PyTorch (if available)
   - FAISS index operations
   - Python random module

4. **Structured Storage**:
   ```
   artifacts/<session_id>/<run_id>/
       metadata.json       # Run metadata (seed, config, models)
       sources.jsonl       # Source documents (1 JSON per line)
       spans.jsonl         # Text spans (1 JSON per line)
       embeddings.npz      # NumPy compressed embeddings + span IDs
   ```

---

## Configuration

Add to `.env` or set as environment variables:

```bash
# Artifact persistence
ENABLE_ARTIFACT_PERSISTENCE=true
ARTIFACTS_DIR=artifacts/

# Cache controls
EMBEDDING_CACHE_ENABLED=true      # Reuse embeddings for same content
RETRIEVAL_CACHE_ENABLED=false     # Cache top-k retrieval results
NLI_CACHE_ENABLED=false           # Cache NLI consistency checks

# Reproducibility
GLOBAL_RANDOM_SEED=42
```

---

## Usage

### Automatic Caching (Transparent)

Evidence building automatically uses artifact store:

```python
from src.retrieval.evidence_builder import build_session_evidence_store
from src.retrieval.embedding_provider import EmbeddingProvider

provider = EmbeddingProvider()
store, stats = build_session_evidence_store(
    session_id="lecture_001",
    input_text="Today we covered derivatives...",
    embedding_provider=provider
)

# First run: cache miss (stats["cache_status"] == "miss")
# Second run: cache hit (stats["cache_status"] == "hit")
```

### Manual Artifact Store

```python
from src.retrieval.artifact_store import (
    ArtifactStore, SourceArtifact, SpanArtifact, RunMetadata,
    compute_source_id, compute_span_id
)

# Create store
store = ArtifactStore(artifacts_dir="artifacts/", session_id="test_001")

# Add source
source_id = compute_source_id("transcript", "lecture.txt", full_text)
store.add_source(SourceArtifact(
    source_id=source_id,
    source_type="transcript",
    origin="lecture.txt",
    page_num=None,
    normalized_text_hash=compute_text_hash(full_text),
    char_count=len(full_text),
    metadata={}
))

# Add spans
span_id = compute_span_id(source_id, 0, 100, span_text)
store.add_span(SpanArtifact(
    span_id=span_id,
    source_id=source_id,
    start=0,
    end=100,
    text=span_text,
    page_num=None,
    chunk_idx=0,
    char_count=100
))

# Set embeddings (must match span count)
embeddings = np.random.rand(1, 384).astype('float32')
store.set_embeddings(embeddings)

# Save
metadata = RunMetadata(
    run_id=store.run_id,
    session_id="test_001",
    timestamp=datetime.now().isoformat(),
    random_seed=42,
    config_snapshot={"embedding_model": "all-MiniLM-L6-v2"},
    git_commit="abc1234",
    model_ids={"embedding": "all-MiniLM-L6-v2"},
    source_count=1,
    span_count=1,
    embedding_dim=384,
    cache_status="miss"
)
store.save(metadata)

# Load later
loaded_store = ArtifactStore.load("artifacts/", "test_001", store.run_id)
```

### Seed Control

```python
from src.utils.seed_control import set_global_seed

# Set before any model operations
set_global_seed(42)

# Now all operations are deterministic:
# - NumPy random
# - PyTorch random
# - FAISS random
# - Python random
```

---

## Artifact Schema

### `metadata.json`

```json
{
  "run_id": "20260217_120000_abc12345",
  "session_id": "lecture_001",
  "timestamp": "2026-02-17T12:00:00",
  "random_seed": 42,
  "config_snapshot": {
    "content_hash": "sha256_hash_of_input",
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50
  },
  "git_commit": "abc1234",
  "model_ids": {
    "embedding": "all-MiniLM-L6-v2",
    "llm": "gpt-4"
  },
  "source_count": 1,
  "span_count": 5,
  "embedding_dim": 384,
  "cache_status": "miss"
}
```

### `sources.jsonl` (1 JSON per line)

```json
{"source_id": "sha256_hash", "source_type": "transcript", "origin": "lecture.txt", "page_num": null, "normalized_text_hash": "sha256_hash", "char_count": 1500, "metadata": {}}
```

### `spans.jsonl` (1 JSON per line)

```json
{"span_id": "sha256_hash", "source_id": "source_sha256", "start": 0, "end": 100, "text": "Today we covered...", "page_num": null, "chunk_idx": 0, "char_count": 100}
{"span_id": "sha256_hash2", "source_id": "source_sha256", "start": 50, "end": 150, "text": "derivatives. A derivative...", "page_num": null, "chunk_idx": 1, "char_count": 100}
```

### `embeddings.npz`

NumPy compressed archive:
- `embeddings`: (N, 384) float32 array
- `span_ids`: (N,) array of span IDs (matches span order)

---

## Cache Lookup Logic

1. **Compute content hash**: `SHA256(normalized_input_text)`
2. **Look for matching run**: Search `artifacts/<session_id>/*/metadata.json`
3. **Match criteria**:
   - `config_snapshot.content_hash == current_content_hash`
   - `model_ids.embedding == current_embedding_model`
4. **If match found**:
   - Load `sources.jsonl`, `spans.jsonl`, `embeddings.npz`
   - Build EvidenceStore from cached data
   - Skip re-embedding (save time/cost)
5. **If no match**:
   - Compute embeddings
   - Save new artifacts
   - Return fresh EvidenceStore

---

## Reproducibility Guarantees

### What is deterministic:

✅ **Chunking**: Fixed char-based boundaries (chunk_size=500, overlap=50)  
✅ **Span IDs**: Content-addressable hashing  
✅ **Embeddings**: Frozen pre-trained models with seed control  
✅ **FAISS retrieval**: Flat index (brute-force) with fixed order  
✅ **Artifact storage**: JSONL + NPZ formats (stable serialization)

### What is NOT deterministic (without mitigation):

❌ **LLM sampling**: Use `temperature=0` for greedy decoding  
❌ **GPU precision**: Use `device="cpu"` for exact reproducibility  
❌ **Reranking**: Disable with `ENABLE_RERANKER=false`  
❌ **NLI checks**: Disable with `VERIFIABLE_CONSISTENCY_ENABLED=false`  
❌ **Approximate FAISS**: Use `IndexFlatIP` (default)

### Full Reproducibility Mode

Set in `.env`:

```bash
GLOBAL_RANDOM_SEED=42
LLM_TEMPERATURE=0.0
EMBEDDING_DEVICE=cpu
ENABLE_RERANKER=false
VERIFIABLE_CONSISTENCY_ENABLED=false
ENABLE_ARTIFACT_PERSISTENCE=true
EMBEDDING_CACHE_ENABLED=true
```

---

## Testing

### Run Tests

```bash
# Artifact roundtrip
pytest tests/test_artifact_roundtrip.py -v

# Reproducible retrieval
pytest tests/test_reproducible_retrieval.py -v
```

### Test Coverage

**Roundtrip Tests** (`test_artifact_roundtrip.py`):
- Source/span ID determinism
- Save/load equality
- Embedding shape preservation
- Cache hit/miss detection

**Reproducibility Tests** (`test_reproducible_retrieval.py`):
- Chunking determinism
- Embedding determinism with seed
- Top-k retrieval consistency
- Full pipeline reproducibility (build → save → load → query)

---

## Performance Impact

### Cache Hit (Typical)

- **Time**: ~10ms (load from disk)
- **Cost**: $0 (no API calls)
- **Speedup**: 10-50x faster than fresh embedding

### Cache Miss (First Run)

- **Time**: ~5s for 500 chunks (embedding)
- **Cost**: Free (local `sentence-transformers`)
- **Overhead**: ~50ms to save artifacts

### Storage

- **Metadata**: ~2 KB per run
- **Sources**: ~1 KB per source
- **Spans**: ~0.5 KB per span
- **Embeddings**: ~1.5 KB per span (384-dim float32)

**Example**: 500 spans = ~1 MB per run

---

## Integration Points

### Current Integrations

1. **Evidence Builder** (`src/retrieval/evidence_builder.py`):
   - Automatic cache lookup
   - Transparent artifact saving

2. **Seed Control** (`src/utils/seed_control.py`):
   - Called in `app.py` startup
   - Called in `verifiable_pipeline.py` init

3. **Config** (`config.py`):
   - `ARTIFACTS_DIR`, `ENABLE_ARTIFACT_PERSISTENCE`
   - `EMBEDDING_CACHE_ENABLED`, `GLOBAL_RANDOM_SEED`

### Future Integrations

- **Retrieval Cache**: Cache top-k results per query
- **NLI Cache**: Cache consistency scores per (claim, evidence) pair
- **Cross-Session Dedup**: Detect duplicate content across sessions
- **Artifact GC**: Garbage collect old runs (keep N most recent)

---

## Troubleshooting

### Cache not working?

1. Check `config.ENABLE_ARTIFACT_PERSISTENCE == True`
2. Check `config.EMBEDDING_CACHE_ENABLED == True`
3. Verify `ARTIFACTS_DIR` is writable
4. Check logs for "Cache HIT" or "Cache MISS"

### Embeddings differ across runs?

1. Set `GLOBAL_RANDOM_SEED=42`
2. Use `device="cpu"` (avoid GPU precision issues)
3. Ensure same model version: `sentence-transformers==X.Y.Z`

### Disk space issues?

1. Old runs accumulate in `artifacts/`
2. Manually delete old runs: `rm -rf artifacts/<session_id>/<old_run_id>/`
3. Implement automatic cleanup (TODO)

### Different results with same seed?

Possible causes:
- Different `sentence-transformers` version
- Different embedding model
- Different input text (whitespace, unicode normalization)
- Reranker enabled (nondeterministic)
- LLM consistency checks enabled (nondeterministic)

---

## Architecture Reference

See [docs/ARCH_FLOW.md](docs/ARCH_FLOW.md) for full system architecture.

**Related Files**:
- `src/retrieval/artifact_store.py`: Core implementation
- `src/utils/seed_control.py`: Seed management
- `src/retrieval/evidence_builder.py`: Integration point
- `tests/test_artifact_roundtrip.py`: Roundtrip tests
- `tests/test_reproducible_retrieval.py`: Reproducibility tests

---

## References

- **Content-Addressable Storage**: [Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- **Deterministic Embeddings**: [sentence-transformers docs](https://www.sbert.net/)
- **FAISS Reproducibility**: [FAISS seed control](https://github.com/facebookresearch/faiss/wiki/Random-seed)
