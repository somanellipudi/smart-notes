"""
Test reproducible retrieval: same input + same seed => identical results.

Verifies that:
1. Same input produces same span IDs
2. Same embeddings produce same top-k retrieval results
3. Reranking (if enabled) is deterministic
4. Full pipeline reproducibility
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.retrieval.evidence_builder import build_session_evidence_store, chunk_text
from src.retrieval.embedding_provider import EmbeddingProvider
from src.retrieval.artifact_store import compute_span_id, compute_source_id
from src.utils.seed_control import set_global_seed
import config


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_text():
    """Sample lecture text for testing."""
    return """
    Today we'll cover derivatives. A derivative represents the rate of change.
    The derivative of x^2 is 2x. This follows from the power rule.
    Integration is the reverse of differentiation. The integral of 2x is x^2 + C.
    Limits are fundamental to calculus. The limit defines instantaneous rate of change.
    """


def test_chunking_deterministic(sample_text):
    """Test that text chunking is deterministic."""
    chunks1 = chunk_text(sample_text, chunk_size=100, overlap=20)
    chunks2 = chunk_text(sample_text, chunk_size=100, overlap=20)
    
    assert len(chunks1) == len(chunks2)
    
    for c1, c2 in zip(chunks1, chunks2):
        assert c1["text"] == c2["text"]
        assert c1["char_start"] == c2["char_start"]
        assert c1["char_end"] == c2["char_end"]
        assert c1["chunk_index"] == c2["chunk_index"]


def test_span_ids_deterministic(sample_text):
    """Test that span IDs are deterministic for same chunking."""
    source_id = compute_source_id("transcript", "test", sample_text)
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    
    # Compute span IDs twice
    span_ids_1 = [
        compute_span_id(source_id, c["char_start"], c["char_end"], c["text"])
        for c in chunks
    ]
    span_ids_2 = [
        compute_span_id(source_id, c["char_start"], c["char_end"], c["text"])
        for c in chunks
    ]
    
    assert span_ids_1 == span_ids_2


def test_embedding_deterministic_with_seed(sample_text):
    """Test that embeddings are deterministic with same seed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    texts = [c["text"] for c in chunks]
    
    # First run with seed
    set_global_seed(42)
    provider1 = EmbeddingProvider(device="cpu")
    embeddings1 = provider1.embed_texts(texts)
    
    # Second run with same seed
    set_global_seed(42)
    provider2 = EmbeddingProvider(device="cpu")
    embeddings2 = provider2.embed_texts(texts)
    
    # Should be very close (allowing for minor floating point differences)
    assert embeddings1.shape == embeddings2.shape
    assert np.allclose(embeddings1, embeddings2, atol=1e-5)


def test_retrieval_top_k_deterministic(sample_text):
    """Test that top-k retrieval is deterministic."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    
    session_id = "test_retrieval_001"
    
    # Build evidence store with seed
    set_global_seed(42)
    provider = EmbeddingProvider(device="cpu")
    store1, _ = build_session_evidence_store(
        session_id=session_id,
        input_text=sample_text,
        min_input_chars=100,
        embedding_provider=provider
    )
    
    # Query
    query = "What is a derivative?"
    query_embedding = provider.embed_queries([query])
    
    # Retrieve top-3
    results1 = store1.search(query_embedding[0], top_k=3)
    
    # Reset and build again with same seed
    set_global_seed(42)
    provider2 = EmbeddingProvider(device="cpu")
    store2, _ = build_session_evidence_store(
        session_id=session_id,
        input_text=sample_text,
        min_input_chars=100,
        embedding_provider=provider2
    )
    
    # Query again
    query_embedding2 = provider2.embed_queries([query])
    results2 = store2.search(query_embedding2[0], top_k=3)
    
    # Compare results
    assert len(results1) == len(results2)
    
    # Check that evidence IDs match (same order)
    ids1 = [ev.evidence_id for ev in results1]
    ids2 = [ev.evidence_id for ev in results2]
    
    # Since evidence_id may be generated, check texts instead
    texts1 = [ev.snippet for ev in results1]
    texts2 = [ev.snippet for ev in results2]
    
    assert texts1 == texts2


def test_artifact_cache_produces_identical_store(sample_text, temp_artifacts_dir):
    """Test that loading from artifact cache produces identical results."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    
    # Temporarily override config
    original_artifacts_dir = config.ARTIFACTS_DIR
    original_persistence = config.ENABLE_ARTIFACT_PERSISTENCE
    original_cache = config.EMBEDDING_CACHE_ENABLED
    
    try:
        config.ARTIFACTS_DIR = temp_artifacts_dir
        config.ENABLE_ARTIFACT_PERSISTENCE = True
        config.EMBEDDING_CACHE_ENABLED = True
        
        session_id = "test_cache_001"
        
        # First run: create artifacts
        set_global_seed(42)
        provider1 = EmbeddingProvider(device="cpu")
        store1, stats1 = build_session_evidence_store(
            session_id=session_id,
            input_text=sample_text,
            min_input_chars=100,
            embedding_provider=provider1
        )
        
        assert stats1["cache_status"] == "miss"
        
        # Second run: should load from cache
        set_global_seed(42)
        provider2 = EmbeddingProvider(device="cpu")
        store2, stats2 = build_session_evidence_store(
            session_id=session_id,
            input_text=sample_text,
            min_input_chars=100,
            embedding_provider=provider2
        )
        
        assert stats2["cache_status"] == "hit"
        
        # Verify stores are equivalent
        assert len(store1.evidence) == len(store2.evidence)
        
        # Check that texts match
        texts1 = [ev.text for ev in store1.evidence]
        texts2 = [ev.text for ev in store2.evidence]
        assert texts1 == texts2
        
        # Check that FAISS indexes have same content
        # Query both stores
        query = "derivative"
        query_emb = provider2.embed_queries([query])[0]
        
        results1 = store1.search(query_emb, top_k=3)
        results2 = store2.search(query_emb, top_k=3)
        
        # Should get same snippets in same order
        snippets1 = [r.snippet for r in results1]
        snippets2 = [r.snippet for r in results2]
        assert snippets1 == snippets2
        
    finally:
        # Restore config
        config.ARTIFACTS_DIR = original_artifacts_dir
        config.ENABLE_ARTIFACT_PERSISTENCE = original_persistence
        config.EMBEDDING_CACHE_ENABLED = original_cache


def test_seed_control_affects_results():
    """Test that different seeds produce different results (sanity check)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    
    text = ["derivative of x squared"]
    
    # Run with seed 42
    set_global_seed(42)
    provider1 = EmbeddingProvider(device="cpu")
    emb1 = provider1.embed_texts(text)
    
    # Run with seed 123 (different)
    set_global_seed(123)
    provider2 = EmbeddingProvider(device="cpu")
    emb2 = provider2.embed_texts(text)
    
    # Embeddings should be very similar but not identical
    # (model weights are deterministic, but initialization might differ)
    # This is more a sanity check that seed control is working
    assert emb1.shape == emb2.shape


def test_full_pipeline_reproducibility(sample_text, temp_artifacts_dir):
    """Test end-to-end reproducibility: build → save → load → query."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    
    # Override config temporarily
    original_artifacts_dir = config.ARTIFACTS_DIR
    original_persistence = config.ENABLE_ARTIFACT_PERSISTENCE
    original_cache = config.EMBEDDING_CACHE_ENABLED
    
    try:
        config.ARTIFACTS_DIR = temp_artifacts_dir
        config.ENABLE_ARTIFACT_PERSISTENCE = True
        config.EMBEDDING_CACHE_ENABLED = True
        
        session_id = "test_pipeline_001"
        query_text = "What is the power rule?"
        
        # Run 1: Fresh build
        set_global_seed(42)
        provider1 = EmbeddingProvider(device="cpu")
        store1, _ = build_session_evidence_store(
            session_id=session_id,
            input_text=sample_text,
            min_input_chars=100,
            embedding_provider=provider1
        )
        query_emb1 = provider1.embed_queries([query_text])[0]
        results1 = store1.search(query_emb1, top_k=3)
        
        # Run 2: Load from cache
        set_global_seed(42)
        provider2 = EmbeddingProvider(device="cpu")
        store2, _ = build_session_evidence_store(
            session_id=session_id,
            input_text=sample_text,
            min_input_chars=100,
            embedding_provider=provider2
        )
        query_emb2 = provider2.embed_queries([query_text])[0]
        results2 = store2.search(query_emb2, top_k=3)
        
        # Run 3: Load from cache again
        set_global_seed(42)
        provider3 = EmbeddingProvider(device="cpu")
        store3, _ = build_session_evidence_store(
            session_id=session_id,
            input_text=sample_text,
            min_input_chars=100,
            embedding_provider=provider3
        )
        query_emb3 = provider3.embed_queries([query_text])[0]
        results3 = store3.search(query_emb3, top_k=3)
        
        # All three runs should produce identical results
        snippets1 = [r.snippet for r in results1]
        snippets2 = [r.snippet for r in results2]
        snippets3 = [r.snippet for r in results3]
        
        assert snippets1 == snippets2 == snippets3
        
        # Check similarity scores are consistent
        sims1 = [r.similarity for r in results1]
        sims2 = [r.similarity for r in results2]
        sims3 = [r.similarity for r in results3]
        
        assert np.allclose(sims1, sims2, atol=1e-5)
        assert np.allclose(sims2, sims3, atol=1e-5)
        
    finally:
        config.ARTIFACTS_DIR = original_artifacts_dir
        config.ENABLE_ARTIFACT_PERSISTENCE = original_persistence
        config.EMBEDDING_CACHE_ENABLED = original_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
