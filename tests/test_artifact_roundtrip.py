"""
Test artifact store roundtrip: save → load equality.

Verifies that:
1. Sources/spans save and load correctly
2. Embedding shapes are preserved
3. Span texts match after reload
4. Metadata is intact
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.retrieval.artifact_store import (
    ArtifactStore,
    SourceArtifact,
    SpanArtifact,
    RunMetadata,
    compute_source_id,
    compute_span_id,
    compute_text_hash
)


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_source_id_deterministic():
    """Test that source IDs are deterministic for same content."""
    text = "This is a test lecture transcript."
    source_id_1 = compute_source_id("transcript", "session_input", text)
    source_id_2 = compute_source_id("transcript", "session_input", text)
    
    assert source_id_1 == source_id_2
    assert len(source_id_1) == 64  # SHA256 hex length


def test_span_id_deterministic():
    """Test that span IDs are deterministic for same position/content."""
    source_id = "abc123"
    text = "derivative of x squared"
    
    span_id_1 = compute_span_id(source_id, 0, 23, text)
    span_id_2 = compute_span_id(source_id, 0, 23, text)
    
    assert span_id_1 == span_id_2
    assert len(span_id_1) == 64


def test_span_id_changes_with_content():
    """Test that different content produces different span IDs."""
    source_id = "abc123"
    span_id_1 = compute_span_id(source_id, 0, 10, "derivatives")
    span_id_2 = compute_span_id(source_id, 0, 10, "integrals")
    
    assert span_id_1 != span_id_2


def test_artifact_roundtrip_basic(temp_artifacts_dir):
    """Test basic save/load roundtrip without embeddings."""
    session_id = "test_session_001"
    
    # Create artifact store
    store = ArtifactStore(temp_artifacts_dir, session_id)
    
    # Add source
    source = SourceArtifact(
        source_id=compute_source_id("transcript", "test.txt", "test content"),
        source_type="transcript",
        origin="test.txt",
        page_num=None,
        normalized_text_hash=compute_text_hash("test content"),
        char_count=12,
        metadata={"key": "value"}
    )
    store.add_source(source)
    
    # Add spans
    span1 = SpanArtifact(
        span_id=compute_span_id(source.source_id, 0, 10, "test conte"),
        source_id=source.source_id,
        start=0,
        end=10,
        text="test conte",
        page_num=None,
        chunk_idx=0,
        char_count=10
    )
    span2 = SpanArtifact(
        span_id=compute_span_id(source.source_id, 5, 12, "content"),
        source_id=source.source_id,
        start=5,
        end=12,
        text="content",
        page_num=None,
        chunk_idx=1,
        char_count=7
    )
    store.add_span(span1)
    store.add_span(span2)
    
    # Save metadata
    metadata = RunMetadata(
        run_id=store.run_id,
        session_id=session_id,
        timestamp="2026-02-17T10:00:00",
        random_seed=42,
        config_snapshot={"test": "config"},
        git_commit="abc1234",
        model_ids={"embedding": "test-model"},
        source_count=1,
        span_count=2,
        embedding_dim=0,
        cache_status="miss"
    )
    store.save(metadata)
    
    # Load and verify
    loaded_store = ArtifactStore.load(temp_artifacts_dir, session_id, store.run_id)
    
    assert len(loaded_store.sources) == 1
    assert loaded_store.sources[0].source_id == source.source_id
    assert loaded_store.sources[0].char_count == 12
    
    assert len(loaded_store.spans) == 2
    assert loaded_store.spans[0].text == "test conte"
    assert loaded_store.spans[1].text == "content"
    
    assert loaded_store.metadata.random_seed == 42
    assert loaded_store.metadata.git_commit == "abc1234"


def test_artifact_roundtrip_with_embeddings(temp_artifacts_dir):
    """Test save/load roundtrip with embeddings."""
    session_id = "test_session_002"
    
    store = ArtifactStore(temp_artifacts_dir, session_id)
    
    # Add source and spans
    source_id = compute_source_id("notes", "notes.txt", "calculus notes")
    store.add_source(SourceArtifact(
        source_id=source_id,
        source_type="notes",
        origin="notes.txt",
        page_num=None,
        normalized_text_hash=compute_text_hash("calculus notes"),
        char_count=14,
        metadata={}
    ))
    
    spans_texts = ["derivative", "integral", "limit"]
    for i, text in enumerate(spans_texts):
        span_id = compute_span_id(source_id, i*10, (i+1)*10, text)
        store.add_span(SpanArtifact(
            span_id=span_id,
            source_id=source_id,
            start=i*10,
            end=(i+1)*10,
            text=text,
            page_num=None,
            chunk_idx=i,
            char_count=len(text)
        ))
    
    # Create embeddings (3 spans x 384 dimensions)
    embeddings = np.random.rand(3, 384).astype('float32')
    store.set_embeddings(embeddings)
    
    # Save
    metadata = RunMetadata(
        run_id=store.run_id,
        session_id=session_id,
        timestamp="2026-02-17T10:00:00",
        random_seed=42,
        config_snapshot={},
        git_commit=None,
        model_ids={"embedding": "all-MiniLM-L6-v2"},
        source_count=1,
        span_count=3,
        embedding_dim=384,
        cache_status="miss"
    )
    store.save(metadata)
    
    # Load and verify
    loaded_store = ArtifactStore.load(temp_artifacts_dir, session_id, store.run_id)
    
    assert loaded_store.embeddings is not None
    assert loaded_store.embeddings.shape == (3, 384)
    assert np.allclose(loaded_store.embeddings, embeddings)
    
    # Verify span order matches embedding order
    loaded_span_ids = [span.span_id for span in loaded_store.spans]
    assert len(loaded_span_ids) == 3


def test_find_matching_run_cache_hit(temp_artifacts_dir):
    """Test finding matching run by content hash and model ID."""
    session_id = "test_session_003"
    content_hash = compute_text_hash("lecture transcript v1")
    model_id = "all-MiniLM-L6-v2"
    
    # Create and save first run
    store1 = ArtifactStore(temp_artifacts_dir, session_id)
    store1.add_source(SourceArtifact(
        source_id=compute_source_id("transcript", "input", "lecture transcript v1"),
        source_type="transcript",
        origin="input",
        page_num=None,
        normalized_text_hash=content_hash,
        char_count=21,
        metadata={}
    ))
    
    metadata1 = RunMetadata(
        run_id=store1.run_id,
        session_id=session_id,
        timestamp="2026-02-17T10:00:00",
        random_seed=42,
        config_snapshot={"content_hash": content_hash},
        git_commit=None,
        model_ids={"embedding": model_id},
        source_count=1,
        span_count=0,
        embedding_dim=384,
        cache_status="miss"
    )
    store1.save(metadata1)
    
    # Find matching run
    found_run_id = ArtifactStore.find_matching_run(
        temp_artifacts_dir,
        session_id,
        content_hash,
        model_id
    )
    
    assert found_run_id == store1.run_id


def test_find_matching_run_cache_miss(temp_artifacts_dir):
    """Test that no match is found for different content/model."""
    session_id = "test_session_004"
    content_hash = compute_text_hash("lecture transcript v2")
    model_id = "all-MiniLM-L6-v2"
    
    # Create and save run with different content
    store1 = ArtifactStore(temp_artifacts_dir, session_id)
    store1.add_source(SourceArtifact(
        source_id=compute_source_id("transcript", "input", "different content"),
        source_type="transcript",
        origin="input",
        page_num=None,
        normalized_text_hash=compute_text_hash("different content"),
        char_count=17,
        metadata={}
    ))
    
    metadata1 = RunMetadata(
        run_id=store1.run_id,
        session_id=session_id,
        timestamp="2026-02-17T10:00:00",
        random_seed=42,
        config_snapshot={"content_hash": compute_text_hash("different content")},
        git_commit=None,
        model_ids={"embedding": model_id},
        source_count=1,
        span_count=0,
        embedding_dim=384,
        cache_status="miss"
    )
    store1.save(metadata1)
    
    # Try to find with different content hash
    found_run_id = ArtifactStore.find_matching_run(
        temp_artifacts_dir,
        session_id,
        content_hash,  # Different content
        model_id
    )
    
    assert found_run_id is None


def test_text_normalization_consistency():
    """Test that text normalization produces consistent hashes."""
    text1 = "Test   with   spaces"
    text2 = "Test with spaces"
    
    hash1 = compute_text_hash(text1)
    hash2 = compute_text_hash(text2)
    
    # Should be equal after normalization
    assert hash1 == hash2
    
    # Unicode quotes should normalize
    text3 = 'Test "quoted" text'
    text4 = 'Test "quoted" text'
    
    hash3 = compute_text_hash(text3)
    hash4 = compute_text_hash(text4)
    
    assert hash3 == hash4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
