"""
Evidence Store with FAISS indexing for claim verification.

This module provides a centralized evidence storage and retrieval system
that ensures claims are verified against properly indexed evidence.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using fallback search")


@dataclass
class Evidence:
    """A single evidence chunk with metadata."""
    evidence_id: str
    source_id: str  # e.g., "session_input", "https://youtu.be/abc"
    source_type: str  # "transcript", "notes", "youtube", "article", "external"
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Generate evidence ID if not provided."""
        if not self.evidence_id:
            hash_input = f"{self.source_id}_{self.chunk_index}_{self.char_start}"
            self.evidence_id = hashlib.md5(hash_input.encode()).hexdigest()[:16]


class EvidenceStore:
    """
    Evidence store with FAISS indexing for efficient retrieval.
    
    This class ensures that:
    1. All evidence is properly indexed before verification
    2. Evidence sources are tracked with metadata
    3. Retrieval is efficient via FAISS (or fallback)
    """
    
    def __init__(self, session_id: str, embedding_dim: int = 384):
        """
        Initialize evidence store.
        
        Args:
            session_id: Unique session identifier
            embedding_dim: Dimension of embeddings (default: 384 for sentence-transformers)
        """
        self.session_id = session_id
        self.embedding_dim = embedding_dim
        
        # Storage
        self.evidence: List[Evidence] = []
        self.evidence_by_id: Dict[str, Evidence] = {}
        
        # FAISS index
        self.faiss_index = None
        self.index_built = False
        
        # Statistics
        self.source_counts: Dict[str, int] = {}
        self.total_chars = 0
        
        logger.info(f"Initialized EvidenceStore for session {session_id}")
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add a single evidence item to the store."""
        self.evidence.append(evidence)
        self.evidence_by_id[evidence.evidence_id] = evidence
        
        # Update statistics
        self.total_chars += len(evidence.text)
        source_key = f"{evidence.source_type}:{evidence.source_id}"
        self.source_counts[source_key] = self.source_counts.get(source_key, 0) + 1
        
        # Mark index as stale
        self.index_built = False
    
    def add_evidence_batch(self, evidence_list: List[Evidence]) -> None:
        """Add multiple evidence items at once."""
        for ev in evidence_list:
            self.add_evidence(ev)
    
    def build_index(self, embeddings: Optional[np.ndarray] = None) -> None:
        """
        Build FAISS index from evidence embeddings.
        
        Args:
            embeddings: Optional pre-computed embeddings (n_evidence x embedding_dim)
                       If None, uses evidence.embedding from each item
        """
        if len(self.evidence) == 0:
            raise ValueError("Cannot build index: evidence store is empty")
        
        # Collect embeddings
        if embeddings is None:
            embeddings_list = []
            for ev in self.evidence:
                if ev.embedding is None:
                    raise ValueError(f"Evidence {ev.evidence_id} has no embedding")
                embeddings_list.append(ev.embedding)
            embeddings = np.vstack(embeddings_list)
        
        # Validate shape
        if embeddings.shape[0] != len(self.evidence):
            raise ValueError(
                f"Embedding count mismatch: {embeddings.shape[0]} embeddings "
                f"vs {len(self.evidence)} evidence items"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} "
                f"vs expected {self.embedding_dim}"
            )
        
        # Build FAISS index
        if FAISS_AVAILABLE:
            # Use L2 index for cosine similarity (after normalization)
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.faiss_index.add(embeddings.astype('float32'))
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Fallback: store embeddings for brute-force search
            self.faiss_index = {
                "embeddings": embeddings,
                "type": "fallback"
            }
            logger.warning("FAISS unavailable, using fallback indexing")
        
        self.index_built = True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[tuple[Evidence, float]]:
        """
        Search for evidence similar to query.
        
        Args:
            query_embedding: Query embedding vector (shape: embedding_dim)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of (Evidence, similarity_score) tuples
        """
        if not self.index_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if len(self.evidence) == 0:
            logger.warning("Search called on empty evidence store")
            return []
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        
        if FAISS_AVAILABLE and isinstance(self.faiss_index, faiss.Index):
            # FAISS search
            faiss.normalize_L2(query)
            distances, indices = self.faiss_index.search(query, min(top_k, len(self.evidence)))
            
            # Convert L2 distance to cosine similarity (since we normalized)
            similarities = 1 - (distances[0] / 2.0)  # L2 of normalized vectors -> cosine
            
            results = []
            for idx, sim in zip(indices[0], similarities):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                if sim >= min_similarity:
                    results.append((self.evidence[idx], float(sim)))
            
            return results
        else:
            # Fallback: brute-force cosine similarity
            embeddings = self.faiss_index["embeddings"]
            
            # Normalize query
            query_norm = query / (np.linalg.norm(query) + 1e-9)
            
            # Compute cosine similarities
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
            similarities = embeddings_norm @ query_norm.T
            similarities = similarities.flatten()
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim >= min_similarity:
                    results.append((self.evidence[idx], sim))
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evidence store statistics."""
        return {
            "session_id": self.session_id,
            "num_chunks": len(self.evidence),
            "total_chars": self.total_chars,
            "avg_chunk_chars": self.total_chars // len(self.evidence) if self.evidence else 0,
            "num_sources": len(self.source_counts),
            "sources": self.source_counts,
            "index_built": self.index_built,
            "faiss_index_size": self.faiss_index.ntotal if FAISS_AVAILABLE and isinstance(self.faiss_index, faiss.Index) else len(self.evidence),
            "faiss_available": FAISS_AVAILABLE
        }
    
    def validate(self, min_chars: int = 500) -> tuple[bool, str]:
        """
        Validate that evidence store is ready for verification.
        
        Args:
            min_chars: Minimum total characters required
        
        Returns:
            (is_valid, error_message)
        """
        if len(self.evidence) == 0:
            return False, "Evidence store has 0 chunks. Cannot run verification."
        
        if self.total_chars < min_chars:
            return False, f"Evidence store has only {self.total_chars} chars (minimum: {min_chars})"
        
        if not self.index_built:
            return False, "FAISS index not built. Call build_index() first."
        
        if FAISS_AVAILABLE and isinstance(self.faiss_index, faiss.Index):
            if self.faiss_index.ntotal == 0:
                return False, "FAISS index is empty"
        
        return True, "Evidence store is valid"


def validate_evidence_store(store: EvidenceStore, min_chars: int = 500) -> None:
    """
    Validate evidence store and raise error if invalid.
    
    Args:
        store: Evidence store to validate
        min_chars: Minimum total characters required
    
    Raises:
        ValueError: If store is invalid
    """
    is_valid, error_msg = store.validate(min_chars=min_chars)
    if not is_valid:
        raise ValueError(f"Evidence store validation failed: {error_msg}")
    
    logger.info(f"✓ Evidence store validated: {len(store.evidence)} chunks, {store.total_chars} chars")
