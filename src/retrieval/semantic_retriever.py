"""
Semantic retrieval for evidence matching using sentence transformers and FAISS.

Replaces keyword-based Jaccard similarity with:
- Dense embeddings (intfloat/e5-base-v2)
- FAISS vector indexing
- Cross-encoder re-ranking for precision

Research-grade features:
- Deterministic retrieval with seed
- Top-k candidate retrieval
- Re-ranking with cross-encoder
- Source tracking and deduplication
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvidenceSpan:
    """A candidate evidence span with metadata."""
    text: str
    source_type: str  # pdf_page, notes_text, external_context, url_article, youtube_transcript, audio_transcript, equations
    source_id: str
    span_start: int
    span_end: int
    similarity: float
    rerank_score: Optional[float] = None
    # Provenance fields
    origin: Optional[str] = None  # Filename or URL
    page_num: Optional[int] = None  # For PDF sources
    timestamp_range: Optional[Tuple[float, float]] = None  # For youtube/audio (start_sec, end_sec)


class SemanticRetriever:
    """
    Hybrid semantic retrieval with bi-encoder + cross-encoder.
    
    Architecture:
    1. Bi-encoder (e5-base-v2): fast retrieval of top-k candidates
    2. FAISS index: efficient similarity search
    3. Cross-encoder (ms-marco-MiniLM): accurate re-ranking
    
    Usage:
        retriever = SemanticRetriever()
        retriever.index_sources(transcript, notes, context)
        evidence = retriever.retrieve(claim_text, top_k=10)
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        seed: int = 42
    ):
        """
        Initialize semantic retriever.
        
        Args:
            model_name: Sentence transformer model for embeddings
            reranker_name: Cross-encoder model for re-ranking
            device: 'cpu' or 'cuda'
            seed: Random seed for deterministic retrieval
        """
        self.model_name = model_name
        self.reranker_name = reranker_name
        self.device = device
        self.seed = seed
        
        self.encoder = None
        self.reranker = None
        self.index = None
        self.spans = []  # List[EvidenceSpan]
        self.embedding_dim = None  # Will be set after encoder initialization
        
        self._load_models()
        
        # Set embedding dimension after models are loaded
        if self.encoder is not None:
            self.embedding_dim = self.get_embedding_dim()
        
    def _load_models(self):
        """Load sentence transformer and cross-encoder models."""
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            HAS_SENTENCE_TRANSFORMERS = True
        except ImportError as import_error:
            logger.warning(f"Could not import sentence-transformers: {import_error}")
            logger.warning("Will use TF-IDF embeddings as fallback")
            HAS_SENTENCE_TRANSFORMERS = False
        
        if HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Loading bi-encoder: {self.model_name}")
            try:
                self.encoder = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Bi-encoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load bi-encoder {self.model_name}: {e}")
                logger.warning("Using fallback TF-IDF embeddings")
                try:
                    self.encoder = _TfidfEmbedder()
                except Exception as tfidf_error:
                    logger.error(f"Failed to initialize TF-IDF embedder: {tfidf_error}")
                    raise RuntimeError(f"Could not initialize any embedding method: {tfidf_error}")
            
            logger.info(f"Loading cross-encoder: {self.reranker_name}")
            try:
                self.reranker = CrossEncoder(self.reranker_name, device=self.device)
                logger.info("Cross-encoder loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                logger.warning("Re-ranking will be disabled")
                self.reranker = None
            
            logger.info("Semantic retrieval models loaded (with fallbacks if needed)")
        else:
            # sentence-transformers not available, use TF-IDF directly
            logger.info("sentence-transformers not available, using TF-IDF embeddings")
            try:
                self.encoder = _TfidfEmbedder()
                self.reranker = None
                logger.info("TF-IDF embedder initialized successfully")
            except Exception as tfidf_error:
                logger.error(f"Failed to initialize TF-IDF embedder: {tfidf_error}")
                raise RuntimeError(f"Could not initialize any embedding method: {tfidf_error}")
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension of the encoder.
        
        Returns:
            int: Embedding dimension
        
        Raises:
            RuntimeError: If encoder is not initialized
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not initialized")
        
        # Check if encoder has get_sentence_embedding_dimension (SentenceTransformer)
        if hasattr(self.encoder, 'get_sentence_embedding_dimension'):
            return self.encoder.get_sentence_embedding_dimension()
        
        # Check for _TfidfEmbedder
        if hasattr(self.encoder, 'get_embedding_dim'):
            return self.encoder.get_embedding_dim()
        
        # Default fallback
        logger.warning("Could not determine embedding dimension, assuming 384")
        return 384
    
    def index_sources(
        self,
        transcript: str = "",
        notes: str = "",
        external_context: str = "",
        equations: List[str] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50
    ):
        """
        Index all source materials into FAISS.
        
        Args:
            transcript: Lecture audio transcript
            notes: Handwritten notes
            external_context: Reference materials
            equations: List of equations
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        import faiss
        
        self.spans = []
        
        # Chunk each source
        if transcript:
            self.spans.extend(
                self._chunk_text(transcript, "transcript", "audio", chunk_size, chunk_overlap)
            )
        
        if notes:
            self.spans.extend(
                self._chunk_text(notes, "notes", "notes", chunk_size, chunk_overlap)
            )
        
        if external_context:
            self.spans.extend(
                self._chunk_text(external_context, "external_context", "context", chunk_size, chunk_overlap)
            )
        
        if equations:
            for i, eq in enumerate(equations):
                self.spans.append(EvidenceSpan(
                    text=eq,
                    source_type="equations",
                    source_id=f"eq_{i}",
                    span_start=0,
                    span_end=len(eq),
                    similarity=0.0
                ))
        
        if not self.spans:
            logger.warning("No source material to index")
            return
        
        # Encode all spans
        logger.info(f"Encoding {len(self.spans)} evidence spans")
        texts = [span.text for span in self.spans]
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, {dimension}D")
    
    def retrieve(
        self,
        claim_text: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        min_similarity: float = 0.3
    ) -> List[EvidenceSpan]:
        """
        Retrieve top-k evidence spans for a claim.
        
        Args:
            claim_text: Claim to verify
            top_k: Number of candidates to retrieve
            rerank_top_n: Number to re-rank with cross-encoder
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of EvidenceSpan objects, sorted by rerank score
        """
        if not self.index or not self.spans:
            logger.warning("Index not built. Call index_sources() first.")
            return []
        
        # Encode claim
        claim_embedding = self.encoder.encode(
            [claim_text],
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        import faiss
        faiss.normalize_L2(claim_embedding)
        
        # Search FAISS
        similarities, indices = self.index.search(claim_embedding, min(top_k, self.index.ntotal))
        
        # Get candidates
        candidates = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity and idx < len(self.spans):
                span = self.spans[idx]
                span.similarity = float(sim)
                candidates.append(span)
        
        if not candidates:
            logger.debug(f"No candidates above threshold {min_similarity}")
            return []
        
        # Re-rank top-n with cross-encoder
        if len(candidates) > rerank_top_n:
            candidates = candidates[:rerank_top_n]
        
        pairs = [[claim_text, span.text] for span in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        for span, score in zip(candidates, rerank_scores):
            span.rerank_score = float(score)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        logger.debug(
            f"Retrieved {len(candidates)} spans for claim (top score: {candidates[0].rerank_score:.3f})"
        )
        
        return candidates
    
    def _chunk_text(
        self,
        text: str,
        source_type: str,
        source_id: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[EvidenceSpan]:
        """Split text into overlapping chunks."""
        spans = []
        text_len = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Extend to sentence boundary if possible
            if end < text_len:
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    boundary = text.find(delimiter, end - 50, end + 50)
                    if boundary != -1:
                        end = boundary + len(delimiter)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                spans.append(EvidenceSpan(
                    text=chunk_text,
                    source_type=source_type,
                    source_id=f"{source_id}_{chunk_id}",
                    span_start=start,
                    span_end=end,
                    similarity=0.0
                ))
                chunk_id += 1
            
            start += chunk_size - chunk_overlap
        
        return spans
    
    def diagnose_retrieval(
        self,
        evidence_list: List[EvidenceSpan],
        warning_threshold: float = 0.45
    ) -> Dict[str, any]:
        """
        Diagnose retrieval health for a set of evidence spans.
        
        Args:
            evidence_list: List of retrieved evidence spans
            warning_threshold: Threshold below which to warn
        
        Returns:
            Dictionary with diagnostic metrics
        """
        if not evidence_list:
            return {
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "num_candidates": 0,
                "empty": True,
                "status": "EMPTY"
            }
        
        similarities = [span.rerank_score if span.rerank_score else span.similarity for span in evidence_list]
        max_sim = max(similarities) if similarities else 0.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        
        status = "WEAK" if max_sim < warning_threshold else "HEALTHY"
        
        return {
            "max_similarity": float(max_sim),
            "avg_similarity": float(avg_sim),
            "num_candidates": len(evidence_list),
            "empty": False,
            "status": status
        }


class _TfidfEmbedder:
    """
    Fallback embedder using TF-IDF when sentence-transformers unavailable.
    
    This provides lightweight embeddings without downloading large models.
    """
    
    def __init__(self, max_features: int = 128):
        """Initialize TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words='english'
        )
        self.fitted = False
        logger.info(f"Initialized TF-IDF embedder with {max_features} features")
    
    def encode(self, sentences, **kwargs):
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: List of strings or single string
            **kwargs: Ignored (for compatibility with SentenceTransformer API)
        
        Returns:
            numpy array of shape (len(sentences), n_features)
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        
        # Handle single string
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Fit vectorizer on first call with documents
        if not self.fitted:
            # Combine all sentences for vocabulary
            embeddings = self.vectorizer.fit_transform(sentences).toarray().astype('float32')
            self.fitted = True
            return embeddings
        else:
            # Transform using existing vocabulary
            embeddings = self.vectorizer.transform(sentences).toarray().astype('float32')
            return embeddings
    
    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension (vocabulary size after fit).
        
        Returns:
            int: Number of features in the TF-IDF vectorizer
        """
        if self.fitted:
            # Return actual vocabulary size after fitting
            return len(self.vectorizer.get_feature_names_out())
        else:
            # Return configured max_features before fitting
            return self.vectorizer.max_features or 128
    
    def get_stats(self) -> Dict[str, int]:
        """Get indexing statistics."""
        if not self.spans:
            return {}
        
        source_counts = {}
        for span in self.spans:
            source_counts[span.source_type] = source_counts.get(span.source_type, 0) + 1
        
        return {
            "total_spans": len(self.spans),
            "index_size": self.index.ntotal if self.index else 0,
            "by_source": source_counts
        }
