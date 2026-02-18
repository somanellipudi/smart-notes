"""
Embedding provider for dense retrieval and optional reranking.
"""

import logging
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Provide dense embeddings and optional cross-encoder reranking."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        encoder: Optional[object] = None,
        reranker: Optional[object] = None
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.normalize = normalize if normalize is not None else config.EMBEDDING_NORMALIZE
        self._encoder = encoder
        self._reranker = reranker
        self._is_e5 = "e5" in (self.model_name or "").lower()

    @classmethod
    def from_config(cls) -> "EmbeddingProvider":
        return cls()

    def _load_encoder(self) -> object:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for dense retrieval"
                ) from exc
            self._encoder = SentenceTransformer(
                self.model_name,
                device=self.device if self.device else None
            )
        return self._encoder

    def _load_reranker(self) -> Optional[object]:
        if not config.ENABLE_RERANKER:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for reranking"
                ) from exc
            self._reranker = CrossEncoder(
                config.RERANKER_MODEL_NAME,
                device=self.device if self.device else None
            )
        return self._reranker

    def _prefix_texts(self, texts: List[str], is_query: bool) -> List[str]:
        if not self._is_e5:
            return texts
        prefix = "query: " if is_query else "passage: "
        return [f"{prefix}{text}" for text in texts]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        encoder = self._load_encoder()
        prepared = self._prefix_texts(texts, is_query=False)
        embeddings = encoder.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return np.atleast_2d(embeddings).astype("float32")

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, 1), dtype="float32")
        encoder = self._load_encoder()
        prepared = self._prefix_texts(queries, is_query=True)
        embeddings = encoder.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        return np.atleast_2d(embeddings).astype("float32")

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        reranker = self._load_reranker()
        if reranker is None:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = reranker.predict(pairs)
        return [float(score) for score in scores]
