"""
Deterministic, persistent Evidence Artifact Store.

This module provides content-addressable storage for:
- Sources (original documents/transcripts)
- Spans (text chunks with positions)
- Embeddings (dense vectors)
- Run metadata (config, seeds, model versions)

Key features:
- Stable IDs: SHA256-based hashing for reproducibility
- Per-run folders: artifacts/<session_id>/<run_id>/
- Format: JSONL for text, NPZ for embeddings
- Cache-aware: skip re-embedding identical content
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SourceArtifact:
    """A source document/transcript with stable ID."""
    source_id: str  # sha256(source_type + origin + normalized_text)
    source_type: str  # "transcript", "notes", "youtube", "article", "external"
    origin: str  # filename, URL, or "session_input"
    page_num: Optional[int]
    normalized_text_hash: str  # sha256(normalized_text)
    char_count: int
    metadata: Dict[str, Any]


@dataclass
class SpanArtifact:
    """A text chunk/span with stable ID."""
    span_id: str  # sha256(source_id + start + end + normalized_text)
    source_id: str
    start: int
    end: int
    text: str
    page_num: Optional[int]
    chunk_idx: int
    char_count: int


@dataclass
class RunMetadata:
    """Metadata for a single artifact run."""
    run_id: str  # timestamp + short hash
    session_id: str
    timestamp: str  # ISO format
    random_seed: int
    config_snapshot: Dict[str, Any]
    git_commit: Optional[str]
    model_ids: Dict[str, str]  # {"embedding": "all-MiniLM-L6-v2", "llm": "gpt-4"}
    source_count: int
    span_count: int
    embedding_dim: int
    cache_status: str  # "hit" | "miss" | "partial"


def _normalize_text(text: str) -> str:
    """Normalize text for hashing (whitespace, unicode)."""
    import re
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', text.strip())
    # Normalize unicode quotes/dashes
    replacements = [
        ('\u2018', "'"), ('\u2019', "'"),
        ('\u201c', '"'), ('\u201d', '"'),
        ('\u2014', '-'), ('\u2013', '-')
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return normalized


def compute_source_id(source_type: str, origin: str, text: str) -> str:
    """Compute stable source ID from content."""
    normalized = _normalize_text(text)
    hash_input = f"{source_type}|{origin}|{normalized}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def compute_span_id(source_id: str, start: int, end: int, text: str) -> str:
    """Compute stable span ID from position and content."""
    normalized = _normalize_text(text)
    hash_input = f"{source_id}|{start}|{end}|{normalized}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def generate_run_id(session_id: str) -> str:
    """Generate unique run ID: timestamp + short hash."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha256(f"{session_id}{timestamp}".encode()).hexdigest()[:8]
    return f"{timestamp}_{short_hash}"


class ArtifactStore:
    """
    Persistent storage for evidence artifacts with content-addressable IDs.
    
    Directory structure:
        artifacts/<session_id>/<run_id>/
            metadata.json
            sources.jsonl
            spans.jsonl
            embeddings.npz
    """
    
    def __init__(self, artifacts_dir: Path, session_id: str, run_id: Optional[str] = None):
        """
        Initialize artifact store.
        
        Args:
            artifacts_dir: Base directory for artifacts
            session_id: Unique session identifier
            run_id: Optional run ID (generated if None)
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.session_id = session_id
        self.run_id = run_id or generate_run_id(session_id)
        
        self.run_dir = self.artifacts_dir / session_id / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources: List[SourceArtifact] = []
        self.spans: List[SpanArtifact] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[RunMetadata] = None
        
        logger.info(f"Initialized ArtifactStore: {self.run_dir}")
    
    def add_source(self, source: SourceArtifact) -> None:
        """Add a source artifact."""
        self.sources.append(source)
    
    def add_span(self, span: SpanArtifact) -> None:
        """Add a span artifact."""
        self.spans.append(span)
    
    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Set embeddings array (must match span order)."""
        if embeddings.shape[0] != len(self.spans):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) != span count ({len(self.spans)})"
            )
        self.embeddings = embeddings
    
    def save(self, metadata: RunMetadata) -> None:
        """
        Save all artifacts to disk.
        
        Args:
            metadata: Run metadata to save
        """
        self.metadata = metadata
        
        # Save metadata
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Save sources
        sources_path = self.run_dir / "sources.jsonl"
        with open(sources_path, 'w', encoding='utf-8') as f:
            for source in self.sources:
                f.write(json.dumps(asdict(source)) + '\n')
        logger.info(f"Saved {len(self.sources)} sources: {sources_path}")
        
        # Save spans
        spans_path = self.run_dir / "spans.jsonl"
        with open(spans_path, 'w', encoding='utf-8') as f:
            for span in self.spans:
                f.write(json.dumps(asdict(span)) + '\n')
        logger.info(f"Saved {len(self.spans)} spans: {spans_path}")
        
        # Save embeddings
        if self.embeddings is not None:
            embeddings_path = self.run_dir / "embeddings.npz"
            span_ids = [span.span_id for span in self.spans]
            np.savez_compressed(
                embeddings_path,
                embeddings=self.embeddings,
                span_ids=span_ids
            )
            logger.info(f"Saved embeddings: {embeddings_path} ({self.embeddings.shape})")
    
    @classmethod
    def load(cls, artifacts_dir: Path, session_id: str, run_id: str) -> "ArtifactStore":
        """
        Load artifacts from disk.
        
        Args:
            artifacts_dir: Base directory for artifacts
            session_id: Session identifier
            run_id: Run identifier
        
        Returns:
            Loaded ArtifactStore instance
        """
        store = cls(artifacts_dir, session_id, run_id)
        
        # Load metadata
        metadata_path = store.run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
            store.metadata = RunMetadata(**metadata_dict)
        
        # Load sources
        sources_path = store.run_dir / "sources.jsonl"
        if sources_path.exists():
            with open(sources_path, 'r', encoding='utf-8') as f:
                for line in f:
                    source_dict = json.loads(line)
                    store.sources.append(SourceArtifact(**source_dict))
        
        # Load spans
        spans_path = store.run_dir / "spans.jsonl"
        if spans_path.exists():
            with open(spans_path, 'r', encoding='utf-8') as f:
                for line in f:
                    span_dict = json.loads(line)
                    store.spans.append(SpanArtifact(**span_dict))
        
        # Load embeddings
        embeddings_path = store.run_dir / "embeddings.npz"
        if embeddings_path.exists():
            data = np.load(embeddings_path, allow_pickle=False)
            store.embeddings = data['embeddings']
            # Verify span_ids match
            loaded_span_ids = data['span_ids'].tolist()
            current_span_ids = [span.span_id for span in store.spans]
            if loaded_span_ids != current_span_ids:
                logger.warning("Loaded span_ids don't match current spans order")
        
        logger.info(
            f"Loaded artifacts: {len(store.sources)} sources, "
            f"{len(store.spans)} spans, embeddings={store.embeddings.shape if store.embeddings is not None else None}"
        )
        return store
    
    @classmethod
    def find_matching_run(
        cls,
        artifacts_dir: Path,
        session_id: str,
        content_hash: str,
        model_id: str
    ) -> Optional[str]:
        """
        Find existing run with matching content hash and model ID.
        
        Args:
            artifacts_dir: Base directory for artifacts
            session_id: Session identifier
            content_hash: SHA256 hash of normalized input content
            model_id: Embedding model identifier
        
        Returns:
            run_id if match found, None otherwise
        """
        session_dir = Path(artifacts_dir) / session_id
        if not session_dir.exists():
            return None
        
        for run_dir in session_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check if content hash and model match
                config_snapshot = metadata.get("config_snapshot", {})
                stored_content_hash = config_snapshot.get("content_hash")
                stored_model_id = metadata.get("model_ids", {}).get("embedding")
                
                if stored_content_hash == content_hash and stored_model_id == model_id:
                    logger.info(f"Found matching run: {run_dir.name}")
                    return run_dir.name
            
            except Exception as e:
                logger.warning(f"Error reading metadata from {run_dir}: {e}")
                continue
        
        return None
    
    def get_span_texts(self) -> List[str]:
        """Get list of span texts in order."""
        return [span.text for span in self.spans]
    
    def get_span_ids(self) -> List[str]:
        """Get list of span IDs in order."""
        return [span.span_id for span in self.spans]


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def create_config_snapshot(content_hash: str, seed: int, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create snapshot of relevant config for reproducibility.
    
    Args:
        content_hash: Hash of input content
        seed: Random seed used
        additional_config: Additional config values to capture
    
    Returns:
        Config snapshot dict
    """
    import config
    
    snapshot = {
        "content_hash": content_hash,
        "random_seed": seed,
        "embedding_model": getattr(config, "EMBEDDING_MODEL_NAME", "unknown"),
        "embedding_device": getattr(config, "EMBEDDING_DEVICE", "cpu"),
        "embedding_normalize": getattr(config, "EMBEDDING_NORMALIZE", True),
        "chunk_size": 500,  # From evidence_builder.py
        "chunk_overlap": 50,
        "min_chunk_size": 100,
    }
    
    if additional_config:
        snapshot.update(additional_config)
    
    return snapshot
