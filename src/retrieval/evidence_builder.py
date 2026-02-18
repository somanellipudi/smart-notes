"""
Evidence ingestion and store building for verification pipeline.

This module provides functions to build evidence stores from various sources:
- Session input (transcript/notes)
- External context
- URLs (YouTube videos, articles)

Now integrates with ArtifactStore for deterministic, persistent evidence.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

from src.retrieval.evidence_store import Evidence, EvidenceStore
from src.retrieval.url_ingest import ingest_urls, chunk_url_sources, get_url_ingestion_summary
from src.retrieval.artifact_store import (
    ArtifactStore, SourceArtifact, SpanArtifact, RunMetadata,
    compute_source_id, compute_span_id, compute_text_hash,
    create_config_snapshot, get_git_commit
)
from src.retrieval.online_evidence_integration import create_integrator
import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.retrieval.embedding_provider import EmbeddingProvider


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping segments.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size (discard smaller chunks)

    Returns:
        List of dicts with keys: text, char_start, char_end, chunk_index
    """
    if not text or len(text) < min_chunk_size:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_value = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "? ", "! ", "\n\n", "\n"]:
                last_sep = chunk_text_value.rfind(sep)
                if last_sep > chunk_size * 0.7:
                    chunk_text_value = chunk_text_value[:last_sep + 1]
                    end = start + last_sep + 1
                    break

        chunk_text_value = chunk_text_value.strip()

        if len(chunk_text_value) >= min_chunk_size:
            chunks.append({
                "text": chunk_text_value,
                "char_start": start,
                "char_end": end,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        start = end - overlap

    return chunks


def build_session_evidence_store(
    session_id: str,
    input_text: str,
    external_context: str = "",
    equations: List[str] = None,
    urls: List[str] = None,
    min_input_chars: int = None,
    embedding_provider: Optional["EmbeddingProvider"] = None
) -> tuple[EvidenceStore, Dict[str, Any]]:
    """
    Build evidence store from session inputs with artifact persistence.

    This function:
    1. Validates input text length
    2. Computes content hash for cache lookup
    3. Checks for cached artifacts (if enabled)
    4. If cache hit: loads artifacts and reuses embeddings
    5. If cache miss: chunks text, computes embeddings, saves artifacts
    6. Creates Evidence objects with proper metadata
    7. Builds FAISS index

    Args:
        session_id: Unique session identifier
        input_text: Main input text (transcript/notes)
        external_context: Optional external reference material
        equations: Optional list of equations
        urls: Optional list of URLs to ingest
        min_input_chars: Minimum input characters (default: from config)
        embedding_provider: Optional embedding provider for dense retrieval

    Returns:
        (evidence_store, ingestion_summary)

    Raises:
        ValueError: If input validation fails
    """
    if min_input_chars is None:
        min_input_chars = getattr(config, "MIN_INPUT_CHARS_FOR_VERIFICATION", 500)

    logger.info(f"Building evidence store for session {session_id}")
    
    # Compute content hash for cache lookup
    combined_text = input_text + (external_context or "")
    content_hash = compute_text_hash(combined_text)
    model_id = embedding_provider.model_name if embedding_provider else "none"
    
    # Try to load from artifact store if enabled
    artifact_store = None
    cache_status = "disabled"
    
    if config.ENABLE_ARTIFACT_PERSISTENCE and config.EMBEDDING_CACHE_ENABLED:
        matching_run_id = ArtifactStore.find_matching_run(
            config.ARTIFACTS_DIR,
            session_id,
            content_hash,
            model_id
        )
        
        if matching_run_id:
            logger.info(f"Cache HIT: Found matching artifacts for run {matching_run_id}")
            try:
                artifact_store = ArtifactStore.load(
                    config.ARTIFACTS_DIR,
                    session_id,
                    matching_run_id
                )
                cache_status = "hit"
                
                # Convert artifacts to Evidence objects
                return _build_store_from_artifacts(
                    session_id,
                    artifact_store,
                    cache_status
                )
            except Exception as e:
                logger.warning(f"Failed to load cached artifacts: {e}")
                cache_status = "miss"
        else:
            logger.info("Cache MISS: No matching artifacts found")
            cache_status = "miss"

    store = EvidenceStore(session_id=session_id)

    stats = {
        "session_id": session_id,
        "input_chars": len(input_text),
        "external_chars": len(external_context) if external_context else 0,
        "equations_count": len(equations) if equations else 0,
        "urls_count": len(urls) if urls else 0,
        "chunks_added": 0,
        "url_ingestion": None,
        "embedding_model": None,
        "embedding_dim": None,
        "index_built": False
    }

    if len(input_text) < min_input_chars:
        logger.warning(
            f"Input text is short ({len(input_text)} chars, minimum: {min_input_chars}). "
            "Verification may not work well with limited evidence."
        )
        if len(input_text) < 100:
            raise ValueError(
                f"Input text too short for verification: {len(input_text)} chars "
                f"(absolute minimum: 100 chars)"
            )

    logger.info(f"Chunking input text ({len(input_text)} chars)")
    input_chunks = chunk_text(input_text, chunk_size=500, overlap=50)

    for chunk in input_chunks:
        evidence = Evidence(
            evidence_id="",
            source_id="session_input",
            source_type="transcript",
            text=chunk["text"],
            chunk_index=chunk["chunk_index"],
            char_start=chunk["char_start"],
            char_end=chunk["char_end"],
            metadata={"session_id": session_id}
        )
        store.add_evidence(evidence)

    stats["chunks_added"] += len(input_chunks)
    logger.info(f"Added {len(input_chunks)} chunks from input text")

    if external_context and len(external_context) >= 100:
        logger.info(f"Chunking external context ({len(external_context)} chars)")
        external_chunks = chunk_text(external_context, chunk_size=500, overlap=50)

        for chunk in external_chunks:
            evidence = Evidence(
                evidence_id="",
                source_id="external_context",
                source_type="external",
                text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                char_start=chunk["char_start"],
                char_end=chunk["char_end"],
                metadata={"session_id": session_id}
            )
            store.add_evidence(evidence)

        stats["chunks_added"] += len(external_chunks)
        logger.info(f"Added {len(external_chunks)} chunks from external context")

    if equations:
        for i, eq in enumerate(equations):
            if not eq or len(eq.strip()) < 3:
                continue

            evidence = Evidence(
                evidence_id="",
                source_id="equations",
                source_type="equations",
                text=eq.strip(),
                chunk_index=i,
                char_start=0,
                char_end=len(eq),
                metadata={"session_id": session_id, "is_equation": True}
            )
            store.add_evidence(evidence)

        stats["chunks_added"] += len(equations)
        logger.info(f"Added {len(equations)} equations as evidence")

    if urls and config.ENABLE_URL_SOURCES:
        logger.info(f"Ingesting {len(urls)} URLs")

        try:
            url_sources = ingest_urls(urls)
            _ = chunk_url_sources(url_sources, chunk_size=500, overlap=50)

            url_chunk_index = 0
            for url_source in url_sources:
                if url_source.error or not url_source.text:
                    continue

                source_chunks = chunk_text(url_source.text, chunk_size=500, overlap=50)

                for chunk in source_chunks:
                    evidence = Evidence(
                        evidence_id="",
                        source_id=url_source.url,
                        source_type=url_source.source_type,
                        text=chunk["text"],
                        chunk_index=chunk["chunk_index"],
                        char_start=chunk["char_start"],
                        char_end=chunk["char_end"],
                        metadata={
                            "session_id": session_id,
                            "url": url_source.url,
                            "title": url_source.title,
                            "fetched_at": url_source.fetched_at
                        }
                    )
                    store.add_evidence(evidence)
                    url_chunk_index += 1

            url_summary = get_url_ingestion_summary(url_sources)
            stats["url_ingestion"] = url_summary
            stats["chunks_added"] += url_summary["successful"]

            logger.info(
                f"Added {url_chunk_index} chunks from URLs "
                f"({url_summary['successful']}/{url_summary['total_urls']} successful)"
            )

        except Exception as e:
            logger.error(f"URL ingestion failed: {e}")
            stats["url_ingestion"] = {
                "error": str(e),
                "total_urls": len(urls),
                "successful": 0,
                "failed": len(urls)
            }

    elif urls and not config.ENABLE_URL_SOURCES:
        logger.warning("URLs provided but ENABLE_URL_SOURCES=False, skipping URL ingestion")

    # Integrate online evidence from authoritative sources
    online_conflicts_summary = None
    if config.ENABLE_ONLINE_VERIFICATION and input_text:
        try:
            logger.info("Retrieving online evidence from authoritative sources")
            integrator = create_integrator(
                enable_online=True,
                enforce_policies=True
            )
            
            # Extract key claims from input text (simple heuristic: sentences with capitalized keywords)
            import re
            sentences = re.split(r'[.!?]+', input_text)
            claims_to_verify = [
                s.strip() for s in sentences 
                if s.strip() and len(s.strip()) > 20 and any(
                    keyword in s.lower() for keyword in ['is', 'are', 'was', 'were', 'define', 'explain']
                )
            ][:config.ONLINE_MAX_SOURCES_PER_CLAIM]
            
            online_evidence_added = 0
            for claim in claims_to_verify:
                try:
                    online_spans = integrator.retrieve_online_evidence(claim, num_results=3)
                    
                    for span in online_spans:
                        evidence = Evidence(
                            evidence_id="",
                            source_id=span.source_id,
                            source_type="online_authority",
                            text=span.text,
                            chunk_index=0,
                            char_start=0,
                            char_end=len(span.text),
                            metadata={
                                "session_id": session_id,
                                "online_source": span.source_id,
                                "authority_tier": str(span.authority_tier),
                                "authority_weight": float(span.authority_weight),
                                "access_date": span.access_date,
                                "from_online_verification": True,
                                "claim_verified": claim
                            }
                        )
                        store.add_evidence(evidence)
                        online_evidence_added += 1
                except Exception as e:
                    logger.warning(f"Failed to retrieve online evidence for claim: {e}")
            
            if online_evidence_added > 0:
                logger.info(f"Added {online_evidence_added} online evidence chunks to store")
                stats["online_evidence_chunks"] = online_evidence_added
            
            # Get conflict summary for reporting
            online_conflicts_summary = integrator.get_conflict_summary()
            stats["online_conflicts"] = online_conflicts_summary
            
        except Exception as e:
            logger.error(f"Online verification failed: {e}")
            stats["online_verification_error"] = str(e)

    if len(store.evidence) == 0:
        raise ValueError(
            "Failed to create any evidence chunks. Input text may be too short or invalid."
        )

    logger.info(
        f"✓ Evidence store built: {len(store.evidence)} chunks, "
        f"{store.total_chars} total chars from {len(store.source_counts)} sources"
    )

    # Generate embeddings if provider available
    if embedding_provider and store.evidence:
        try:
            logger.info(f"Generating embeddings for {len(store.evidence)} evidence chunks")
            texts = [ev.text for ev in store.evidence]
            embeddings = embedding_provider.embed_texts(texts)

            stats["embedding_model"] = embedding_provider.model_name
            stats["embedding_dim"] = embeddings.shape[1] if len(embeddings.shape) > 1 else None

            # Save embeddings to artifact store
            if artifact_store:
                artifact_store.set_embeddings(embeddings)

            # Build FAISS index
            store.build_index(embeddings=embeddings)
            stats["index_built"] = True

            logger.info(
                f"Built FAISS index: {len(store.evidence)} evidence, "
                f"embedding_dim={stats['embedding_dim']}"
            )
        except Exception as exc:
            logger.error(f"Failed to build evidence index: {exc}")
            raise ValueError("Failed to embed evidence for verification") from exc
    
    # Save artifacts if enabled
    if artifact_store:
        try:
            metadata = RunMetadata(
                run_id=artifact_store.run_id,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                random_seed=config.GLOBAL_RANDOM_SEED,
                config_snapshot=create_config_snapshot(content_hash, config.GLOBAL_RANDOM_SEED),
                git_commit=get_git_commit(),
                model_ids={"embedding": model_id, "llm": getattr(config, "LLM_MODEL", "unknown")},
                source_count=len(artifact_store.sources),
                span_count=len(artifact_store.spans),
                embedding_dim=stats["embedding_dim"] or 0,
                cache_status=cache_status
            )
            artifact_store.save(metadata)
            logger.info(f"Saved artifacts to {artifact_store.run_dir}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")

    return store, stats


def _build_store_from_artifacts(
    session_id: str,
    artifact_store: ArtifactStore,
    cache_status: str
) -> tuple[EvidenceStore, Dict[str, Any]]:
    """
    Build EvidenceStore from loaded artifacts (cache hit).
    
    Args:
        session_id: Session identifier
        artifact_store: Loaded ArtifactStore with spans and embeddings
        cache_status: Cache status string
    
    Returns:
        (evidence_store, stats)
    """
    store = EvidenceStore(session_id=session_id)
    
    # Convert spans to Evidence objects
    for span in artifact_store.spans:
        evidence = Evidence(
            evidence_id=span.span_id,
            source_id=span.source_id,
            source_type="transcript",  # Could be inferred from source_id
            text=span.text,
            chunk_index=span.chunk_idx,
            char_start=span.start,
            char_end=span.end,
            metadata={"session_id": session_id, "from_cache": True}
        )
        store.add_evidence(evidence)
    
    # Build FAISS index with cached embeddings
    if artifact_store.embeddings is not None:
        store.build_index(embeddings=artifact_store.embeddings)
    
    stats = {
        "session_id": session_id,
        "input_chars": sum(s.char_count for s in artifact_store.sources),
        "external_chars": 0,
        "equations_count": 0,
        "urls_count": 0,
        "chunks_added": len(artifact_store.spans),
        "url_ingestion": None,
        "embedding_model": artifact_store.metadata.model_ids.get("embedding") if artifact_store.metadata else None,
        "embedding_dim": artifact_store.metadata.embedding_dim if artifact_store.metadata else None,
        "index_built": artifact_store.embeddings is not None,
        "cache_status": cache_status,
        "artifact_run_id": artifact_store.run_id
    }
    
    logger.info(f"Built store from cached artifacts: {len(store.evidence)} evidence chunks")
    return store, stats


def add_url_sources_to_store(
    store: EvidenceStore,
    urls: List[str],
    embedding_provider: Optional["EmbeddingProvider"] = None
) -> Dict[str, Any]:
    """
    Add URL sources to an existing evidence store.

    Args:
        store: Evidence store to add to
        urls: List of URLs to ingest
        embedding_provider: Optional embedding provider to rebuild index

    Returns:
        Ingestion summary dict
    """
    if not urls or not config.ENABLE_URL_SOURCES:
        return {"total_urls": 0, "successful": 0, "failed": 0}

    logger.info(f"Adding {len(urls)} URLs to evidence store")

    try:
        url_sources = ingest_urls(urls)

        added_count = 0
        for url_source in url_sources:
            if url_source.error or not url_source.text:
                continue

            source_chunks = chunk_text(url_source.text, chunk_size=500, overlap=50)

            for chunk in source_chunks:
                evidence = Evidence(
                    evidence_id="",
                    source_id=url_source.url,
                    source_type=url_source.source_type,
                    text=chunk["text"],
                    chunk_index=chunk["chunk_index"],
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    metadata={
                        "session_id": store.session_id,
                        "url": url_source.url,
                        "title": url_source.title,
                        "fetched_at": url_source.fetched_at
                    }
                )
                store.add_evidence(evidence)
                added_count += 1

        if embedding_provider is not None and store.evidence:
            logger.info("Rebuilding evidence index after URL ingestion")
            evidence_texts = [ev.text for ev in store.evidence]
            embeddings = embedding_provider.embed_texts(evidence_texts)
            for i, ev in enumerate(store.evidence):
                ev.embedding = embeddings[i]
            store.build_index(embeddings=embeddings, embedding_dim=embeddings.shape[1])

        summary = get_url_ingestion_summary(url_sources)

        logger.info(
            f"Added {added_count} chunks from URLs "
            f"({summary['successful']}/{summary['total_urls']} successful)"
        )

        return summary

    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        return {
            "error": str(e),
            "total_urls": len(urls),
            "successful": 0,
            "failed": len(urls)
        }
