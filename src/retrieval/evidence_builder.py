"""
Evidence ingestion and store building for verification pipeline.

This module provides functions to build evidence stores from various sources:
- Session input (transcript/notes)
- External context
- URLs (YouTube videos, articles)
"""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.retrieval.evidence_store import Evidence, EvidenceStore
from src.retrieval.url_ingest import ingest_urls, chunk_url_sources, get_url_ingestion_summary
import config

logger = logging.getLogger(__name__)


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
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for last period/question/exclamation mark
            for sep in ['. ', '? ', '! ', '\n\n', '\n']:
                last_sep = chunk_text.rfind(sep)
                if last_sep > chunk_size * 0.7:  # At least 70% of chunk
                    chunk_text = chunk_text[:last_sep + 1]
                    end = start + last_sep + 1
                    break
        
        # Clean whitespace
        chunk_text = chunk_text.strip()
        
        # Only add if meets minimum size
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                "text": chunk_text,
                "char_start": start,
                "char_end": end,
                "chunk_index": chunk_index
            })
            chunk_index += 1
        
        # Move to next chunk with overlap
        start = end - overlap
    
    return chunks


def build_session_evidence_store(
    session_id: str,
    input_text: str,
    external_context: str = "",
    equations: List[str] = None,
    urls: List[str] = None,
    min_input_chars: int = None
) -> tuple[EvidenceStore, Dict[str, Any]]:
    """
    Build evidence store from session inputs.
    
    This function:
    1. Validates input text length
    2. Chunks input text into evidence
    3. Adds external context if provided
    4. Ingests URLs if provided and enabled
    5. Creates Evidence objects with proper metadata
    6. Returns store WITHOUT building index (caller must add embeddings and build)
    
    Args:
        session_id: Unique session identifier
        input_text: Main input text (transcript/notes)
        external_context: Optional external reference material
        equations: Optional list of equations
        urls: Optional list of URLs to ingest
        min_input_chars: Minimum input characters (default: from config)
    
    Returns:
        (evidence_store, ingestion_summary)
    
    Raises:
        ValueError: If input validation fails
    """
    if min_input_chars is None:
        min_input_chars = getattr(config, 'MIN_INPUT_CHARS_FOR_VERIFICATION', 500)
    
    logger.info(f"Building evidence store for session {session_id}")
    
    # Initialize store
    store = EvidenceStore(session_id=session_id)
    
    # Statistics
    stats = {
        "session_id": session_id,
        "input_chars": len(input_text),
        "external_chars": len(external_context) if external_context else 0,
        "equations_count": len(equations) if equations else 0,
        "urls_count": len(urls) if urls else 0,
        "chunks_added": 0,
        "url_ingestion": None
    }
    
    # Validate input text length
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
    
    # 1. Chunk main input text
    logger.info(f"Chunking input text ({len(input_text)} chars)")
    input_chunks = chunk_text(input_text, chunk_size=500, overlap=50)
    
    for chunk in input_chunks:
        evidence = Evidence(
            evidence_id="",  # Will be auto-generated
            source_id="session_input",
            source_type="transcript",  # Could also be "notes" based on context
            text=chunk["text"],
            chunk_index=chunk["chunk_index"],
            char_start=chunk["char_start"],
            char_end=chunk["char_end"],
            metadata={"session_id": session_id}
        )
        store.add_evidence(evidence)
    
    stats["chunks_added"] += len(input_chunks)
    logger.info(f"Added {len(input_chunks)} chunks from input text")
    
    # 2. Add external context if provided
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
    
    # 3. Add equations if provided
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
    
    # 4. Ingest URLs if provided and enabled
    if urls and config.ENABLE_URL_SOURCES:
        logger.info(f"Ingesting {len(urls)} URLs")
        
        try:
            # Fetch URL content
            url_sources = ingest_urls(urls)
            
            # Chunk URL sources
            url_chunks_raw = chunk_url_sources(url_sources, chunk_size=500, overlap=50)
            
            # Convert to Evidence objects
            url_chunk_index = 0
            for url_source in url_sources:
                if url_source.error or not url_source.text:
                    continue
                
                # Chunk this source's text
                source_chunks = chunk_text(url_source.text, chunk_size=500, overlap=50)
                
                for chunk in source_chunks:
                    evidence = Evidence(
                        evidence_id="",
                        source_id=url_source.url,
                        source_type=url_source.source_type,  # "youtube" or "article"
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
            
            # Get summary
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
    
    # Final validation
    if len(store.evidence) == 0:
        raise ValueError(
            "Failed to create any evidence chunks. Input text may be too short or invalid."
        )
    
    logger.info(
        f"✓ Evidence store built: {len(store.evidence)} chunks, "
        f"{store.total_chars} total chars from {len(store.source_counts)} sources"
    )
    
    return store, stats


def add_url_sources_to_store(
    store: EvidenceStore,
    urls: List[str]
) -> Dict[str, Any]:
    """
    Add URL sources to an existing evidence store.
    
    Args:
        store: Evidence store to add to
        urls: List of URLs to ingest
    
    Returns:
        Ingestion summary dict
    """
    if not urls or not config.ENABLE_URL_SOURCES:
        return {"total_urls": 0, "successful": 0, "failed": 0}
    
    logger.info(f"Adding {len(urls)} URLs to evidence store")
    
    try:
        # Fetch URL content
        url_sources = ingest_urls(urls)
        
        # Convert to Evidence objects
        added_count = 0
        for url_source in url_sources:
            if url_source.error or not url_source.text:
                continue
            
            # Chunk this source's text
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
        
        # Get summary
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
