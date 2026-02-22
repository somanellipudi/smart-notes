"""
Pipeline instrumentation helpers.

Provides decorators and context managers for automatically tracking
pipeline execution in RunContext.
"""

import functools
import logging
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager

from src.ui.progress_tracker import RunContext

logger = logging.getLogger(__name__)


@contextmanager
def track_stage(run_context: RunContext, stage_key: str, skip_if_empty: Optional[Callable] = None):
    """
    Context manager for tracking a pipeline stage.
    
    Usage:
        with track_stage(run_context, "ingestion_cleaning"):
            # do work
            result = process_data()
            # optionally add metrics at end
            run_context.complete_stage("ingestion_cleaning", {
                "files_processed": len(result)
            })
    
    Args:
        run_context: RunContext to update
        stage_key: Stage identifier (must be in STAGE_ORDER)
        skip_if_empty: Optional callable that returns bool to skip stage
    """
    # Check if should skip
    if skip_if_empty and skip_if_empty():
        reason = "No input data"
        run_context.skip_stage(stage_key, reason)
        logger.info(f"Skipping stage {stage_key}: {reason}")
        yield None
        return
    
    # Start stage
    run_context.start_stage(stage_key)
    
    try:
        yield run_context
        
        # If not already completed/failed, mark as completed
        stage = run_context.get_stage(stage_key)
        if stage.status.value == "running":
            run_context.complete_stage(stage_key)
    
    except Exception as e:
        # Mark stage as failed
        run_context.fail_stage(stage_key, str(e))
        logger.error(f"Stage {stage_key} failed: {e}")
        raise


def track_llm_call(run_context: RunContext, provider: str, model: str, tokens: Optional[int] = None):
    """
    Track an LLM API call in RunContext.
    
    Args:
        run_context: RunContext to update
        provider: "openai" | "ollama" | "anthropic" etc.
        model: Model identifier (e.g., "gpt-4o-mini", "llama3")
        tokens: Optional token count
    """
    engines = run_context.engines_used
    
    # Set provider (handle mixed case)
    if engines.llm_provider is None:
        engines.llm_provider = provider
    elif engines.llm_provider != provider:
        engines.llm_provider = "mixed"
    
    # Set model (first one used)
    if engines.llm_model is None:
        engines.llm_model = model
    elif engines.llm_model != model:
        engines.llm_model = f"{engines.llm_model}, {model}"
    
    # Increment call count
    engines.llm_calls += 1
    
    # Track tokens
    if tokens is not None:
        if engines.llm_tokens_total is None:
            engines.llm_tokens_total = tokens
        else:
            engines.llm_tokens_total += tokens


def track_ocr_usage(
    run_context: RunContext,
    enabled: bool,
    engine: Optional[str] = None,
    device: Optional[str] = None,
    pages_processed: int = 0,
    model_downloaded: bool = False
):
    """
    Track OCR usage in RunContext.
    
    Args:
        run_context: RunContext to update
        enabled: Whether OCR is enabled
        engine: OCR engine name (e.g., "easyocr", "tesseract")
        device: Device used (e.g., "cpu", "cuda", "mps")
        pages_processed: Number of pages processed
        model_downloaded: Whether model was downloaded this run
    """
    engines = run_context.engines_used
    
    engines.ocr_enabled = enabled
    if engine:
        engines.ocr_engine = engine
    if device:
        engines.ocr_device = device
    if pages_processed > 0:
        engines.ocr_pages_processed += pages_processed
    if model_downloaded:
        engines.ocr_model_downloaded = True


def track_url_ingestion(
    run_context: RunContext,
    total: int = 0,
    success: int = 0,
    youtube: int = 0,
    articles: int = 0
):
    """
    Track URL ingestion in RunContext.
    
    Args:
        run_context: RunContext to update
        total: Total URLs attempted
        success: Successfully extracted URLs
        youtube: YouTube transcripts extracted
        articles: Article content extracted
    """
    engines = run_context.engines_used
    
    engines.url_count_total += total
    engines.url_count_success += success
    engines.youtube_transcripts_extracted += youtube
    engines.article_extractions += articles


def track_retrieval_usage(
    run_context: RunContext,
    enabled: bool,
    embedding_model: Optional[str] = None,
    reranker_model: Optional[str] = None,
    nli_model: Optional[str] = None
):
    """
    Track retrieval/verification engine usage.
    
    Args:
        run_context: RunContext to update
        enabled: Whether retrieval is enabled
        embedding_model: Embedding model name
        reranker_model: Reranker model name
        nli_model: NLI model name
    """
    engines = run_context.engines_used
    
    engines.retrieval_enabled = enabled
    if embedding_model:
        engines.embedding_model = embedding_model
    if reranker_model:
        engines.reranker_model = reranker_model
    if nli_model:
        engines.nli_model = nli_model


def update_stage_metrics(
    run_context: RunContext,
    stage_key: str,
    metrics: Dict[str, Any],
    complete: bool = False
):
    """
    Update metrics for a stage without changing status.
    
    Args:
        run_context: RunContext to update
        stage_key: Stage identifier
        metrics: Metrics to add/update
        complete: If True, also mark stage as completed
    """
    stage = run_context.get_stage(stage_key)
    stage.metrics.update(metrics)
    
    if complete:
        run_context.complete_stage(stage_key)


def add_stage_warning(run_context: RunContext, stage_key: str, warning: str):
    """
    Add a warning to a stage.
    
    Args:
        run_context: RunContext to update
        stage_key: Stage identifier
        warning: Warning message
    """
    run_context.warn_stage(stage_key, warning)
