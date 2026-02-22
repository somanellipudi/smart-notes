"""
Tests for engines used metadata tracking.
"""

import pytest
from unittest.mock import Mock, patch

from src.ui.progress_tracker import RunContext, EnginesUsed, create_run_context
from src.ui.pipeline_instrumentation import (
    track_llm_call,
    track_ocr_usage,
    track_url_ingestion,
    track_retrieval_usage,
    update_stage_metrics,
    add_stage_warning
)


class TestLLMTracking:
    """Test LLM provider and model tracking."""
    
    def test_track_single_provider(self):
        """Test tracking single LLM provider."""
        ctx = create_run_context("session_123")
        
        track_llm_call(ctx, "openai", "gpt-4o-mini", tokens=500)
        track_llm_call(ctx, "openai", "gpt-4o-mini", tokens=300)
        
        engines = ctx.engines_used
        assert engines.llm_provider == "openai"
        assert engines.llm_model == "gpt-4o-mini"
        assert engines.llm_calls == 2
        assert engines.llm_tokens_total == 800
    
    def test_track_mixed_providers(self):
        """Test tracking multiple LLM providers."""
        ctx = create_run_context("session_123")
        
        track_llm_call(ctx, "openai", "gpt-4")
        track_llm_call(ctx, "ollama", "llama3")
        
        engines = ctx.engines_used
        assert engines.llm_provider == "mixed"
        assert engines.llm_calls == 2
    
    def test_track_multiple_models(self):
        """Test tracking multiple models from same provider."""
        ctx = create_run_context("session_123")
        
        track_llm_call(ctx, "openai", "gpt-4o-mini")
        track_llm_call(ctx, "openai", "gpt-4")
        
        engines = ctx.engines_used
        assert engines.llm_provider == "openai"
        assert "gpt-4o-mini" in engines.llm_model
        assert "gpt-4" in engines.llm_model
    
    def test_track_without_tokens(self):
        """Test tracking LLM calls without token counts."""
        ctx = create_run_context("session_123")
        
        track_llm_call(ctx, "ollama", "mistral")
        
        engines = ctx.engines_used
        assert engines.llm_provider == "ollama"
        assert engines.llm_calls == 1
        assert engines.llm_tokens_total is None


class TestOCRTracking:
    """Test OCR usage tracking."""
    
    def test_track_ocr_disabled(self):
        """Test tracking when OCR is disabled."""
        ctx = create_run_context("session_123")
        
        track_ocr_usage(ctx, enabled=False)
        
        engines = ctx.engines_used
        assert engines.ocr_enabled is False
        assert engines.ocr_engine is None
        assert engines.ocr_pages_processed == 0
    
    def test_track_ocr_enabled_cpu(self):
        """Test tracking OCR on CPU."""
        ctx = create_run_context("session_123")
        
        track_ocr_usage(
            ctx,
            enabled=True,
            engine="easyocr",
            device="cpu",
            pages_processed=5
        )
        
        engines = ctx.engines_used
        assert engines.ocr_enabled is True
        assert engines.ocr_engine == "easyocr"
        assert engines.ocr_device == "cpu"
        assert engines.ocr_pages_processed == 5
    
    def test_track_ocr_enabled_gpu(self):
        """Test tracking OCR on GPU."""
        ctx = create_run_context("session_123")
        
        track_ocr_usage(
            ctx,
            enabled=True,
            engine="easyocr",
            device="cuda",
            pages_processed=10,
            model_downloaded=True
        )
        
        engines = ctx.engines_used
        assert engines.ocr_device == "cuda"
        assert engines.ocr_pages_processed == 10
        assert engines.ocr_model_downloaded is True
    
    def test_track_incremental_pages(self):
        """Test tracking OCR pages incrementally."""
        ctx = create_run_context("session_123")
        
        track_ocr_usage(ctx, enabled=True, pages_processed=3)
        track_ocr_usage(ctx, enabled=True, pages_processed=2)
        track_ocr_usage(ctx, enabled=True, pages_processed=5)
        
        engines = ctx.engines_used
        assert engines.ocr_pages_processed == 10


class TestURLIngestionTracking:
    """Test URL ingestion tracking."""
    
    def test_track_url_success(self):
        """Test tracking successful URL ingestion."""
        ctx = create_run_context("session_123")
        
        track_url_ingestion(
            ctx,
            total=5,
            success=4,
            youtube=2,
            articles=2
        )
        
        engines = ctx.engines_used
        assert engines.url_count_total == 5
        assert engines.url_count_success == 4
        assert engines.youtube_transcripts_extracted == 2
        assert engines.article_extractions == 2
    
    def test_track_url_failures(self):
        """Test tracking URL ingestion with failures."""
        ctx = create_run_context("session_123")
        
        track_url_ingestion(ctx, total=3, success=1)
        
        engines = ctx.engines_used
        assert engines.url_count_total == 3
        assert engines.url_count_success == 1
    
    def test_track_youtube_only(self):
        """Test tracking only YouTube transcripts."""
        ctx = create_run_context("session_123")
        
        track_url_ingestion(ctx, total=2, success=2, youtube=2, articles=0)
        
        engines = ctx.engines_used
        assert engines.youtube_transcripts_extracted == 2
        assert engines.article_extractions == 0
    
    def test_track_incremental_urls(self):
        """Test tracking URLs incrementally."""
        ctx = create_run_context("session_123")
        
        track_url_ingestion(ctx, total=2, success=2, youtube=1, articles=1)
        track_url_ingestion(ctx, total=1, success=1, articles=1)
        
        engines = ctx.engines_used
        assert engines.url_count_total == 3
        assert engines.url_count_success == 3
        assert engines.youtube_transcripts_extracted == 1
        assert engines.article_extractions == 2


class TestRetrievalTracking:
    """Test retrieval/verification engine tracking."""
    
    def test_track_retrieval_disabled(self):
        """Test tracking when retrieval is disabled."""
        ctx = create_run_context("session_123")
        
        track_retrieval_usage(ctx, enabled=False)
        
        engines = ctx.engines_used
        assert engines.retrieval_enabled is False
    
    def test_track_retrieval_enabled(self):
        """Test tracking retrieval with models."""
        ctx = create_run_context("session_123")
        
        track_retrieval_usage(
            ctx,
            enabled=True,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            nli_model="microsoft/deberta-v3-base"
        )
        
        engines = ctx.engines_used
        assert engines.retrieval_enabled is True
        assert engines.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert engines.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert engines.nli_model == "microsoft/deberta-v3-base"
    
    def test_track_retrieval_partial(self):
        """Test tracking retrieval with some models."""
        ctx = create_run_context("session_123")
        
        track_retrieval_usage(
            ctx,
            enabled=True,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        engines = ctx.engines_used
        assert engines.retrieval_enabled is True
        assert engines.embedding_model == "all-MiniLM-L6-v2"
        assert engines.reranker_model is None


class TestStageMetricsUpdate:
    """Test updating stage metrics."""
    
    def test_update_metrics_running_stage(self):
        """Test updating metrics for running stage."""
        ctx = create_run_context("session_123")
        ctx.start_stage("ingestion_cleaning")
        
        update_stage_metrics(
            ctx,
            "ingestion_cleaning",
            {"files_processed": 5, "lines_removed": 100}
        )
        
        stage = ctx.get_stage("ingestion_cleaning")
        assert stage.metrics["files_processed"] == 5
        assert stage.metrics["lines_removed"] == 100
    
    def test_update_metrics_and_complete(self):
        """Test updating metrics and completing stage."""
        ctx = create_run_context("session_123")
        ctx.start_stage("chunking_provenance")
        
        update_stage_metrics(
            ctx,
            "chunking_provenance",
            {"chunks_total": 42},
            complete=True
        )
        
        stage = ctx.get_stage("chunking_provenance")
        assert stage.metrics["chunks_total"] == 42
        assert stage.status.value == "completed"
    
    def test_add_stage_warning(self):
        """Test adding warning to stage."""
        ctx = create_run_context("session_123")
        ctx.start_stage("verification")
        ctx.complete_stage("verification")
        
        add_stage_warning(ctx, "verification", "Low confidence claims detected")
        
        stage = ctx.get_stage("verification")
        assert stage.status.value == "warning"
        assert len(stage.warnings) == 1


class TestIntegratedEnginesTracking:
    """Test integrated engines tracking across pipeline."""
    
    def test_full_pipeline_tracking(self):
        """Test tracking all engines through full pipeline."""
        ctx = create_run_context("session_123", seed=42)
        
        # Track all engines
        track_llm_call(ctx, "openai", "gpt-4o-mini", tokens=1000)
        track_ocr_usage(ctx, enabled=True, engine="easyocr", device="cpu", pages_processed=5)
        track_url_ingestion(ctx, total=3, success=3, youtube=1, articles=2)
        track_retrieval_usage(
            ctx,
            enabled=True,
            embedding_model="all-MiniLM-L6-v2",
            reranker_model="ms-marco-MiniLM"
        )
        
        engines = ctx.engines_used
        
        # Verify all tracked
        assert engines.llm_provider == "openai"
        assert engines.llm_tokens_total == 1000
        assert engines.ocr_enabled is True
        assert engines.ocr_pages_processed == 5
        assert engines.url_count_success == 3
        assert engines.retrieval_enabled is True
    
    def test_minimal_pipeline_tracking(self):
        """Test tracking minimal pipeline (text-only, fast mode)."""
        ctx = create_run_context("session_123")
        
        # Only LLM is used
        track_llm_call(ctx, "ollama", "llama3")
        track_ocr_usage(ctx, enabled=False)
        track_retrieval_usage(ctx, enabled=False)
        
        engines = ctx.engines_used
        
        assert engines.llm_provider == "ollama"
        assert engines.ocr_enabled is False
        assert engines.retrieval_enabled is False
        assert engines.url_count_total == 0
