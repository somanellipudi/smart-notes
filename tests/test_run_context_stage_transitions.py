"""
Tests for RunContext stage transitions and progress tracking.
"""

import pytest
import time
from pathlib import Path
import tempfile
import shutil

from src.ui.progress_tracker import (
    RunContext,
    StageStatus,
    StageEvent,
    EnginesUsed,
    create_run_context
)


class TestStageEvent:
    """Test StageEvent lifecycle management."""
    
    def test_stage_event_initialization(self):
        """Test stage event starts in correct state."""
        stage = StageEvent(
            stage_name="Test Stage",
            status=StageStatus.NOT_STARTED
        )
        
        assert stage.stage_name == "Test Stage"
        assert stage.status == StageStatus.NOT_STARTED
        assert stage.started_at is None
        assert stage.ended_at is None
        assert stage.duration_s is None
        assert len(stage.metrics) == 0
        assert len(stage.warnings) == 0
        assert stage.error is None
    
    def test_stage_start(self):
        """Test starting a stage."""
        stage = StageEvent(stage_name="Test", status=StageStatus.NOT_STARTED)
        
        stage.start()
        
        assert stage.status == StageStatus.RUNNING
        assert stage.started_at is not None
        assert stage.ended_at is None
    
    def test_stage_complete(self):
        """Test completing a stage."""
        stage = StageEvent(stage_name="Test", status=StageStatus.NOT_STARTED)
        stage.start()
        time.sleep(0.1)
        
        metrics = {"items_processed": 42, "success_rate": 0.95}
        stage.complete(metrics)
        
        assert stage.status == StageStatus.COMPLETED
        assert stage.ended_at is not None
        assert stage.duration_s is not None
        assert stage.duration_s >= 0.1
        assert stage.metrics["items_processed"] == 42
        assert stage.metrics["success_rate"] == 0.95
    
    def test_stage_skip(self):
        """Test skipping a stage."""
        stage = StageEvent(stage_name="Test", status=StageStatus.NOT_STARTED)
        
        stage.skip("Feature not enabled")
        
        assert stage.status == StageStatus.SKIPPED
        assert stage.skip_reason == "Feature not enabled"
        assert stage.duration_s == 0.0
    
    def test_stage_fail(self):
        """Test failing a stage."""
        stage = StageEvent(stage_name="Test", status=StageStatus.NOT_STARTED)
        stage.start()
        time.sleep(0.05)
        
        stage.fail("Connection timeout")
        
        assert stage.status == StageStatus.FAILED
        assert stage.error == "Connection timeout"
        assert stage.ended_at is not None
        assert stage.duration_s is not None
    
    def test_stage_warning(self):
        """Test adding warnings to stage."""
        stage = StageEvent(stage_name="Test", status=StageStatus.NOT_STARTED)
        stage.start()
        stage.complete()
        
        stage.warn("Low confidence detected")
        
        assert stage.status == StageStatus.WARNING
        assert len(stage.warnings) == 1
        assert stage.warnings[0] == "Low confidence detected"
    
    def test_stage_to_dict(self):
        """Test serialization to dict."""
        stage = StageEvent(stage_name="Test", status=StageStatus.COMPLETED)
        stage.metrics = {"count": 10}
        stage.warnings = ["warning1"]
        
        data = stage.to_dict()
        
        assert data["stage_name"] == "Test"
        assert data["status"] == "completed"
        assert data["metrics"]["count"] == 10
        assert data["warnings"] == ["warning1"]


class TestEnginesUsed:
    """Test EnginesUsed tracking."""
    
    def test_initialization(self):
        """Test engines tracking initialization."""
        engines = EnginesUsed()
        
        assert engines.llm_provider is None
        assert engines.llm_calls == 0
        assert engines.ocr_enabled is False
        assert engines.url_count_total == 0
        assert engines.retrieval_enabled is False
    
    def test_llm_tracking(self):
        """Test LLM usage tracking."""
        engines = EnginesUsed(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_calls=5,
            llm_tokens_total=1500
        )
        
        assert engines.llm_provider == "openai"
        assert engines.llm_model == "gpt-4o-mini"
        assert engines.llm_calls == 5
        assert engines.llm_tokens_total == 1500
    
    def test_ocr_tracking(self):
        """Test OCR usage tracking."""
        engines = EnginesUsed(
            ocr_enabled=True,
            ocr_engine="easyocr",
            ocr_device="cuda",
            ocr_pages_processed=10,
            ocr_model_downloaded=True
        )
        
        assert engines.ocr_enabled is True
        assert engines.ocr_engine == "easyocr"
        assert engines.ocr_device == "cuda"
        assert engines.ocr_pages_processed == 10
        assert engines.ocr_model_downloaded is True
    
    def test_url_tracking(self):
        """Test URL ingestion tracking."""
        engines = EnginesUsed(
            url_count_total=5,
            url_count_success=4,
            youtube_transcripts_extracted=2,
            article_extractions=2
        )
        
        assert engines.url_count_total == 5
        assert engines.url_count_success == 4
        assert engines.youtube_transcripts_extracted == 2
        assert engines.article_extractions == 2
    
    def test_to_dict(self):
        """Test serialization to dict."""
        engines = EnginesUsed(llm_provider="ollama", ocr_enabled=True)
        data = engines.to_dict()
        
        assert data["llm_provider"] == "ollama"
        assert data["ocr_enabled"] is True


class TestRunContext:
    """Test RunContext pipeline tracking."""
    
    def test_initialization(self):
        """Test run context initialization."""
        ctx = create_run_context("session_123", seed=42)
        
        assert ctx.session_id == "session_123"
        assert ctx.seed == 42
        assert ctx.run_id.startswith("run_")
        assert len(ctx.stage_events) == 10  # All 10 stages
        
        # All stages should start as NOT_STARTED
        for stage in ctx.stage_events.values():
            assert stage.status == StageStatus.NOT_STARTED
    
    def test_stage_transitions(self):
        """Test stage transitions through lifecycle."""
        ctx = create_run_context("session_123")
        
        # Start first stage
        ctx.start_stage("inputs_received")
        stage = ctx.get_stage("inputs_received")
        assert stage.status == StageStatus.RUNNING
        
        # Complete with metrics
        ctx.complete_stage("inputs_received", {"files": 3})
        assert stage.status == StageStatus.COMPLETED
        assert stage.metrics["files"] == 3
        assert stage.duration_s is not None
    
    def test_skip_stage(self):
        """Test skipping a stage."""
        ctx = create_run_context("session_123")
        
        ctx.skip_stage("ocr_extraction", "OCR disabled")
        stage = ctx.get_stage("ocr_extraction")
        
        assert stage.status == StageStatus.SKIPPED
        assert stage.skip_reason == "OCR disabled"
    
    def test_fail_stage(self):
        """Test failing a stage."""
        ctx = create_run_context("session_123")
        
        ctx.start_stage("llm_generation")
        ctx.fail_stage("llm_generation", "API timeout")
        stage = ctx.get_stage("llm_generation")
        
        assert stage.status == StageStatus.FAILED
        assert stage.error == "API timeout"
    
    def test_warn_stage(self):
        """Test adding warning to stage."""
        ctx = create_run_context("session_123")
        
        ctx.start_stage("verification")
        ctx.complete_stage("verification")
        ctx.warn_stage("verification", "Low confidence claims detected")
        
        stage = ctx.get_stage("verification")
        assert stage.status == StageStatus.WARNING
        assert len(stage.warnings) == 1
    
    def test_progress_calculation(self):
        """Test progress fraction calculation."""
        ctx = create_run_context("session_123")
        
        # Initially 0%
        assert ctx.get_progress_fraction() == 0.0
        
        # Complete 5 out of 10 stages
        for stage_key in ctx.STAGE_ORDER[:5]:
            ctx.start_stage(stage_key)
            ctx.complete_stage(stage_key)
        
        assert ctx.get_progress_fraction() == 0.5
        
        # Skip remaining stages
        for stage_key in ctx.STAGE_ORDER[5:]:
            ctx.skip_stage(stage_key, "Not needed")
        
        assert ctx.get_progress_fraction() == 1.0
    
    def test_total_duration(self):
        """Test total duration calculation."""
        ctx = create_run_context("session_123")
        
        # Complete some stages with durations
        ctx.start_stage("inputs_received")
        time.sleep(0.1)
        ctx.complete_stage("inputs_received")
        
        ctx.start_stage("ingestion_cleaning")
        time.sleep(0.1)
        ctx.complete_stage("ingestion_cleaning")
        
        total_duration = ctx.get_total_duration()
        assert total_duration >= 0.2
    
    def test_to_dict(self):
        """Test serialization to dict."""
        ctx = create_run_context("session_123", seed=42)
        ctx.engines_used.llm_provider = "openai"
        ctx.start_stage("inputs_received")
        ctx.complete_stage("inputs_received", {"files": 2})
        
        data = ctx.to_dict()
        
        assert data["session_id"] == "session_123"
        assert data["seed"] == 42
        assert data["engines_used"]["llm_provider"] == "openai"
        assert data["stage_events"]["inputs_received"]["metrics"]["files"] == 2
        assert "total_duration_s" in data
        assert "progress_fraction" in data


class TestRunContextPersistence:
    """Test RunContext save/load functionality."""
    
    def setup_method(self):
        """Create temporary artifacts directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_run_context(self):
        """Test saving run context to disk."""
        ctx = create_run_context("session_123", seed=42)
        ctx.engines_used.llm_provider = "openai"
        ctx.start_stage("inputs_received")
        ctx.complete_stage("inputs_received", {"files": 3})
        
        output_path = ctx.save(self.temp_dir)
        
        assert output_path.exists()
        assert output_path.name == "run_context.json"
        
        # Verify directory structure
        expected_path = self.temp_dir / "session_123" / ctx.run_id / "run_context.json"
        assert output_path == expected_path
    
    def test_load_run_context(self):
        """Test loading run context from disk."""
        # Create and save context
        ctx = create_run_context("session_123", seed=42)
        ctx.engines_used.llm_provider = "ollama"
        ctx.engines_used.llm_model = "llama3"
        ctx.start_stage("inputs_received")
        ctx.complete_stage("inputs_received", {"files": 5})
        ctx.save(self.temp_dir)
        
        # Load context
        loaded_ctx = RunContext.load(self.temp_dir, "session_123", ctx.run_id)
        
        assert loaded_ctx.session_id == "session_123"
        assert loaded_ctx.seed == 42
        assert loaded_ctx.engines_used.llm_provider == "ollama"
        assert loaded_ctx.engines_used.llm_model == "llama3"
        
        stage = loaded_ctx.get_stage("inputs_received")
        assert stage.status == StageStatus.COMPLETED
        assert stage.metrics["files"] == 5
    
    def test_roundtrip_persistence(self):
        """Test full save/load roundtrip preserves all data."""
        ctx = create_run_context("session_456", seed=99)
        
        # Set up complete context
        ctx.engines_used.llm_provider = "openai"
        ctx.engines_used.ocr_enabled = True
        ctx.engines_used.ocr_pages_processed = 10
        
        # Process some stages
        ctx.start_stage("inputs_received")
        ctx.complete_stage("inputs_received", {"files": 2})
        ctx.start_stage("ingestion_cleaning")
        ctx.complete_stage("ingestion_cleaning", {"chars": 5000})
        ctx.skip_stage("ocr_extraction", "No images")
        
        # Save and load
        ctx.save(self.temp_dir)
        loaded = RunContext.load(self.temp_dir, "session_456", ctx.run_id)
        
        # Verify all data preserved
        assert loaded.session_id == ctx.session_id
        assert loaded.seed == ctx.seed
        assert loaded.engines_used.llm_provider == "openai"
        assert loaded.engines_used.ocr_enabled is True
        assert loaded.engines_used.ocr_pages_processed == 10
        
        assert loaded.get_stage("inputs_received").status == StageStatus.COMPLETED
        assert loaded.get_stage("ingestion_cleaning").status == StageStatus.COMPLETED
        assert loaded.get_stage("ocr_extraction").status == StageStatus.SKIPPED


class TestStageStatusIcons:
    """Test stage status display helpers."""
    
    def test_status_icons(self):
        """Test status enum returns correct icons."""
        assert StageStatus.NOT_STARTED.icon() == "⬜"
        assert StageStatus.RUNNING.icon() == "⏳"
        assert StageStatus.COMPLETED.icon() == "✅"
        assert StageStatus.SKIPPED.icon() == "⊘"
        assert StageStatus.WARNING.icon() == "⚠️"
        assert StageStatus.FAILED.icon() == "❌"
    
    def test_status_colors(self):
        """Test status enum returns valid colors."""
        for status in StageStatus:
            color = status.color()
            assert color.startswith("#") or color == "#grey"
