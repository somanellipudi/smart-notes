"""
Real-time execution flow tracking for Smart Notes pipeline.

Provides transparent visibility into:
- Pipeline stage progression with timing
- Engine usage (LLM provider, OCR, retrieval models)
- Metrics at each stage
- Warnings and errors

This module is the single source of truth for pipeline execution state.
"""

import time
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Pipeline stage execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    WARNING = "warning"
    FAILED = "failed"
    
    def icon(self) -> str:
        """Get emoji icon for status."""
        return {
            StageStatus.NOT_STARTED: "⬜",
            StageStatus.RUNNING: "⏳",
            StageStatus.COMPLETED: "✅",
            StageStatus.SKIPPED: "⊘",
            StageStatus.WARNING: "⚠️",
            StageStatus.FAILED: "❌"
        }[self]
    
    def color(self) -> str:
        """Get color for UI borders/backgrounds."""
        return {
            StageStatus.NOT_STARTED: "#grey",
            StageStatus.RUNNING: "#1f77b4",
            StageStatus.COMPLETED: "#2ca02c",
            StageStatus.SKIPPED: "#7f7f7f",
            StageStatus.WARNING: "#ff7f0e",
            StageStatus.FAILED: "#d62728"
        }[self]


@dataclass
class StageEvent:
    """Records execution of a single pipeline stage."""
    stage_name: str
    status: StageStatus
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_s: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    
    def start(self):
        """Mark stage as started."""
        self.started_at = time.time()
        self.status = StageStatus.RUNNING
    
    def complete(self, metrics: Optional[Dict[str, Any]] = None):
        """Mark stage as completed."""
        self.ended_at = time.time()
        if self.started_at:
            self.duration_s = round(self.ended_at - self.started_at, 2)
        self.status = StageStatus.COMPLETED
        if metrics:
            self.metrics.update(metrics)
    
    def skip(self, reason: str):
        """Mark stage as skipped."""
        self.status = StageStatus.SKIPPED
        self.skip_reason = reason
        self.duration_s = 0.0
    
    def fail(self, error: str):
        """Mark stage as failed."""
        self.ended_at = time.time()
        if self.started_at:
            self.duration_s = round(self.ended_at - self.started_at, 2)
        self.status = StageStatus.FAILED
        self.error = error
    
    def warn(self, warning: str):
        """Add warning to stage."""
        self.warnings.append(warning)
        if self.status == StageStatus.COMPLETED:
            self.status = StageStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
            "skip_reason": self.skip_reason
        }


@dataclass
class EnginesUsed:
    """Tracks which engines/models were used during execution."""
    # LLM
    llm_provider: Optional[str] = None  # "openai" | "ollama" | "mixed"
    llm_model: Optional[str] = None
    llm_calls: int = 0
    llm_tokens_total: Optional[int] = None
    
    # OCR
    ocr_enabled: bool = False
    ocr_engine: Optional[str] = None  # "easyocr" | "tesseract" | None
    ocr_device: Optional[str] = None  # "cpu" | "cuda" | "mps"
    ocr_pages_processed: int = 0
    ocr_model_downloaded: bool = False
    
    # URL Ingestion
    url_count_total: int = 0
    url_count_success: int = 0
    youtube_transcripts_extracted: int = 0
    article_extractions: int = 0
    
    # Retrieval & Verification
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    nli_model: Optional[str] = None
    retrieval_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class RunContext:
    """
    Complete execution context for a pipeline run.
    
    This is the single source of truth for what happened during execution.
    Persisted to artifacts/<session>/<run_id>/run_context.json
    """
    run_id: str
    session_id: str
    started_at: float
    seed: Optional[int] = None
    config_snapshot_hash: Optional[str] = None
    
    engines_used: EnginesUsed = field(default_factory=EnginesUsed)
    stage_events: Dict[str, StageEvent] = field(default_factory=dict)
    
    # Pipeline stages in execution order
    STAGE_ORDER = [
        "inputs_received",
        "ingestion_cleaning",
        "ocr_extraction",
        "chunking_provenance",
        "embedding_indexing",
        "llm_generation",
        "claim_extraction",
        "retrieval_reranking",
        "verification",
        "reporting_exports"
    ]
    
    STAGE_DISPLAY_NAMES = {
        "inputs_received": "Inputs Received",
        "ingestion_cleaning": "Ingestion & Cleaning",
        "ocr_extraction": "OCR Extraction",
        "chunking_provenance": "Chunking & Provenance",
        "embedding_indexing": "Embedding & Indexing",
        "llm_generation": "LLM Generation",
        "claim_extraction": "Claim Extraction",
        "retrieval_reranking": "Retrieval & Reranking",
        "verification": "Multi-Signal Verification",
        "reporting_exports": "Reporting + Exports"
    }
    
    def __post_init__(self):
        """Initialize all stages as NOT_STARTED."""
        for stage_key in self.STAGE_ORDER:
            if stage_key not in self.stage_events:
                self.stage_events[stage_key] = StageEvent(
                    stage_name=self.STAGE_DISPLAY_NAMES[stage_key],
                    status=StageStatus.NOT_STARTED
                )
    
    def get_stage(self, stage_key: str) -> StageEvent:
        """Get stage event by key."""
        if stage_key not in self.stage_events:
            raise ValueError(f"Unknown stage: {stage_key}")
        return self.stage_events[stage_key]
    
    def start_stage(self, stage_key: str):
        """Mark stage as started."""
        stage = self.get_stage(stage_key)
        stage.start()
        logger.info(f"Stage started: {stage.stage_name}")
    
    def complete_stage(self, stage_key: str, metrics: Optional[Dict[str, Any]] = None):
        """Mark stage as completed with optional metrics."""
        stage = self.get_stage(stage_key)
        stage.complete(metrics)
        logger.info(f"Stage completed: {stage.stage_name} ({stage.duration_s}s)")
    
    def skip_stage(self, stage_key: str, reason: str):
        """Mark stage as skipped."""
        stage = self.get_stage(stage_key)
        stage.skip(reason)
        logger.info(f"Stage skipped: {stage.stage_name} - {reason}")
    
    def fail_stage(self, stage_key: str, error: str):
        """Mark stage as failed."""
        stage = self.get_stage(stage_key)
        stage.fail(error)
        logger.error(f"Stage failed: {stage.stage_name} - {error}")
    
    def warn_stage(self, stage_key: str, warning: str):
        """Add warning to stage."""
        stage = self.get_stage(stage_key)
        stage.warn(warning)
        logger.warning(f"Stage warning: {stage.stage_name} - {warning}")
    
    def get_total_duration(self) -> float:
        """Get total execution time in seconds."""
        completed_stages = [
            s for s in self.stage_events.values()
            if s.duration_s is not None
        ]
        return sum(s.duration_s for s in completed_stages)
    
    def get_progress_fraction(self) -> float:
        """Get completion progress as fraction 0.0-1.0."""
        completed = sum(
            1 for s in self.stage_events.values()
            if s.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        )
        return completed / len(self.STAGE_ORDER)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "seed": self.seed,
            "config_snapshot_hash": self.config_snapshot_hash,
            "engines_used": self.engines_used.to_dict(),
            "stage_events": {k: v.to_dict() for k, v in self.stage_events.items()},
            "total_duration_s": self.get_total_duration(),
            "progress_fraction": self.get_progress_fraction()
        }
    
    def save(self, artifacts_dir: Path):
        """
        Save run context to artifacts directory.
        
        Saves to: artifacts/<session_id>/<run_id>/run_context.json
        """
        output_dir = artifacts_dir / self.session_id / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "run_context.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"RunContext saved to {output_path}")
        return output_path
    
    @classmethod
    def load(cls, artifacts_dir: Path, session_id: str, run_id: str) -> "RunContext":
        """Load run context from artifacts directory."""
        path = artifacts_dir / session_id / run_id / "run_context.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct from dict
        engines = EnginesUsed(**data["engines_used"])
        stage_events = {
            k: StageEvent(
                stage_name=v["stage_name"],
                status=StageStatus(v["status"]),
                started_at=v["started_at"],
                ended_at=v["ended_at"],
                duration_s=v["duration_s"],
                metrics=v["metrics"],
                warnings=v["warnings"],
                error=v["error"],
                skip_reason=v["skip_reason"]
            )
            for k, v in data["stage_events"].items()
        }
        
        return cls(
            run_id=data["run_id"],
            session_id=data["session_id"],
            started_at=data["started_at"],
            seed=data["seed"],
            config_snapshot_hash=data["config_snapshot_hash"],
            engines_used=engines,
            stage_events=stage_events
        )


def create_run_context(session_id: str, seed: Optional[int] = None) -> RunContext:
    """
    Create a new run context for pipeline execution.
    
    Args:
        session_id: Session identifier
        seed: Random seed for reproducibility
    
    Returns:
        RunContext instance
    """
    run_id = f"run_{int(time.time() * 1000)}"
    return RunContext(
        run_id=run_id,
        session_id=session_id,
        started_at=time.time(),
        seed=seed
    )
