"""Verification configuration with validation and defaults.

Centralizes thresholds and weights used by the verifiable fact-checking
pipeline. Provide a ValidationError for clear errors on misconfiguration.

Supports deployment modes:
- full_default: All optimization models enabled (maximum throughput)
- minimal_deployment: Result caching + pre-screening only (75% cost savings)
- verifiable: Minimal deps for verification (baseline for ablation)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple, Literal

DEPLOYMENT_MODES = Literal["full_default", "minimal_deployment", "verifiable"]


class VerificationConfigError(ValueError):
    pass


@dataclass
class VerificationConfig:
    # Confidence thresholds
    verified_confidence_threshold: float = field(default=0.70)
    rejected_confidence_threshold: float = field(default=0.30)
    low_confidence_range: Tuple[float, float] = field(default=(0.30, 0.70))

    # Multi-source and retrieval
    min_entailing_sources_for_verified: int = field(default=2)
    top_k_retrieval: int = field(default=20)
    top_k_rerank: int = field(default=5)
    mmr_lambda: float = field(default=0.5)

    # Temperature scaling / calibration
    temperature_scaling_enabled: bool = field(default=True)
    temperature_init: float = field(default=1.0)
    temperature_grid_min: float = field(default=0.8)
    temperature_grid_max: float = field(default=2.0)
    temperature_grid_steps: int = field(default=100)
    calibration_split: str = field(default="validation")

    # Baseline and heuristic thresholds (configurable for reproducible baselines)
    retriever_threshold: float = field(default=0.70)
    nli_positive_threshold: float = field(default=0.6)
    nli_negative_threshold: float = field(default=0.4)
    rag_positive_threshold: float = field(default=0.65)
    rag_negative_threshold: float = field(default=0.35)

    # Deployment mode flags (deterministic & reproducible)
    deployment_mode: DEPLOYMENT_MODES = field(default="full_default")
    
    # Optimization layer flags (configure per mode)
    enable_result_cache: bool = field(default=True)
    enable_quality_screening: bool = field(default=True)
    enable_query_expansion: bool = field(default=True)
    enable_evidence_ranker: bool = field(default=True)
    enable_type_classifier: bool = field(default=True)
    enable_semantic_deduplicator: bool = field(default=True)
    enable_adaptive_depth: bool = field(default=True)
    enable_priority_scorer: bool = field(default=True)

    # Misc
    random_seed: int = field(default=42)

    def __post_init__(self):
        self._validate()
        self._apply_deployment_mode()

    def _apply_deployment_mode(self) -> None:
        """Apply deployment mode settings to optimization flags."""
        if self.deployment_mode == "full_default":
            # All optimizations enabled (maximum throughput)
            self.enable_result_cache = True
            self.enable_quality_screening = True
            self.enable_query_expansion = True
            self.enable_evidence_ranker = True
            self.enable_type_classifier = True
            self.enable_semantic_deduplicator = True
            self.enable_adaptive_depth = True
            self.enable_priority_scorer = True
        elif self.deployment_mode == "minimal_deployment":
            # 75% cost savings: caching + screening only
            self.enable_result_cache = True
            self.enable_quality_screening = True
            self.enable_query_expansion = False
            self.enable_evidence_ranker = False
            self.enable_type_classifier = False
            self.enable_semantic_deduplicator = False
            self.enable_adaptive_depth = False
            self.enable_priority_scorer = False
        elif self.deployment_mode == "verifiable":
            # Minimal for verification only (baseline)
            self.enable_result_cache = False
            self.enable_quality_screening = False
            self.enable_query_expansion = False
            self.enable_evidence_ranker = False
            self.enable_type_classifier = False
            self.enable_semantic_deduplicator = False
            self.enable_adaptive_depth = False
            self.enable_priority_scorer = False

    def _validate(self) -> None:
        lo, hi = self.low_confidence_range
        if not (0.0 <= self.rejected_confidence_threshold <= lo < hi <= 1.0):
            raise VerificationConfigError(
                f"Invalid low_confidence_range={self.low_confidence_range} or rejected_confidence_threshold={self.rejected_confidence_threshold}. "
                "Expected 0.0 <= rejected < low < high <= 1.0"
            )
        if not (0 < self.min_entailing_sources_for_verified):
            raise VerificationConfigError("min_entailing_sources_for_verified must be >=1")
        if not (0.0 <= self.mmr_lambda <= 1.0):
            raise VerificationConfigError("mmr_lambda must be in [0.0, 1.0]")
        if self.top_k_retrieval < 1 or self.top_k_rerank < 1:
            raise VerificationConfigError("top_k_retrieval and top_k_rerank must be >=1")
        if self.temperature_grid_min <= 0 or self.temperature_grid_max <= 0:
            raise VerificationConfigError("temperature grid bounds must be > 0")
        if not (0.0 <= self.retriever_threshold <= 1.0):
            raise VerificationConfigError("retriever_threshold must be in [0,1]")
        if not (0.0 <= self.nli_positive_threshold <= 1.0 and 0.0 <= self.nli_negative_threshold <= 1.0):
            raise VerificationConfigError("nli thresholds must be in [0,1]")
        if self.nli_negative_threshold >= self.nli_positive_threshold:
            raise VerificationConfigError("nli_negative_threshold must be < nli_positive_threshold")
        if self.temperature_grid_steps < 2:
            raise VerificationConfigError("temperature_grid_steps must be >=2")
        if self.deployment_mode not in ("full_default", "minimal_deployment", "verifiable"):
            raise VerificationConfigError(f"Unknown deployment_mode: {self.deployment_mode}")

    @classmethod
    def from_env(cls) -> "VerificationConfig":
        """Construct config from environment variables (optional overrides)."""
        def env_float(name, default):
            v = os.getenv(name)
            return float(v) if v is not None else default

        def env_int(name, default):
            v = os.getenv(name)
            return int(v) if v is not None else default

        def env_bool(name, default):
            v = os.getenv(name)
            return v.lower() == "true" if isinstance(v, str) else default

        cfg = cls(
            verified_confidence_threshold=env_float("VERIFIED_CONFIDENCE_THRESHOLD", 0.70),
            rejected_confidence_threshold=env_float("REJECTED_CONFIDENCE_THRESHOLD", 0.30),
            low_confidence_range=(env_float("LOW_CONFIDENCE_LO", 0.30), env_float("LOW_CONFIDENCE_HI", 0.70)),
            min_entailing_sources_for_verified=env_int("MIN_ENTAILING_SOURCES_FOR_VERIFIED", 2),
            top_k_retrieval=env_int("TOP_K_RETRIEVAL", 20),
            top_k_rerank=env_int("TOP_K_RERANK", 5),
            mmr_lambda=env_float("MMR_LAMBDA", 0.5),
            temperature_scaling_enabled=env_bool("TEMPERATURE_SCALING_ENABLED", True),
            temperature_init=env_float("TEMPERATURE_INIT", 1.0),
            temperature_grid_min=env_float("TEMPERATURE_GRID_MIN", 0.8),
            temperature_grid_max=env_float("TEMPERATURE_GRID_MAX", 2.0),
            temperature_grid_steps=env_int("TEMPERATURE_GRID_STEPS", 100),
            calibration_split=os.getenv("CALIBRATION_SPLIT", "validation"),
            random_seed=env_int("GLOBAL_RANDOM_SEED", 42),
            retriever_threshold=env_float("RETRIEVER_THRESHOLD", 0.70),
            nli_positive_threshold=env_float("NLI_POSITIVE_THRESHOLD", 0.6),
            nli_negative_threshold=env_float("NLI_NEGATIVE_THRESHOLD", 0.4),
            rag_positive_threshold=env_float("RAG_POSITIVE_THRESHOLD", 0.65),
            rag_negative_threshold=env_float("RAG_NEGATIVE_THRESHOLD", 0.35),
            deployment_mode=os.getenv("DEPLOYMENT_MODE", "full_default"),  # type: ignore
        )
        return cfg

    def as_dict(self) -> dict:
        return {
            "verified_confidence_threshold": self.verified_confidence_threshold,
            "rejected_confidence_threshold": self.rejected_confidence_threshold,
            "low_confidence_range": list(self.low_confidence_range),
            "min_entailing_sources_for_verified": self.min_entailing_sources_for_verified,
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_rerank": self.top_k_rerank,
            "mmr_lambda": self.mmr_lambda,
            "temperature_scaling_enabled": self.temperature_scaling_enabled,
            "temperature_init": self.temperature_init,
            "temperature_grid_min": self.temperature_grid_min,
            "temperature_grid_max": self.temperature_grid_max,
            "temperature_grid_steps": self.temperature_grid_steps,
            "calibration_split": self.calibration_split,
            "random_seed": self.random_seed,
            "retriever_threshold": self.retriever_threshold,
            "nli_positive_threshold": self.nli_positive_threshold,
            "nli_negative_threshold": self.nli_negative_threshold,
            "rag_positive_threshold": self.rag_positive_threshold,
            "rag_negative_threshold": self.rag_negative_threshold,
            "deployment_mode": self.deployment_mode,
            "enable_result_cache": self.enable_result_cache,
            "enable_quality_screening": self.enable_quality_screening,
            "enable_query_expansion": self.enable_query_expansion,
            "enable_evidence_ranker": self.enable_evidence_ranker,
            "enable_type_classifier": self.enable_type_classifier,
            "enable_semantic_deduplicator": self.enable_semantic_deduplicator,
            "enable_adaptive_depth": self.enable_adaptive_depth,
            "enable_priority_scorer": self.enable_priority_scorer,
        }


__all__ = ["VerificationConfig", "VerificationConfigError", "DEPLOYMENT_MODES"]
