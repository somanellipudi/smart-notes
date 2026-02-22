"""
Verifiable reasoning pipeline integration.

This module wraps the standard reasoning pipeline with verifiable mode logic,
converting outputs to claims, retrieving evidence, validating, and filtering.
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

import config
from src.reasoning.pipeline import ReasoningPipeline
from src.schema.output_schema import ClassSessionOutput
from src.claims.schema import ClaimCollection, ClaimType, VerificationStatus, RejectionReason
from src.claims.extractor import ClaimExtractor
from src.retrieval.claim_rag import ClaimRAG
from src.claims.validator import ClaimValidator
from src.graph.claim_graph import ClaimGraph
from src.evaluation.verifiability_metrics import VerifiabilityMetrics
from src.agents.concept_agent import ConceptAgent
from src.agents.base import AgentRefusalError
from src.preprocessing.claim_preprocessor import ClaimPreprocessor
from src.policies.granularity_policy import enforce_granularity
from src.policies.evidence_policy import apply_sufficiency_policy
from src.policies.verification_policy import evaluate_claim, tag_claim_type
from src.verification.dependency_checker import check_dependencies, apply_dependency_enforcement
from src.policies.threat_model import get_threat_model_summary
from src.verification.diagnostics import VerificationDiagnostics
from src.retrieval.url_ingest import ingest_urls, chunk_url_sources, get_url_ingestion_summary
from src.preprocessing.text_quality import compute_text_quality, log_quality_report
from src.verification.rejection_analysis import (
    RejectionHistogram, VerificationDebugMetadata, create_verification_response_metadata
)
from src.retrieval.evidence_store import (
    EvidenceStore,
    validate_evidence_store,
    get_ingestion_diagnostics
)
from src.retrieval.evidence_builder import build_session_evidence_store
from src.retrieval.online_evidence_search import build_online_evidence_store
from src.retrieval.embedding_provider import EmbeddingProvider
from src.utils.performance_logger import PerformanceTimer, set_session_id
from src.utils.ml_optimizations import (
    SemanticDeduplicationCache,
    EvidenceQualityPredictor,
    ClaimPriorityScorer
)

logger = logging.getLogger(__name__)


class VerifiablePipelineWrapper:
    """
    Wrapper that adds verifiable mode to the standard reasoning pipeline.
    
    When verifiable_mode=False: delegates directly to standard pipeline
    When verifiable_mode=True: extracts claims, retrieves evidence, validates
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        provider_type: str = "openai",
        api_key: str = None,
        ollama_url: str = None,
        domain_profile: str = None
    ):
        """
        Initialize verifiable pipeline wrapper.
        
        Args:
            model: LLM model identifier
            temperature: Sampling temperature
            provider_type: LLM provider type
            api_key: API key for LLM provider
            ollama_url: Ollama server URL
            domain_profile: Domain profile name (physics, discrete_math, algorithms)
        """
        # Set global seed for reproducibility (if not already set)
        try:
            from src.utils.seed_control import set_global_seed
            if hasattr(config, 'GLOBAL_RANDOM_SEED'):
                set_global_seed(config.GLOBAL_RANDOM_SEED)
        except Exception:
            pass
        
        # Standard pipeline
        self.standard_pipeline = ReasoningPipeline(
            model=model,
            temperature=temperature,
            provider_type=provider_type,
            api_key=api_key,
            ollama_url=ollama_url
        )
        
        # Cited generation pipeline (FAST MODE)
        from src.reasoning.cited_pipeline import CitedGenerationPipeline
        self.cited_pipeline = CitedGenerationPipeline(
            model=model,
            temperature=temperature,
            provider_type=provider_type,
            api_key=api_key,
            embedding_provider=None  # Will be created in CitedGenerationPipeline
        )
        
        # Domain profile
        self.domain_profile = config.get_domain_profile(domain_profile)
        logger.info(f"Using domain profile: {self.domain_profile.display_name}")
        
        # Verifiable components
        self.claim_extractor = ClaimExtractor()
        self.evidence_retriever = ClaimRAG()
        self.embedding_provider = EmbeddingProvider.from_config()
        self.claim_validator = ClaimValidator(
            verified_threshold=config.VERIFIABLE_VERIFIED_THRESHOLD,
            rejected_threshold=config.VERIFIABLE_REJECTED_THRESHOLD,
            min_evidence_count=config.VERIFIABLE_MIN_EVIDENCE,
            strict_mode=config.VERIFIABLE_STRICT_MODE
        )
        self.metrics_calculator = VerifiabilityMetrics()
        
        # ML-based optimizations for performance
        from src.utils.ml_optimizations import (
            SemanticDeduplicationCache,
            EvidenceQualityPredictor,
            ClaimPriorityScorer
        )
        from src.utils.ml_advanced_optimizations import (
            get_claim_clusterer,
            get_query_expander,
            get_evidence_ranker,
            get_type_classifier
        )
        
        # Basic ML optimizations
        self.dedup_cache = SemanticDeduplicationCache(similarity_threshold=0.95)
        self.evidence_predictor = EvidenceQualityPredictor(
            min_similarity=0.4,
            min_sources=1,
            min_evidence_length=50
        )
        self.priority_scorer = ClaimPriorityScorer()
        
        # Advanced ML optimizations
        self.claim_clusterer = get_claim_clusterer()
        self.query_expander = get_query_expander()
        self.evidence_ranker = get_evidence_ranker()
        self.type_classifier = get_type_classifier()
        
        logger.info("ML optimizations enabled: cache, quality prediction, priority scoring, clustering, query expansion, evidence ranking, type classification")
        
        # Initialize agent with config
        agent_config = {
            "min_evidence_count": config.VERIFIABLE_MIN_EVIDENCE,
            "min_similarity": config.VERIFIABLE_RELEVANCE_THRESHOLD,
            "llm_provider": config.VERIFIABLE_CONSISTENCY_PROVIDER,
            "max_definition_length": 300
        }
        self.concept_agent = ConceptAgent(agent_config)
        self.preprocessor = ClaimPreprocessor()
        
        logger.info("VerifiablePipelineWrapper initialized")
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Get verification thresholds based on RELAXED_VERIFICATION_MODE.
        
        Returns:
            Dictionary with threshold values for evidence policy
        """
        if config.RELAXED_VERIFICATION_MODE:
            return {
                "min_entailment_prob": config.MIN_ENTAILMENT_PROB_RELAXED,
                "min_supporting_sources": config.MIN_SUPPORTING_SOURCES_RELAXED,
                "max_contradiction_prob": config.MAX_CONTRADICTION_PROB_RELAXED,
                "mode": "RELAXED"
            }
        else:
            return {
                "min_entailment_prob": config.MIN_ENTAILMENT_PROB_DEFAULT,
                "min_supporting_sources": config.MIN_SUPPORTING_SOURCES_DEFAULT,
                "max_contradiction_prob": config.MAX_CONTRADICTION_PROB_DEFAULT,
                "mode": "STRICT"
            }
    
    def process(
        self,
        combined_content: str,
        equations: List[str],
        external_context: str = "",
        session_id: str = None,
        verifiable_mode: bool = False,
        output_filters: Dict[str, bool] = None,
        urls: List[str] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Tuple[ClassSessionOutput, Optional[Dict[str, Any]]]:
        """
        Process content through pipeline with optional verifiable mode.
        
        Args:
            combined_content: Preprocessed classroom content
            equations: List of equations
            external_context: Optional reference material
            session_id: Unique session identifier
            verifiable_mode: If True, use verifiable mode; if False, standard mode
            output_filters: Dict of section names to booleans (e.g., {'summary': True, 'topics': False})
            urls: Optional list of URLs (YouTube videos or articles) to ingest as evidence sources
        
        Returns:
            Tuple of (output, verifiable_metadata)
            - output: ClassSessionOutput (standard format)
            - verifiable_metadata: Additional verifiable mode data (if enabled)
        """
        logger.info(f"Processing with verifiable_mode={verifiable_mode}, filters={output_filters}")
        
        if urls:
            logger.info(f"URLs provided for ingestion: {len(urls)}")
        
        if output_filters is None:
            output_filters = {
                'summary': True,
                'topics': True,
                'concepts': True,
                'equations': True,
                'misconceptions': True,
                'faqs': True,
                'worked_examples': True,
                'real_world': True
            }
        
        # FAST MODE: Cited generation (generates with sources in single pass)
        if verifiable_mode and config.ENABLE_CITED_GENERATION:
            logger.info("🚀 Running in CITED GENERATION mode (FAST - generates with sources)")
            output, metadata = self.cited_pipeline.process(
                combined_content=combined_content,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                output_filters=output_filters
            )
            
            # Convert concepts to synthetic claims with sources for report compatibility
            from src.claims.schema import ClaimCollection, LearningClaim, ClaimType, VerificationStatus
            
            synthetic_claims = []
            skipped_claims = []
            evidence_summary = metadata.get("evidence_summary", {})
            skipped_concepts = metadata.get("skipped_concepts", [])
            
            logger.info(f"🔍 Evidence summary from pipeline:")
            logger.info(f"   Keys: {list(evidence_summary.keys())}")
            logger.info(f"   Concepts to verify: {[c.name for c in output.key_concepts]}")
            
            verified_count = 0
            low_conf_count = 0
            
            for concept in output.key_concepts:
                # Try exact match first, then case-insensitive
                sources = evidence_summary.get(concept.name, [])
                if not sources:
                    # Try case-insensitive lookup
                    for key, val in evidence_summary.items():
                        if key.lower() == concept.name.lower():
                            sources = val
                            logger.info(f"Found case-insensitive match for {concept.name}")
                            break
                
                source_urls = [s.get("url") for s in sources] if sources else []
                has_sources = len(source_urls) > 0
                
                # STRICT VERIFICATION: Only VERIFIED if external sources found
                if has_sources and any(url not in ["Class Notes", "Class Notes / Primary Source"] for url in source_urls):
                    status = VerificationStatus.VERIFIED
                    verified_count += 1
                else:
                    # Mark as LOW_CONFIDENCE - requires external verification
                    status = VerificationStatus.LOW_CONFIDENCE
                    low_conf_count += 1
                
                logger.info(f"✓ Concept '{concept.name}': external_sources={has_sources}, status={status.value}, urls={len(source_urls)}")
                
                claim = LearningClaim(
                    claim_text=f"{concept.name}: {concept.definition}",
                    claim_type=ClaimType.DEFINITION,
                    status=status,
                    confidence=0.95 if has_sources else 0.5,  # Lower confidence for unverified
                    metadata={
                        "source": "cited_generation",
                        "concept_name": concept.name,
                        "source_urls": source_urls,
                        "source_count": len(source_urls),
                        "sources": sources,
                        "requires_verification": not has_sources
                    }
                )
                synthetic_claims.append(claim)
            
            # Create skipped claims for documentation
            for skipped in skipped_concepts:
                skip_claim = LearningClaim(
                    claim_text=f"{skipped['name']}: (Skipped - {skipped['reason']})",
                    claim_type=ClaimType.DEFINITION,
                    status=VerificationStatus.REJECTED,
                    confidence=0.0,
                    metadata={
                        "source": "cited_generation",
                        "skipped": True,
                        "skip_reason": skipped['reason']
                    }
                )
                skipped_claims.append(skip_claim)
            
            claim_collection = ClaimCollection(session_id=session_id, claims=synthetic_claims + skipped_claims)
            verified_collection = ClaimCollection(session_id=session_id, claims=synthetic_claims)  # Only verified, no skipped
            
            # Enhanced metadata to match verifiable format
            num_concepts = metadata.get("concepts", 0)
            num_with_sources = len([c for c in synthetic_claims if c.metadata.get("source_urls")])
            num_skipped = len(skipped_concepts)
            num_verified_citations = metadata.get("verified_citations", 0)
            
            # Debug logging
            logger.info(f"📊 Cited mode metadata: topics={metadata.get('topics')}, concepts={num_concepts}, citations={num_verified_citations}")
            logger.info(f"📝 Topics extracted: {metadata.get('topics_list', [])}")
            logger.info(f"📝 Concepts extracted: {metadata.get('concepts_list', [])}")
            logger.info(f"📚 Concepts with sources: {num_with_sources}/{num_concepts}")
            logger.info(f"⏭️  Concepts skipped (no evidence): {num_skipped}")
            if skipped_concepts:
                for skip in skipped_concepts:
                    logger.info(f"   - {skip['name']}: {skip['reason']}")
            
            if num_concepts == 0:
                logger.error("❌ CRITICAL: Cited generation returned 0 concepts! Check extraction logic.")
            
            logger.info(
                f"✅ Cited generation complete: {num_concepts} concepts, "
                f"{num_verified_citations} verified citations"
            )
            
            verifiable_metadata = {
                "verifiable_mode": True,
                "mode": "cited_generation",
                "processing_time": metadata.get("processing_time", 0),
                "metrics": {
                    "total_claims": num_concepts,
                    "verified_claims": num_with_sources,  # Only those with verified sources
                    "rejected_claims": num_skipped,
                    "low_confidence_claims": num_concepts - num_with_sources,
                    "evidence_docs": metadata.get("total_evidence", 0),
                    "verification_rate": num_with_sources / num_concepts if num_concepts > 0 else 0.0,
                    "quality_flags": [
                        f"Fast cited generation: {num_verified_citations} online citations",
                        f"Sources verified: {num_with_sources}/{num_concepts} concepts"
                    ]
                },
                "citations": metadata.get("citations", []),
                "claim_collection": claim_collection,
                "verified_collection": verified_collection,
                "fast_mode": True,
                "evidence_summary": evidence_summary,
                "skipped_concepts": skipped_concepts
            }
            return output, verifiable_metadata
        
        if not verifiable_mode:
            # Standard mode: delegate directly to standard pipeline
            logger.info("Running in STANDARD mode")
            output = self.standard_pipeline.process(
                combined_content=combined_content,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                output_filters=output_filters
            )
            return output, None
        
        else:
            # Verifiable mode: extract claims, retrieve evidence, validate
            logger.info("Running in VERIFIABLE mode")
            return self._process_verifiable(
                combined_content=combined_content,
                equations=equations,
                external_context=external_context,
                session_id=session_id,
                output_filters=output_filters,
                urls=urls,
                progress_callback=progress_callback
            )
    
    def _process_verifiable(
        self,
        combined_content: str,
        equations: List[str],
        external_context: str,
        session_id: str,
        output_filters: Dict[str, bool] = None,
        urls: List[str] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> Tuple[ClassSessionOutput, Dict[str, Any]]:
        """
        Process content in verifiable mode.
        
        Steps:
        1. Run standard pipeline to get baseline output
        2. Extract claims from baseline output
        3. Retrieve evidence for claims
        4. Validate and filter claims
        5. Build claim-evidence graph
        6. Calculate verifiability metrics
        7. Return both standard output and verifiable metadata
        
        Args:
            combined_content: Classroom content
            equations: Equations
            external_context: Reference material
            session_id: Session ID
        
        Returns:
            Tuple of (output, verifiable_metadata)
        """
        start_time = time.perf_counter()
        step_timings: Dict[str, float] = {}

        def _update_progress(stage: str, status: str) -> None:
            if progress_callback:
                try:
                    progress_callback(stage, status)
                except Exception:
                    pass
        
        # Initialize diagnostics if enabled
        diagnostics = VerificationDiagnostics(session_id) if config.DEBUG_VERIFICATION else None
        if config.RELAXED_VERIFICATION_MODE:
            logger.warning("="*70)
            logger.warning("Running in RELAXED_VERIFICATION_MODE")
            logger.warning(f"  MIN_ENTAILMENT_PROB: {config.RELAXED_MIN_ENTAILMENT_PROB}")
            logger.warning(f"  MIN_SUPPORTING_SOURCES: {config.RELAXED_MIN_SUPPORTING_SOURCES}")
            logger.warning(f"  MAX_CONTRADICTION_PROB: {config.RELAXED_MAX_CONTRADICTION_PROB}")
            logger.warning("="*70)
        
        # Initialize rejection histogram for tracking verification results
        rejection_histogram = RejectionHistogram()
        debug_metadata = VerificationDebugMetadata()
        
        # Step 0: Ingest URL sources if provided
        url_sources = []
        url_chunks = []
        url_ingestion_summary = None
        
        if urls and config.ENABLE_URL_SOURCES:
            logger.info(f"Step 0: Ingesting {len(urls)} URL sources")
            step_start = time.perf_counter()
            
            try:
                url_sources = ingest_urls(urls)
                url_chunks = chunk_url_sources(url_sources, chunk_size=500, overlap=50)
                url_ingestion_summary = get_url_ingestion_summary(url_sources)
                
                logger.info(
                    f"URL ingestion complete: {url_ingestion_summary['successful']}/{url_ingestion_summary['total_urls']} successful, "
                    f"{len(url_chunks)} chunks, {url_ingestion_summary['total_chars']} total chars"
                )
                
                if url_ingestion_summary['failed'] > 0:
                    logger.warning(f"{url_ingestion_summary['failed']} URL(s) failed to ingest")
                    for url_info in url_ingestion_summary['urls']:
                        if not url_info['success']:
                            logger.warning(f"  - {url_info['url']}: {url_info['error']}")
            
            except Exception as e:
                logger.error(f"URL ingestion failed: {e}")
                url_ingestion_summary = {
                    "total_urls": len(urls),
                    "successful": 0,
                    "failed": len(urls),
                    "error": str(e)
                }
            
            step_timings["step_0_ingest_urls"] = time.perf_counter() - step_start
            logger.info(f"Step 0 time: {step_timings['step_0_ingest_urls']:.2f}s")
        elif urls and not config.ENABLE_URL_SOURCES:
            logger.warning("URLs provided but ENABLE_URL_SOURCES is False, skipping URL ingestion")
        
        # Step 0.25: Check text quality and detect extraction failures
        logger.info("Step 0.25: Assessing input text quality")
        step_start = time.perf_counter()
        
        quality_report = compute_text_quality(combined_content)
        log_quality_report(quality_report, context="Input text assessment")
        
        if quality_report.is_unverifiable:
            # Input text below absolute minimum - return UNVERIFIABLE_INPUT
            logger.error(
                f"❌ Input text unverifiable: {quality_report.text_length} chars "
                f"< {config.MIN_INPUT_CHARS_ABSOLUTE} absolute minimum"
            )
            logger.error("Failure reasons:")
            for reason in quality_report.failure_reasons:
                logger.error(f"  - {reason}")
            
            # Return early with UNVERIFIABLE_INPUT status
            # Generate baseline output but mark it as unverifiable
            try:
                baseline_output = self.standard_pipeline.process(
                    combined_content=combined_content,
                    equations=equations,
                    external_context=external_context,
                    session_id=session_id,
                    output_filters=output_filters
                )
            except Exception as e:
                logger.warning(f"Could not generate baseline output: {str(e)}")
                baseline_output = None
            
            verifiable_metadata = {
                "verifiable_mode": False,
                "status": "UNVERIFIABLE_INPUT",
                "reason": "Input text too short for verification",
                "failure_reasons": quality_report.failure_reasons,
                "text_length": quality_report.text_length,
                "minimum_required": config.MIN_INPUT_CHARS_ABSOLUTE,
                "suggestion": f"Please provide at least {config.MIN_INPUT_CHARS_ABSOLUTE} characters of input text",
                "quality_report": {
                    'alphabetic_ratio': quality_report.alphabetic_ratio,
                    'cid_ratio': quality_report.cid_ratio,
                    'printable_ratio': quality_report.printable_ratio,
                },
                "metrics": {
                    "total_claims": 0,
                    "verified_claims": 0,
                    "rejected_claims": 0,
                    "low_confidence_claims": 0,
                    "rejection_rate": 0.0,
                    "verification_rate": 0.0,
                    "low_confidence_rate": 0.0,
                    "avg_confidence": 0.0,
                    "rejection_reasons": {},
                    "evidence_metrics": {
                        "total_evidence": 0,
                        "avg_evidence_per_claim": 0.0,
                        "unsupported_rate": 1.0,
                        "avg_evidence_quality": 0.0
                    },
                    "graph_metrics": None,
                    "baseline_comparison": None,
                    "negative_control": True,
                    "quality_flags": [
                        "Input text too short for verification",
                        f"Text length: {quality_report.text_length} chars < {config.MIN_INPUT_CHARS_ABSOLUTE} minimum"
                    ]
                }
            }
            
            # If OCR fallback might help (CID glyphs detected), mention it
            if quality_report.cid_ratio > config.MAX_CID_RATIO and config.ENABLE_OCR_FALLBACK:
                verifiable_metadata["suggestion"] += (
                    f"\n\nAlternatively, if this is a PDF: "
                    f"The high CID glyph ratio ({quality_report.cid_ratio:.4f}) indicates "
                    f"extraction corruption. Retry with OCR fallback enabled."
                )
            
            # Ensure we return a ClassSessionOutput, not a dict
            if baseline_output is None:
                from src.schema.output_schema import ClassSessionOutput
                baseline_output = ClassSessionOutput(
                    session_id=session_id,
                    class_summary="Unable to process input. The provided content does not meet minimum requirements for analysis. Please check input quality and provide at least 100 characters of substantive text, then try again.",
                    topics=[],
                    key_concepts=[],
                    equation_explanations=[],
                    worked_examples=[],
                    common_mistakes=[],
                    faqs=[],
                    real_world_connections=[],
                    metadata={"verifiable": False, "status": "UNVERIFIABLE_INPUT", "error": "baseline_generation_failed"}
                )
            
            return baseline_output, verifiable_metadata
        
        step_timings["step_0_25_quality_assessment"] = time.perf_counter() - step_start
        logger.info(f"Step 0.25 time: {step_timings['step_0_25_quality_assessment']:.2f}s")
        
        # Set session ID for performance logging
        set_session_id(session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Determine evidence source mode
        evidence_source_mode = getattr(config, 'EVIDENCE_SOURCE_MODE', 'input')
        logger.info(f"Evidence source mode: {evidence_source_mode}")
        
        # Step 0.5: Build Evidence Store (if using INPUT mode)
        # For ONLINE mode, evidence store will be built AFTER claim extraction (Step 2)
        evidence_store = None
        evidence_stats = None
        
        if evidence_source_mode == "input":
            logger.info("Step 0.5: Building session evidence store from INPUT")
            step_start = time.perf_counter()
            
            try:
                with PerformanceTimer("build_input_evidence_store", session_id=session_id):
                    # Build evidence store from all inputs
                    evidence_store, evidence_stats = build_session_evidence_store(
                        session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        input_text=combined_content,
                        external_context=external_context,
                        equations=equations,
                        urls=urls if config.ENABLE_URL_SOURCES else None,
                        min_input_chars=config.MIN_INPUT_CHARS_FOR_VERIFICATION,
                        embedding_provider=self.embedding_provider
                    )
            
                if not evidence_store.index_built:
                    raise ValueError("Evidence index not built; dense retrieval unavailable")
                
                # Validate evidence store (non-blocking)
                is_valid, validation_msg, classification = validate_evidence_store(
                    evidence_store,
                    min_chars=config.MIN_INPUT_CHARS_FOR_VERIFICATION,
                    strict=False
                )
                
                # Log evidence statistics
                ev_stats = evidence_store.get_statistics()
                evidence_stats = ev_stats
                logger.info(
                    f"Evidence store stats: {ev_stats['num_chunks']} chunks, "
                    f"{ev_stats['total_chars']} chars, {ev_stats['num_sources']} sources"
                )

                if not is_valid:
                    logger.warning("Insufficient evidence for verification mode. Falling back to baseline mode.")
                    baseline_output = self.standard_pipeline.process(
                        combined_content=combined_content,
                        equations=equations,
                        external_context=external_context,
                        session_id=session_id,
                        output_filters=output_filters
                    )

                    verifiable_metadata = {
                        "verifiable_mode": False,
                        "verification_unavailable": True,
                        "status": classification,
                        "reason": validation_msg,
                        "message": (
                            "Verification unavailable due to insufficient evidence. "
                            "Showing baseline study guide (unverified content)."
                        ),
                        "metrics": {
                            "total_claims": 0,
                            "verified_claims": 0,
                            "rejected_claims": 0,
                            "evidence_docs": 0,
                            "quality_flags": [validation_msg]
                        },
                        "evidence_diagnostics": get_ingestion_diagnostics(
                            evidence_store,
                            min_chars=config.MIN_INPUT_CHARS_FOR_VERIFICATION
                        )
                    }

                    return baseline_output, verifiable_metadata

                # Store evidence stats for diagnostics
                if diagnostics:
                    diagnostics.evidence_stats = ev_stats

            except ValueError as e:
                error_msg = f"Evidence validation skipped: {str(e)}"
                logger.warning(f"⚠ {error_msg}")
                logger.warning("Insufficient evidence for verification mode. Falling back to baseline mode.")

                baseline_output = self.standard_pipeline.process(
                    combined_content=combined_content,
                    equations=equations,
                    external_context=external_context,
                    session_id=session_id,
                    output_filters=output_filters
                )

                verifiable_metadata = {
                    "verifiable_mode": False,
                    "verification_unavailable": True,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason": str(e),
                    "message": (
                        "Verification unavailable due to insufficient evidence. "
                        "Showing baseline study guide (unverified content)."
                    ),
                    "metrics": {
                        "total_claims": 0,
                        "verified_claims": 0,
                        "rejected_claims": 0,
                        "evidence_docs": 0,
                        "quality_flags": [error_msg]
                    }
                }

                return baseline_output, verifiable_metadata
            
            step_timings["step_0_5_build_evidence_store"] = time.perf_counter() - step_start
            logger.info(f"Step 0.5 time: {step_timings['step_0_5_build_evidence_store']:.2f}s")
        
        else:
            # ONLINE mode: Evidence store will be built after claim extraction
            logger.info("Step 0.5: Skipping input evidence store (ONLINE mode - will build after claims)")
            step_timings["step_0_5_build_evidence_store"] = 0.0
        
        # Step 1: Run standard pipeline to get baseline
        logger.info("Step 1: Running standard pipeline for baseline")
        step_start = time.perf_counter()
        baseline_output = self.standard_pipeline.process(
            combined_content=combined_content,
            equations=equations,
            external_context=external_context,
            session_id=session_id,
            output_filters=output_filters
        )
        step_timings["step_1_standard_pipeline"] = time.perf_counter() - step_start
        logger.info(f"Step 1 time: {step_timings['step_1_standard_pipeline']:.2f}s")
        
        # Step 2: Extract claims from baseline
        logger.info("Step 2: Extracting claims from baseline output")
        _update_progress("claim_extraction", "running")
        step_start = time.perf_counter()
        claim_collection = self.claim_extractor.extract_from_session(baseline_output)
        
        # Filter out questions and claims with skip_verification flag
        initial_count = len(claim_collection.claims)
        claim_collection.claims = [
            c for c in claim_collection.claims 
            if not c.metadata.get("skip_verification", False) 
            and c.claim_type != ClaimType.QUESTION
        ]
        filtered_count = initial_count - len(claim_collection.claims)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} non-verifiable claims (questions, etc.)")
        
        step_timings["step_2_extract_claims"] = time.perf_counter() - step_start
        logger.info(f"Step 2 time: {step_timings['step_2_extract_claims']:.2f}s")
        _update_progress("claim_extraction", "complete")
        
        # Step 2.5: Enforce claim granularity (split compound claims)
        logger.info("Step 2.5: Enforcing claim granularity policy")
        step_start = time.perf_counter()
        atomic_claims = enforce_granularity(
            claim_collection.claims,
            max_propositions=config.MAX_PROPOSITIONS_PER_CLAIM
        )
        claim_collection.claims = atomic_claims
        for claim in claim_collection.claims:
            tag_claim_type(claim, self.domain_profile)
        step_timings["step_2_5_enforce_granularity"] = time.perf_counter() - step_start
        logger.info(f"Step 2.5 time: {step_timings['step_2_5_enforce_granularity']:.2f}s")
        
        # Step 2.75: Build online evidence store (if using ONLINE mode)
        if evidence_source_mode == "online" and evidence_store is None:
            logger.info("Step 2.75: Building evidence store from ONLINE sources")
            step_start = time.perf_counter()
            
            try:
                max_claims_for_search = getattr(config, 'ONLINE_MAX_CLAIMS_TO_SEARCH', 20)
                claims_to_search = claim_collection.claims[:max_claims_for_search]
                
                logger.info(f"Searching online evidence for {len(claims_to_search)} claims")
                
                evidence_store, evidence_stats = build_online_evidence_store(
                    session_id=session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    claims=claims_to_search,
                    embedding_provider=self.embedding_provider,
                    max_evidence_per_claim=getattr(config, 'ONLINE_MAX_SOURCES_PER_CLAIM', 5)
                )
                
                # Log evidence statistics
                if evidence_stats:
                    logger.info(
                        f"Online evidence stats: {evidence_stats.get('total_evidence_chunks', 0)} chunks, "
                        f"{evidence_stats.get('num_sources', 0)} online sources"
                    )
                    
                    # Store evidence stats for diagnostics
                    if diagnostics:
                        diagnostics.evidence_stats = evidence_stats
                
                # Validate that we got useful evidence
                if not evidence_store or not evidence_store.index_built:
                    logger.warning("Failed to build online evidence store. Falling back to baseline mode.")
                    
                    verifiable_metadata = {
                        "verifiable_mode": False,
                        "verification_unavailable": True,
                        "status": "ONLINE_EVIDENCE_UNAVAILABLE",
                        "reason": "Could not retrieve evidence from online sources",
                        "message": (
                            "Online evidence retrieval unavailable. "
                            "Showing baseline study guide (unverified content)."
                        ),
                        "metrics": {
                            "total_claims": len(claim_collection.claims),
                            "verified_claims": 0,
                            "rejected_claims": 0,
                            "evidence_docs": 0,
                            "quality_flags": ["Online evidence unavailable"]
                        }
                    }
                    
                    return baseline_output, verifiable_metadata
                    
            except Exception as e:
                logger.error(f"Online evidence retrieval failed: {e}", exc_info=True)
                logger.warning("Falling back to baseline mode due to online evidence error.")
                
                verifiable_metadata = {
                    "verifiable_mode": False,
                    "verification_unavailable": True,
                    "status": "ONLINE_EVIDENCE_ERROR",
                    "reason": str(e),
                    "message": (
                        f"Online evidence retrieval error: {str(e)}. "
                        "Showing baseline study guide (unverified content)."
                    ),
                    "metrics": {
                        "total_claims": len(claim_collection.claims),
                        "verified_claims": 0,
                        "rejected_claims": 0,
                        "evidence_docs": 0,
                        "quality_flags": [f"Error: {str(e)}"]
                    }
                }
                
                return baseline_output, verifiable_metadata
            
            step_timings["step_2_75_build_online_evidence"] = time.perf_counter() - step_start
            logger.info(f"Step 2.75 time: {step_timings['step_2_75_build_online_evidence']:.2f}s")
        
        # Step 3: Retrieve evidence for each claim
        logger.info("Step 3: Retrieving evidence for claims")
        _update_progress("retrieval", "running")
        step_start = time.perf_counter()
        
        # Use evidence store built in Step 0.5
        claim_texts = [
            (claim.claim_text or claim.metadata.get("draft_text", ""))
            for claim in claim_collection.claims
        ]
        logger.info(f"Encoding {len(claim_texts)} claims for evidence retrieval")

        claim_embeddings = self.embedding_provider.embed_queries(claim_texts)

        retrieval_top_k = config.DENSE_RETRIEVAL_TOP_K
        min_similarity = config.DENSE_RETRIEVAL_MIN_SIMILARITY
        use_reranker = config.ENABLE_RERANKER
        rerank_top_k = config.RERANKER_TOP_K if use_reranker else retrieval_top_k
        final_top_k = config.RERANKER_KEEP_K if use_reranker else config.VERIFIABLE_MAX_EVIDENCE_PER_CLAIM

        def _sigmoid(value: float) -> float:
            return 1.0 / (1.0 + math.exp(-value))

        # PHASE 1: Retrieve candidates for ALL claims (parallelizable with FAISS)
        logger.info("Retrieving initial candidates for all claims...")
        all_retrieved = []
        for i, claim in enumerate(claim_collection.claims):
            query_embedding = claim_embeddings[i].astype("float32")
            retrieved = evidence_store.search(
                query_embedding=query_embedding,
                top_k=rerank_top_k,
                min_similarity=min_similarity
            )
            all_retrieved.append((claim, retrieved))
        
        # PHASE 2: Batch reranking (if enabled)
        if use_reranker:
            logger.info("Batch reranking all claim-evidence pairs...")
            with PerformanceTimer("batch_rerank_all_claims", session_id=session_id):
                # Build all (claim, passage) pairs with metadata
                batch_pairs = []
                batch_metadata = []  # (claim_idx, passage_idx, ev, vec_sim)
                
                for claim_idx, (claim, retrieved) in enumerate(all_retrieved):
                    if retrieved:
                        for passage_idx, (ev, vec_sim) in enumerate(retrieved):
                            batch_pairs.append((claim.claim_text, ev.text))
                            batch_metadata.append((claim_idx, passage_idx, ev, vec_sim))
                
                # Single batch rerank call for ALL pairs
                if batch_pairs:
                    logger.info(f"Reranking {len(batch_pairs)} claim-evidence pairs in one batch...")
                    all_raw_scores = []
                    
                    # Batch in chunks to avoid OOM (max 256 pairs per batch)
                    batch_size = 256
                    for chunk_start in range(0, len(batch_pairs), batch_size):
                        chunk_end = min(chunk_start + batch_size, len(batch_pairs))
                        chunk_pairs = batch_pairs[chunk_start:chunk_end]
                        
                        # Call reranker on chunk
                        reranker = self.embedding_provider._load_reranker()
                        if reranker:
                            chunk_scores = reranker.predict(chunk_pairs, show_progress_bar=False)
                            all_raw_scores.extend([float(s) for s in chunk_scores])
                        else:
                            all_raw_scores.extend([0.0] * len(chunk_pairs))
                    
                    # Now group scores back by claim
                    claim_rerank_results = [[] for _ in range(len(claim_collection.claims))]
                    for (claim_idx, passage_idx, ev, vec_sim), raw_score in zip(batch_metadata, all_raw_scores):
                        claim_rerank_results[claim_idx].append((ev, vec_sim, raw_score))
                else:
                    claim_rerank_results = [[] for _ in range(len(claim_collection.claims))]
        else:
            claim_rerank_results = [[] for _ in range(len(claim_collection.claims))]
        
        # PHASE 3: Assign evidence to claims
        logger.info("Assigning reranked evidence to claims...")
        all_nli_results = []
        from src.claims.schema import EvidenceItem
        
        for i, claim in enumerate(claim_collection.claims):
            retrieved = all_retrieved[i][1]
            
            # Use reranked scores if available
            if use_reranker and claim_rerank_results[i]:
                reranked = []
                for ev, vec_sim, raw_score in claim_rerank_results[i]:
                    reranked.append((ev, _sigmoid(raw_score), vec_sim, raw_score))
                reranked.sort(key=lambda item: item[1], reverse=True)
                reranked = reranked[:final_top_k]
            elif retrieved:
                # No reranking, use vector similarity
                reranked = [(ev, sim, sim, None) for ev, sim in retrieved[:final_top_k]]
            else:
                reranked = []
            
            # Build evidence objects
            evidence = []
            for ev, score, vec_sim, raw_score in reranked:
                span_metadata = {"vector_similarity": vec_sim}
                if raw_score is not None:
                    span_metadata["rerank_score"] = raw_score
                evidence_obj = EvidenceItem(
                    source_id=ev.source_id,
                    source_type=ev.source_type,
                    snippet=ev.text,
                    span_metadata=span_metadata,
                    similarity=score,
                    reliability_prior=0.8
                )
                evidence.append(evidence_obj)
            
            claim.evidence_objects = evidence
            claim.evidence_ids = [ev.evidence_id for ev in evidence]

            decision = evaluate_claim(
                claim=claim,
                evidence_items=evidence,
                domain_profile=self.domain_profile,
                nli_results=claim.metadata.get("nli_results") if hasattr(claim, "metadata") else None
            )
            claim.status = decision.status
            claim.confidence = decision.confidence
            claim.rejection_reason = decision.rejection_reason
            
            # Track in rejection histogram
            rejection_histogram.add_claim_result(
                claim_text=claim.claim_text,
                status=claim.status,
                rejection_reason=(
                    claim.rejection_reason.value
                    if claim.status == VerificationStatus.REJECTED and claim.rejection_reason
                    else None
                ),
                retrieval_hit_count=len(evidence)
            )
            
            # Log diagnostics if enabled
            if diagnostics:
                # Extract metrics from evidence
                similarities = [ev.similarity for ev in evidence if hasattr(ev, 'similarity')]
                entailments = []
                contradictions = []
                
                # Try to extract NLI results if available
                if hasattr(claim, 'metadata') and 'nli_results' in claim.metadata:
                    nli_data = claim.metadata['nli_results']
                    for nli in nli_data:
                        if nli.get('label') == 'entailment':
                            entailments.append(nli.get('score', 0.0))
                        elif nli.get('label') == 'contradiction':
                            contradictions.append(nli.get('score', 0.0))
                    all_nli_results.extend(nli_data)
                
                # Count independent sources
                from src.policies.evidence_policy import count_independent_sources
                independent_sources = count_independent_sources(evidence)
                
                # Log claim verification
                diagnostics.log_claim_verification(
                    claim=claim,
                    retrieved_passages=len(evidence),
                    similarities=similarities,
                    entailments=entailments,
                    contradictions=contradictions,
                    independent_sources=independent_sources,
                    status=claim.status,
                    reason=claim.rejection_reason
                )
                
                # Log retrieval health if enabled
                if config.DEBUG_RETRIEVAL_HEALTH and evidence:
                    max_sim = max(similarities) if similarities else 0.0
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                    diagnostics.log_retrieval_diagnostics(
                        max_similarity=max_sim,
                        avg_similarity=avg_sim,
                        num_candidates=len(evidence)
                    )
        
        _update_progress("nli", "running")

        # Log NLI distribution if enabled
        if diagnostics and config.DEBUG_NLI_DISTRIBUTION and all_nli_results:
            diagnostics.log_nli_distribution(all_nli_results)

        _update_progress("nli", "complete")
        
        step_timings["step_3_retrieve_evidence"] = time.perf_counter() - step_start
        logger.info(f"Step 3 time: {step_timings['step_3_retrieve_evidence']:.2f}s")
        _update_progress("retrieval", "complete")

        # Step 3.5: Evidence-first claim text generation (PARALLEL)
        logger.info("Step 3.5: Generating claim text only after evidence (with parallel LLM calls)")
        step_start = time.perf_counter()
        
        with PerformanceTimer("step_3_5_generate_text_parallel", session_id=session_id):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Separate claims by type
            definition_claims = []
            other_claims = []
            
            for claim in claim_collection.claims:
                if claim.status == VerificationStatus.REJECTED:
                    continue
                if not claim.evidence_objects:
                    claim.status = VerificationStatus.REJECTED
                    claim.confidence = 0.0
                    continue
                
                if claim.claim_type == ClaimType.DEFINITION:
                    definition_claims.append(claim)
                else:
                    other_claims.append(claim)
            
            # Process non-definition claims (fast, no LLM)
            for claim in other_claims:
                draft_text = claim.metadata.get("draft_text", "")
                if draft_text:
                    claim.claim_text = draft_text
                else:
                    claim.status = VerificationStatus.REJECTED
                    claim.confidence = 0.0
            
            # PARALLEL LLM calls for DEFINITION claims with ML optimizations
            if definition_claims:
                logger.info(f"Generating definitions for {len(definition_claims)} claims (with ML optimizations)...")
                
                # Compute embeddings for deduplication AND clustering
                claim_texts_for_cache = [
                    claim.metadata.get("draft_text", claim.claim_text) for claim in definition_claims
                ]
                claim_embeddings_for_cache = self.embedding_provider.embed_queries(claim_texts_for_cache)
                
                # ML Optimization: Cluster similar claims for batch processing
                clusters = self.claim_clusterer.cluster_claims(definition_claims, claim_embeddings_for_cache)
                logger.info(f"Clustered {len(definition_claims)} claims into {len(clusters)} batches")
                
                def generate_definition(claim_idx_tuple):
                    """Generate definition text for a single claim (parallel execution).""" 
                    idx, claim = claim_idx_tuple
                    
                    # ML Optimization 1: Evidence quality check (skip bad evidence early)
                    is_sufficient, reason = self.evidence_predictor.predict_quality(
                        claim.evidence_objects,
                        claim.metadata.get("draft_text", claim.claim_text)
                    )
                    if not is_sufficient:
                        logger.warning(
                            f"⚠️ Skipping LLM call for claim '{claim.metadata.get('draft_text', '')[:50]}...' - "
                            f"Reason: {reason} | "
                            f"Evidence count: {len(claim.evidence_objects)} | "
                            f"Max similarity: {max((ev.similarity for ev in claim.evidence_objects), default=0.0):.3f}"
                        )
                        return claim, None, f"evidence_quality_{reason}"
                    
                    # ML Optimization 2: Semantic deduplication (check cache)
                    claim_embedding = claim_embeddings_for_cache[idx]
                    cached_result = self.dedup_cache.lookup(claim_embedding, claim.evidence_objects)
                    if cached_result:
                        logger.info("Using cached result (semantic match)")
                        return claim, cached_result, None
                    
                    try:
                        with PerformanceTimer(
                            "concept_agent_generate",
                            session_id=session_id,
                            metadata={"claim_preview": claim.metadata.get("draft_text", "")[:30]}
                        ):
                            generated_text = self.concept_agent.generate(claim, claim.evidence_objects)
                            
                            # Store in cache for future reuse
                            self.dedup_cache.store(
                                claim_text=claim.metadata.get("draft_text", claim.claim_text),
                                claim_embedding=claim_embedding,
                                evidence_items=claim.evidence_objects,
                                generated_text=generated_text,
                                confidence=claim.confidence
                            )
                            
                            return claim, generated_text, None
                    except AgentRefusalError as e:
                        logger.info(f"Agent refused to generate: {e}")
                        return claim, None, e
                    except Exception as e:
                        logger.error(f"Error generating definition: {e}", exc_info=True)
                        return claim, None, e
                
                # ML Optimization 3: Priority-based processing (process important claims first)
                evidence_map = {i: claim.evidence_objects for i, claim in enumerate(definition_claims)}
                ranked_claims = self.priority_scorer.rank_claims(definition_claims, evidence_map)
                
                logger.info(f"Claim priorities: {[(i, f'{score:.2f}') for i, score in ranked_claims[:5]]}")
                
                # Execute LLM calls in parallel (max 3 concurrent to avoid rate limits)
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit in priority order
                    futures = {
                        executor.submit(generate_definition, (i, definition_claims[i])): i 
                        for i, score in ranked_claims
                    }
                    
                    for future in as_completed(futures):
                        claim, generated_text, error = future.result()
                        if error:
                            claim.status = VerificationStatus.REJECTED
                            claim.confidence = 0.0
                            claim.rejection_reason = RejectionReason.NO_EVIDENCE if "evidence" in str(error) else None
                            logger.info(f"Claim rejected (ML optimization): {error}")
                        else:
                            claim.claim_text = generated_text
                            # Confidence already set by decision policy
                
                # Log ML optimization stats
                cache_stats = self.dedup_cache.get_stats()
                predictor_stats = self.evidence_predictor.get_stats()
                logger.info(
                    f"ML Optimization Results: "
                    f"Cache hits={cache_stats['hits']} (saved {cache_stats['llm_calls_saved']} LLM calls), "
                    f"Quality skips={predictor_stats['skipped']} (saved {predictor_stats['llm_calls_saved']} LLM calls)"
                )
        
        step_timings["step_3_5_generate_text"] = time.perf_counter() - step_start
        logger.info(f"Step 3.5 time: {step_timings['step_3_5_generate_text']:.2f}s")
        
        # Step 4: Validate claims
        logger.info("Step 4: Validating claims")
        _update_progress("decision_policy", "running")
        step_start = time.perf_counter()
        claim_collection = self.claim_validator.validate_collection(claim_collection)
        step_timings["step_4_validate_claims"] = time.perf_counter() - step_start
        logger.info(f"Step 4 time: {step_timings['step_4_validate_claims']:.2f}s")
        _update_progress("decision_policy", "complete")
        
        # Step 5: Build claim-evidence graph and compute metrics
        logger.info("Step 5: Building claim-evidence graph")
        step_start = time.perf_counter()
        graph = ClaimGraph(claim_collection.claims)
        graph_metrics = graph.compute_metrics()
        step_timings["step_5_graph_metrics"] = time.perf_counter() - step_start
        logger.info(f"Step 5 time: {step_timings['step_5_graph_metrics']:.2f}s")
        
        # Step 5.5: Check cross-claim dependencies
        logger.info("Step 5.5: Checking cross-claim dependencies")
        step_start = time.perf_counter()
        dependency_warnings = check_dependencies(
            claim_collection.claims,
            strict_mode=self.domain_profile.strict_dependencies
        )
        if config.ENABLE_DEPENDENCY_WARNINGS and dependency_warnings:
            logger.info(f"Found {len(dependency_warnings)} dependency warnings")
            # Apply enforcement if strict mode is enabled
            if config.STRICT_DEPENDENCY_ENFORCEMENT or self.domain_profile.strict_dependencies:
                claim_collection.claims = apply_dependency_enforcement(
                    claim_collection.claims,
                    dependency_warnings,
                    downgrade_to_low_confidence=True
                )
        step_timings["step_5_5_check_dependencies"] = time.perf_counter() - step_start
        logger.info(f"Step 5.5 time: {step_timings['step_5_5_check_dependencies']:.2f}s")
        
        # Step 6: Calculate metrics
        logger.info("Step 6: Calculating verifiability metrics")
        step_start = time.perf_counter()
        metrics = self.metrics_calculator.calculate_metrics(
            claim_collection=claim_collection,
            graph_metrics=graph_metrics,
            baseline_output=baseline_output.to_dict()
        )
        step_timings["step_6_calculate_metrics"] = time.perf_counter() - step_start
        logger.info(f"Step 6 time: {step_timings['step_6_calculate_metrics']:.2f}s")
        
        # Step 7: Filter to verified claims only for output
        logger.info("Step 7: Filtering to verified claims")
        step_start = time.perf_counter()
        verified_collection = self.claim_validator.filter_collection(
            claim_collection,
            include_verified=True,
            include_low_confidence=False,  # Exclude low-confidence
            include_rejected=False  # Exclude rejected
        )
        
        # Log filtering breakdown
        total_claims = len(claim_collection.claims)
        verified_count = len(verified_collection.claims)
        rejected_count = sum(1 for c in claim_collection.claims if c.status == VerificationStatus.REJECTED)
        low_conf_count = sum(1 for c in claim_collection.claims if c.status == VerificationStatus.LOW_CONFIDENCE)
        
        logger.info(
            f"📊 Step 7 Filtering Results: "
            f"Total={total_claims}, "
            f"Verified={verified_count}, "
            f"Rejected={rejected_count}, "
            f"Low-Confidence={low_conf_count}"
        )
        
        if rejected_count > 0:
            # Log top rejection reasons
            rejection_reasons = {}
            for claim in claim_collection.claims:
                if claim.status == VerificationStatus.REJECTED:
                    reason = str(claim.rejection_reason) if claim.rejection_reason else "UNKNOWN"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            logger.warning(f"⚠️ Rejection reasons breakdown: {rejection_reasons}")
        
        step_timings["step_7_filter_verified"] = time.perf_counter() - step_start
        logger.info(f"Step 7 time: {step_timings['step_7_filter_verified']:.2f}s")

        processing_time = time.perf_counter() - start_time
        logger.info(f"Total verifiable pipeline time: {processing_time:.2f}s")
        
        # Print session diagnostics summary if enabled
        if diagnostics:
            mode = "RELAXED" if config.RELAXED_VERIFICATION_MODE else "STANDARD"
            summary = diagnostics.get_session_summary(mode=mode)
            diagnostics.print_session_summary(summary)
            
            # Save JSON debug report
            if config.SAVE_DEBUG_REPORT:
                diagnostics.save_debug_report(config.OUTPUT_DIR)
        
        # SAFE FALLBACK: Auto-relaxed retry if mass rejection detected
        auto_relaxed_retry = False
        total_claims = len(claim_collection.claims)
        rejected_count = sum(1 for c in claim_collection.claims if c.status == VerificationStatus.REJECTED)
        
        if total_claims > 0:
            rejection_rate = rejected_count / total_claims
            
            # If ≥95% rejected and NOT already in relaxed mode, retry with relaxed thresholds
            if rejection_rate >= 0.95 and not config.RELAXED_VERIFICATION_MODE:
                logger.error(
                    f"⚠️  MASS REJECTION DETECTED: {rejection_rate*100:.1f}% rejected ({rejected_count}/{total_claims})"
                )
                logger.error("Auto-retrying verification with RELAXED thresholds...")
                
                # Temporarily enable relaxed mode
                original_mode = config.RELAXED_VERIFICATION_MODE
                config.RELAXED_VERIFICATION_MODE = True
                
                try:
                    # Re-apply evidence policy to existing claims (no re-generation needed)
                    logger.info("Re-evaluating claims with relaxed thresholds...")
                    for claim in claim_collection.claims:
                        if claim.evidence_objects:
                            decision = evaluate_claim(
                                claim=claim,
                                evidence_items=claim.evidence_objects,
                                domain_profile=self.domain_profile,
                                nli_results=claim.metadata.get("nli_results") if hasattr(claim, "metadata") else None
                            )
                            claim.status = decision.status
                            claim.confidence = decision.confidence
                            claim.rejection_reason = decision.rejection_reason
                    
                    # Re-filter to verified claims
                    verified_collection = self.claim_validator.filter_collection(
                        claim_collection,
                        include_verified=True,
                        include_low_confidence=False,
                        include_rejected=False
                    )
                    
                    # Check if retry helped
                    new_rejected_count = sum(1 for c in claim_collection.claims if c.status == VerificationStatus.REJECTED)
                    new_rejection_rate = new_rejected_count / total_claims if total_claims > 0 else 0
                    
                    logger.info(
                        f"After relaxed retry: {new_rejection_rate*100:.1f}% rejected ({new_rejected_count}/{total_claims})"
                    )
                    auto_relaxed_retry = True
                    
                    # If still ≥95% rejected, log diagnostic hints
                    if new_rejection_rate >= 0.95:
                        logger.error(
                            "⚠️  Still high rejection rate after relaxed retry. Possible issues:\n"
                            "  1. Retrieval failure (no relevant evidence found)\n"
                            "  2. NLI model outputs mostly NEUTRAL\n"
                            "  3. Evidence chunks too small or too large\n"
                            "  4. Source content doesn't support generated claims\n"
                            f"  → Check debug report at: {config.DEBUG_REPORT_PATH}"
                        )
                
                finally:
                    # Restore original mode
                    config.RELAXED_VERIFICATION_MODE = original_mode
        
        # Log rejection histogram summary
        rejection_histogram.log_summary()
        
        # Save JSON debug report if enabled
        if diagnostics and config.DEBUG_VERIFICATION:
            diagnostics.save_debug_report(config.OUTPUT_DIR)
        
        # Serialize graph_metrics to dict for JSON compatibility
        graph_metrics_dict = None
        if graph_metrics:
            if hasattr(graph_metrics, 'model_dump'):
                # Pydantic v2
                graph_metrics_dict = graph_metrics.model_dump()
            elif hasattr(graph_metrics, 'dict'):
                # Pydantic v1
                graph_metrics_dict = graph_metrics.dict()
            elif isinstance(graph_metrics, dict):
                # Already a dict (backward compat)
                graph_metrics_dict = graph_metrics
            else:
                # Fallback: try to convert to dict
                try:
                    import dataclasses
                    if dataclasses.is_dataclass(graph_metrics):
                        graph_metrics_dict = dataclasses.asdict(graph_metrics)
                    else:
                        graph_metrics_dict = {}
                except:
                    graph_metrics_dict = {}
        
        # Serialize claim_graph to dict for JSON compatibility
        claim_graph_dict = None
        if graph:
            if hasattr(graph, 'to_dict'):
                claim_graph_dict = graph.to_dict()
            else:
                claim_graph_dict = None
        
        # Serialize baseline_output to dict if needed
        baseline_output_dict = None
        if baseline_output:
            if hasattr(baseline_output, 'to_dict'):
                baseline_output_dict = baseline_output.to_dict()
            elif hasattr(baseline_output, 'model_dump'):
                baseline_output_dict = baseline_output.model_dump()
            elif hasattr(baseline_output, 'dict'):
                baseline_output_dict = baseline_output.dict()
            elif isinstance(baseline_output, dict):
                baseline_output_dict = baseline_output
            else:
                baseline_output_dict = None
        
        # Assemble verifiable metadata
        # NOTE: claim_collection and verified_collection are kept as objects for UI access
        # but are NOT included in JSON exports (handled separately in export functions)
        verifiable_metadata = {
            "verifiable_mode": True,
            "domain_profile": self.domain_profile.name,
            "domain_profile_display": self.domain_profile.display_name,
            "threat_model": get_threat_model_summary(),
            "dependency_warnings": dependency_warnings,
            "processing_time": processing_time,
            "claim_collection": claim_collection,
            "verified_collection": verified_collection,
            "claim_graph": graph,
            "claim_graph_dict": claim_graph_dict,  # Serialized version for exports
            "graph_metrics": graph_metrics_dict,
            "metrics": metrics,
            "evidence_stats": evidence_stats,
            "baseline_output": baseline_output,
            "baseline_output_dict": baseline_output_dict,  # Serialized version for exports
            "timings": step_timings,
            "total_time_seconds": processing_time,
            "auto_relaxed_retry": auto_relaxed_retry,  # Flag indicating if auto-retry occurred
            "url_ingestion_summary": url_ingestion_summary  # URL ingestion statistics
        }
        
        # Log summary
        logger.info(
            f"Verifiable mode complete: "
            f"{len(verified_collection.claims)}/{len(claim_collection.claims)} verified, "
            f"time={processing_time:.1f}s"
        )
        
        # Return baseline output + verifiable metadata
        return baseline_output, verifiable_metadata
    
    def _parse_combined_content(
        self,
        combined_content: str
    ) -> Tuple[str, str]:
        """
        Parse combined content into transcript and notes.
        
        Args:
            combined_content: Combined preprocessed content
        
        Returns:
            Tuple of (transcript, notes)
        """
        # Simple heuristic: look for section markers
        transcript = ""
        notes = ""
        
        if "TRANSCRIPT:" in combined_content:
            parts = combined_content.split("TRANSCRIPT:", 1)
            if len(parts) == 2:
                notes = parts[0].strip()
                transcript = parts[1].split("NOTES:", 1)[0].strip()
        elif "HANDWRITTEN NOTES:" in combined_content:
            parts = combined_content.split("HANDWRITTEN NOTES:", 1)
            if len(parts) == 2:
                transcript = parts[0].strip()
                notes = parts[1].strip()
        else:
            # Fallback: treat all as notes
            notes = combined_content
        
        return transcript, notes
