"""
Verifiable reasoning pipeline integration.

This module wraps the standard reasoning pipeline with verifiable mode logic,
converting outputs to claims, retrieving evidence, validating, and filtering.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import config
from src.reasoning.pipeline import ReasoningPipeline
from src.schema.output_schema import ClassSessionOutput
from src.claims.schema import ClaimCollection, ClaimType, VerificationStatus
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
from src.verification.dependency_checker import check_dependencies, apply_dependency_enforcement
from src.policies.threat_model import get_threat_model_summary
from src.verification.diagnostics import VerificationDiagnostics

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
        # Standard pipeline
        self.standard_pipeline = ReasoningPipeline(
            model=model,
            temperature=temperature,
            provider_type=provider_type,
            api_key=api_key,
            ollama_url=ollama_url
        )
        
        # Domain profile
        self.domain_profile = config.get_domain_profile(domain_profile)
        logger.info(f"Using domain profile: {self.domain_profile.display_name}")
        
        # Verifiable components
        self.claim_extractor = ClaimExtractor()
        self.evidence_retriever = ClaimRAG()
        self.claim_validator = ClaimValidator(
            verified_threshold=config.VERIFIABLE_VERIFIED_THRESHOLD,
            rejected_threshold=config.VERIFIABLE_REJECTED_THRESHOLD,
            min_evidence_count=config.VERIFIABLE_MIN_EVIDENCE,
            strict_mode=config.VERIFIABLE_STRICT_MODE
        )
        self.metrics_calculator = VerifiabilityMetrics()
        
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
    
    def process(
        self,
        combined_content: str,
        equations: List[str],
        external_context: str = "",
        session_id: str = None,
        verifiable_mode: bool = False,
        output_filters: Dict[str, bool] = None
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
        
        Returns:
            Tuple of (output, verifiable_metadata)
            - output: ClassSessionOutput (standard format)
            - verifiable_metadata: Additional verifiable mode data (if enabled)
        """
        logger.info(f"Processing with verifiable_mode={verifiable_mode}, filters={output_filters}")
        
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
                output_filters=output_filters
            )
    
    def _process_verifiable(
        self,
        combined_content: str,
        equations: List[str],
        external_context: str,
        session_id: str,
        output_filters: Dict[str, bool] = None
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
        
        # Initialize diagnostics if enabled
        diagnostics = VerificationDiagnostics(session_id) if config.DEBUG_VERIFICATION else None
        if config.RELAXED_VERIFICATION_MODE:
            logger.warning("="*70)
            logger.warning("Running in RELAXED_VERIFICATION_MODE")
            logger.warning(f"  MIN_ENTAILMENT_PROB: {config.RELAXED_MIN_ENTAILMENT_PROB}")
            logger.warning(f"  MIN_SUPPORTING_SOURCES: {config.RELAXED_MIN_SUPPORTING_SOURCES}")
            logger.warning(f"  MAX_CONTRADICTION_PROB: {config.RELAXED_MAX_CONTRADICTION_PROB}")
            logger.warning("="*70)
        
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
        step_start = time.perf_counter()
        claim_collection = self.claim_extractor.extract_from_session(baseline_output)
        step_timings["step_2_extract_claims"] = time.perf_counter() - step_start
        logger.info(f"Step 2 time: {step_timings['step_2_extract_claims']:.2f}s")
        
        # Step 2.5: Enforce claim granularity (split compound claims)
        logger.info("Step 2.5: Enforcing claim granularity policy")
        step_start = time.perf_counter()
        atomic_claims = enforce_granularity(
            claim_collection.claims,
            max_propositions=config.MAX_PROPOSITIONS_PER_CLAIM
        )
        claim_collection.claims = atomic_claims
        step_timings["step_2_5_enforce_granularity"] = time.perf_counter() - step_start
        logger.info(f"Step 2.5 time: {step_timings['step_2_5_enforce_granularity']:.2f}s")
        
        # Step 3: Retrieve evidence for each claim
        logger.info("Step 3: Retrieving evidence for claims")
        step_start = time.perf_counter()
        
        # Convert equations list to content string
        equations_content = "\n".join(equations) if equations else ""
        
        # Parse combined content to separate transcript and notes
        transcript, notes = self._parse_combined_content(combined_content)

        # Preprocess content to reduce noise and avoid long-running retrieval
        cleaned_transcript = self.preprocessor.filter_noise(transcript) if transcript else ""
        cleaned_notes = self.preprocessor.filter_noise(notes) if notes else ""
        cleaned_external = self.preprocessor.filter_noise(external_context) if external_context else ""

        # Data mining: extract key entities for metadata
        mined_entities = self.preprocessor.extract_key_entities(
            " ".join([cleaned_transcript, cleaned_notes, cleaned_external]).strip()
        )
        claim_collection.metadata["mined_entities"] = mined_entities

        # Segment content and rank segments by evidence quality
        segments = []
        for source_name, text in [
            ("transcript", cleaned_transcript),
            ("notes", cleaned_notes),
            ("external", cleaned_external)
        ]:
            if not text:
                continue
            for seg in self.preprocessor.segment_content(text, max_segment_length=300):
                score = self.preprocessor.rank_evidence_quality(seg["text"])
                segments.append({
                    "text": seg["text"],
                    "score": score,
                    "source": source_name
                })

        # Keep only top segments to prevent runaway processing
        segments.sort(key=lambda s: (s["score"], len(s["text"])), reverse=True)
        top_segments = segments[:12]

        # Combine all sources together
        sources = [s["text"] for s in top_segments]
        if equations_content and equations_content.strip():
            sources.append(equations_content)
        sources = list(dict.fromkeys([s for s in sources if s]))  # Filter empty + dedupe
        
        # Log chunking diagnostics
        if diagnostics and config.DEBUG_CHUNKING:
            total_length = sum(len(s) for s in sources)
            avg_chunk_size = total_length / len(sources) if sources else 0
            diagnostics.log_chunking_validation(
                total_source_length=total_length,
                num_chunks=len(sources),
                avg_chunk_size=avg_chunk_size
            )

        # Retrieve evidence for each claim
        all_nli_results = []
        for claim in claim_collection.claims:
            evidence = self.evidence_retriever.retrieve_evidence_for_claim(claim, sources)
            claim.evidence_objects.extend(evidence)
            
            # Apply decision policy to this claim
            status, reason, confidence = self.evidence_retriever.apply_decision_policy(claim, evidence)
            claim.status = status
            claim.confidence = confidence
            
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
                    status=status,
                    reason=reason
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
        
        # Log NLI distribution if enabled
        if diagnostics and config.DEBUG_NLI_DISTRIBUTION and all_nli_results:
            diagnostics.log_nli_distribution(all_nli_results)
        
        step_timings["step_3_retrieve_evidence"] = time.perf_counter() - step_start
        logger.info(f"Step 3 time: {step_timings['step_3_retrieve_evidence']:.2f}s")

        # Step 3.5: Evidence-first claim text generation
        logger.info("Step 3.5: Generating claim text only after evidence")
        step_start = time.perf_counter()
        for claim in claim_collection.claims:
            if claim.status == VerificationStatus.REJECTED:
                continue
            if not claim.evidence_objects:
                claim.status = VerificationStatus.REJECTED
                claim.confidence = 0.0
                continue

            if claim.claim_type == ClaimType.DEFINITION:
                try:
                    # ConceptAgent.generate() requires claim and evidence list
                    generated_text = self.concept_agent.generate(claim, claim.evidence_objects)
                    claim.claim_text = generated_text
                    # Confidence already set by decision policy
                except AgentRefusalError as e:
                    logger.info(f"Agent refused to generate: {e}")
                    claim.status = VerificationStatus.REJECTED
                    claim.confidence = 0.0
            else:
                # For non-definition claims, use draft if available
                draft_text = claim.metadata.get("draft_text", "")
                if draft_text:
                    claim.claim_text = draft_text
                else:
                    claim.status = VerificationStatus.REJECTED
                    claim.confidence = 0.0
            step_timings["step_3_5_generate_text"] = time.perf_counter() - step_start
            logger.info(f"Step 3.5 time: {step_timings['step_3_5_generate_text']:.2f}s")
        
        # Step 4: Validate claims
        logger.info("Step 4: Validating claims")
        step_start = time.perf_counter()
        claim_collection = self.claim_validator.validate_collection(claim_collection)
        step_timings["step_4_validate_claims"] = time.perf_counter() - step_start
        logger.info(f"Step 4 time: {step_timings['step_4_validate_claims']:.2f}s")
        
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
            if config.DEBUG_VERIFICATION:
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
            "baseline_output": baseline_output,
            "baseline_output_dict": baseline_output_dict,  # Serialized version for exports
            "timings": step_timings,
            "total_time_seconds": processing_time
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
