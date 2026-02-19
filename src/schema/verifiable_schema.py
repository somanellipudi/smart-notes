"""
Verifiable Mode schema extensions for evidence-based, citation-backed generation.

This module extends the base output schema with claim-based structures that
enforce evidence grounding and source attribution for research-oriented use cases.
"""

from typing import List, Optional, Dict, Any, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator

from src.schema.output_schema import (
    ClassSessionOutput,
    Topic,
    Concept,
    WorkedExample,
    EquationExplanation,
    FAQ,
    Misconception,
    RealWorldConnection
)


class EvidenceCitation(BaseModel):
    """
    Citation linking a claim to its source evidence.
    
    Attributes:
        source_type: Type of source
        quote: Exact quote from source
        location: Where in source (line, timestamp, page)
        confidence: Confidence in citation accuracy (0-1)
        origin: Filename or URL of source
        page_num: Page number for PDF sources
        timestamp_range: (start_sec, end_sec) for audio/video
        span_id: Evidence span identifier for traceability
    """
    source_type: Literal[
        "pdf_page",
        "notes_text",
        "external_context",
        "url_article",
        "youtube_transcript",
        "audio_transcript",
        "equation",
        "inferred"
    ] = Field(
        ...,
        description="Type of source material"
    )
    quote: str = Field(
        ...,
        description="Exact quote or reference from source",
        min_length=5
    )
    location: Optional[str] = Field(
        None,
        description="Location identifier (e.g., 'line 42', 'timestamp 1:23', 'page 5')"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence that citation accurately supports claim"
    )
    # Provenance fields
    origin: Optional[str] = Field(
        None,
        description="Filename or URL of source (e.g., 'lecture.pdf', 'https://youtube.com/...')"
    )
    page_num: Optional[int] = Field(
        None,
        description="Page number for PDF sources"
    )
    timestamp_range: Optional[Tuple[float, float]] = Field(
        None,
        description="(start_sec, end_sec) for audio/video sources"
    )
    span_id: Optional[str] = Field(
        None,
        description="Evidence span identifier for traceability"
    )
    
    @validator('quote')
    def validate_quote_substance(cls, v):
        """Ensure quote is substantive."""
        if len(v.strip()) < 5:
            raise ValueError("Citation quote must be at least 5 characters")
        return v.strip()


class VerifiableClaim(BaseModel):
    """
    A verifiable claim with evidence citations.
    
    Attributes:
        claim: The factual claim or assertion
        evidence: Citations supporting this claim
        confidence: Overall confidence in claim validity (0-1)
        claim_type: Type of claim (definition, fact, procedure, etc.)
    """
    claim: str = Field(
        ...,
        description="The factual claim or statement",
        min_length=10
    )
    evidence: List[EvidenceCitation] = Field(
        ...,
        min_items=1,
        description="Evidence citations supporting this claim"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in claim validity"
    )
    claim_type: Literal[
        "definition",
        "fact",
        "procedure",
        "relationship",
        "application",
        "example"
    ] = Field(
        default="fact",
        description="Type of claim being made"
    )
    
    @validator('evidence')
    def validate_evidence_present(cls, v):
        """Ensure at least one evidence citation exists."""
        if not v or len(v) == 0:
            raise ValueError("VerifiableClaim must have at least one evidence citation")
        return v


class VerifiableTopic(Topic):
    """
    Topic with verifiable claims and citations.
    
    Extends base Topic with evidence grounding.
    """
    claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Verifiable claims made about this topic"
    )
    evidence_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall quality of evidence for this topic"
    )


class VerifiableConcept(Concept):
    """
    Concept with verifiable definition and evidence.
    
    Extends base Concept with claim-based structure.
    """
    definition_claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Claims constituting the definition"
    )
    evidence_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Quality of evidence for this concept"
    )


class VerifiableWorkedExample(WorkedExample):
    """
    Worked example with step-by-step verification.
    
    Extends base WorkedExample with evidence for each step.
    """
    solution_claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Verifiable claims for each solution step"
    )
    source_evidence: List[EvidenceCitation] = Field(
        default_factory=list,
        description="Citations to source material for this example"
    )


class VerifiableEquationExplanation(EquationExplanation):
    """
    Equation explanation with cited derivation.
    
    Extends base EquationExplanation with source citations.
    """
    explanation_claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Claims about the equation's meaning"
    )
    derivation_evidence: List[EvidenceCitation] = Field(
        default_factory=list,
        description="Citations for where equation was introduced/derived"
    )


class VerifiableMisconception(Misconception):
    """
    Misconception with evidence of the error and correction.
    
    Extends base Misconception with citations.
    """
    error_evidence: List[EvidenceCitation] = Field(
        default_factory=list,
        description="Evidence that this misconception exists"
    )
    correction_evidence: List[EvidenceCitation] = Field(
        default_factory=list,
        description="Evidence for the correct understanding"
    )


class VerifiableFAQ(FAQ):
    """
    FAQ with evidence-backed answer.
    
    Extends base FAQ with answer citations.
    """
    answer_claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Verifiable claims in the answer"
    )


class VerifiableRealWorldConnection(RealWorldConnection):
    """
    Real-world connection with evidence.
    
    Extends base RealWorldConnection with citations.
    """
    connection_evidence: List[EvidenceCitation] = Field(
        default_factory=list,
        description="Evidence supporting this connection"
    )


class VerificationMetadata(BaseModel):
    """
    Metadata about the verification process.
    
    Attributes:
        mode: Verification mode used
        total_claims: Total number of claims generated
        total_citations: Total number of citations
        avg_citations_per_claim: Average citations per claim
        evidence_coverage: Fraction of content with citations (0-1)
        verification_duration: Time spent on verification
        quality_flags: Any quality concerns raised
    """
    mode: str = Field(default="verifiable", description="Verification mode")
    total_claims: int = Field(default=0, ge=0, description="Total claims")
    total_citations: int = Field(default=0, ge=0, description="Total citations")
    avg_citations_per_claim: float = Field(
        default=0.0,
        ge=0.0,
        description="Average citations per claim"
    )
    evidence_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of output with evidence"
    )
    verification_duration: float = Field(
        default=0.0,
        ge=0.0,
        description="Verification processing time (seconds)"
    )
    quality_flags: List[str] = Field(
        default_factory=list,
        description="Quality concerns or warnings"
    )
    citation_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of citations by source type"
    )


class VerifiableClassSessionOutput(ClassSessionOutput):
    """
    Verifiable output extending base ClassSessionOutput.
    
    This schema enforces evidence-grounded, claim-based generation while
    maintaining compatibility with the standard output format.
    
    All content is decomposed into verifiable claims with citations to
    source material, enabling fact-checking and transparency.
    """
    # Override with verifiable versions
    topics: List[VerifiableTopic] = Field(
        default_factory=list,
        description="Topics with verifiable claims"
    )
    key_concepts: List[VerifiableConcept] = Field(
        default_factory=list,
        description="Concepts with evidence-based definitions"
    )
    equation_explanations: List[VerifiableEquationExplanation] = Field(
        default_factory=list,
        description="Equation explanations with derivation citations"
    )
    worked_examples: List[VerifiableWorkedExample] = Field(
        default_factory=list,
        description="Worked examples with step-by-step evidence"
    )
    common_mistakes: List[VerifiableMisconception] = Field(
        default_factory=list,
        description="Misconceptions with error and correction evidence"
    )
    faqs: List[VerifiableFAQ] = Field(
        default_factory=list,
        description="FAQs with evidence-backed answers"
    )
    real_world_connections: List[VerifiableRealWorldConnection] = Field(
        default_factory=list,
        description="Real-world connections with supporting evidence"
    )
    
    # New verifiable-specific fields
    verification_metadata: VerificationMetadata = Field(
        default_factory=VerificationMetadata,
        description="Metadata about verification process"
    )
    summary_claims: List[VerifiableClaim] = Field(
        default_factory=list,
        description="Verifiable claims extracted from class summary"
    )
    
    def calculate_verification_stats(self) -> None:
        """Calculate and update verification metadata statistics."""
        all_claims = []
        all_citations = []
        citation_types = {}
        
        # Collect from summary
        all_claims.extend(self.summary_claims)
        for claim in self.summary_claims:
            all_citations.extend(claim.evidence)
        
        # Collect from topics
        for topic in self.topics:
            all_claims.extend(topic.claims)
            for claim in topic.claims:
                all_citations.extend(claim.evidence)
        
        # Collect from concepts
        for concept in self.key_concepts:
            all_claims.extend(concept.definition_claims)
            for claim in concept.definition_claims:
                all_citations.extend(claim.evidence)
        
        # Collect from equations
        for eq in self.equation_explanations:
            all_claims.extend(eq.explanation_claims)
            all_citations.extend(eq.derivation_evidence)
        
        # Collect from examples
        for ex in self.worked_examples:
            all_claims.extend(ex.solution_claims)
            all_citations.extend(ex.source_evidence)
        
        # Collect from misconceptions
        for misc in self.common_mistakes:
            all_citations.extend(misc.error_evidence)
            all_citations.extend(misc.correction_evidence)
        
        # Collect from FAQs
        for faq in self.faqs:
            all_claims.extend(faq.answer_claims)
            for claim in faq.answer_claims:
                all_citations.extend(claim.evidence)
        
        # Collect from connections
        for conn in self.real_world_connections:
            all_citations.extend(conn.connection_evidence)
        
        # Count citation types
        for citation in all_citations:
            citation_types[citation.source_type] = citation_types.get(citation.source_type, 0) + 1
        
        # Update metadata
        self.verification_metadata.total_claims = len(all_claims)
        self.verification_metadata.total_citations = len(all_citations)
        self.verification_metadata.avg_citations_per_claim = (
            len(all_citations) / len(all_claims) if all_claims else 0.0
        )
        self.verification_metadata.citation_distribution = citation_types
        
        # Calculate evidence coverage (rough heuristic)
        total_fields = (
            len(self.topics) + len(self.key_concepts) + 
            len(self.equation_explanations) + len(self.worked_examples) +
            len(self.common_mistakes) + len(self.faqs) + 
            len(self.real_world_connections)
        )
        self.verification_metadata.evidence_coverage = (
            len(all_citations) / max(total_fields, 1)
        ) if total_fields > 0 else 0.0
        
        # Quality flags
        if self.verification_metadata.avg_citations_per_claim < 1.0:
            self.verification_metadata.quality_flags.append(
                "Low citation density (< 1 per claim)"
            )
        
        if len(all_citations) == 0:
            self.verification_metadata.quality_flags.append(
                "No citations found"
            )
        
        inferred_count = citation_types.get("inferred", 0)
        if inferred_count > len(all_citations) * 0.5:
            self.verification_metadata.quality_flags.append(
                f"High proportion of inferred citations ({inferred_count}/{len(all_citations)})"
            )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "calc101_2026-01-31_verifiable",
                "class_summary": "Derivatives introduced as rate of change.",
                "summary_claims": [
                    {
                        "claim": "A derivative represents the rate of change of a function",
                        "evidence": [
                            {
                                "source_type": "transcript",
                                "quote": "The derivative tells us how fast something is changing",
                                "location": "timestamp 2:15"
                            }
                        ],
                        "claim_type": "definition"
                    }
                ],
                "verification_metadata": {
                    "total_claims": 42,
                    "total_citations": 67,
                    "avg_citations_per_claim": 1.6,
                    "evidence_coverage": 0.85
                }
            }
        }
