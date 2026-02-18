"""
Learning Claims Schema for Research-Grade Verifiable Mode

This module defines structured claim types that enforce evidence-grounded
generation with explicit confidence tracking, rejection reasons, and
full auditability for research reproducibility.

Key Innovation: Learning claims treat AI outputs as testable hypotheses
with explicit evidence backing, enabling systematic hallucination detection.

Reference: See docs/VERIFIABILITY_CONTRACT.md for formal specification
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class ClaimType(str, Enum):
    """Types of learning claims that can be generated."""
    DEFINITION = "definition"
    EQUATION = "equation"
    EXAMPLE = "example"
    MISCONCEPTION = "misconception"
    ALGORITHM_STEP = "algorithm_step"
    COMPLEXITY = "complexity"
    INVARIANT = "invariant"


class VerificationStatus(str, Enum):
    """Verification status of claims."""
    VERIFIED = "verified"  # confidence >= t_verify, sufficient evidence
    LOW_CONFIDENCE = "low_confidence"  # t_reject <= confidence < t_verify
    REJECTED = "rejected"  # confidence < t_reject OR no evidence


class RejectionReason(str, Enum):
    """Coded reasons why a claim was rejected or low-confidence."""
    NO_EVIDENCE = "NO_EVIDENCE"
    """Zero evidence sources found in retrieval"""
    
    LOW_SIMILARITY = "LOW_SIMILARITY"
    """Best evidence similarity < τ threshold"""

    MISSING_BIGO = "MISSING_BIGO"
    """Complexity claim lacks Big-O evidence"""

    MISSING_PSEUDOCODE = "MISSING_PSEUDOCODE"
    """Algorithm-step claim lacks pseudocode-like evidence"""

    DISALLOWED_CLAIM_TYPE = "DISALLOWED_CLAIM_TYPE"
    """Claim type is not allowed for the current domain"""
    
    INSUFFICIENT_SOURCES = "INSUFFICIENT_SOURCES"
    """Independent sources < k requirement"""
    
    LOW_CONSISTENCY = "LOW_CONSISTENCY"
    """Evidence pieces inconsistent (σ_actual < σ_min)"""
    
    CONFLICT = "CONFLICT"
    """Explicit contradiction detected across evidence"""
    
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
    """confidence < t_reject threshold"""
    
    DEPENDENCY_REQUIRED = "DEPENDENCY_REQUIRED"
    """Requires undefined prerequisite concept"""
    
    LOW_CONFIDENCE_GENERIC = "LOW_CONFIDENCE"
    """Generic low confidence (multiple causes possible)"""


class EvidenceItem(BaseModel):
    """
    Individual piece of evidence supporting a claim.
    
    Core research schema: Every evidence item tracks source provenance,
    similarity scores, and reliability priors for auditability.
    
    Attributes:
        evidence_id: Unique identifier for this evidence
        source_id: Source document/material identifier
        source_type: Type of source (notes, transcript, textbook, web, etc.)
        snippet: The actual text span from source
        span_metadata: Location details (page, line, start_char, etc.)
        similarity: Cosine similarity to claim (0-1)
        reliability_prior: Prior belief in source reliability (0-1)
        timestamp: When evidence was extracted
    """
    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evidence identifier"
    )
    source_id: str = Field(
        ...,
        description="Source document identifier (e.g., 'calculus_lecture.txt')"
    )
    source_type: str = Field(
        ...,
        description="Source type: notes, transcript, textbook, web, equation, etc."
    )
    snippet: str = Field(
        ...,
        min_length=15,
        description="The actual text span from the source"
    )
    span_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source location: {'page': 5, 'line': 120, 'start_char': 1000}"
    )
    similarity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity of claim vs. evidence (0-1)"
    )
    reliability_prior: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Prior belief in source reliability (e.g., textbook=0.95, web=0.6)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When evidence was extracted"
    )
    
    @validator('snippet')
    def validate_content_substance(cls, v):
        """Ensure evidence content is substantive."""
        if len(v.strip()) < 15:
            raise ValueError("Evidence snippet must be at least 15 characters")
        return v.strip()


class DependencyRequest(BaseModel):
    """Request for prerequisite concept to be defined."""
    concept: str = Field(..., description="The undefined concept needed")
    reason: str = Field(..., description="Why it's needed")
    claimed_type: Optional[ClaimType] = Field(None, description="Suggested type of required claim")


class LearningClaim(BaseModel):
    """
    A learning claim with evidence grounding and confidence tracking.
    
    This is the core data structure for Verifiable Mode, representing
    a single factual assertion with explicit evidence links and
    verification status.
    
    Key Design: Learning claims treat AI outputs as **testable hypotheses**
    with explicit evidence backing, enabling systematic hallucination detection.
    
    Attributes:
        claim_id: Unique identifier
        claim_type: Type of claim (definition, equation, etc.)
        claim_text: The actual claim statement (empty until evidence-first retrieved)
        evidence_ids: List of evidence IDs supporting this claim
        evidence_objects: Full evidence objects (optional, for convenience)
        confidence: Confidence score (0.0-1.0)
        status: Verification status (verified, low_confidence, rejected)
        rejection_reason: Coded reason for rejection (NO_EVIDENCE, LOW_SIMILARITY, etc.)
        dependency_requests: Prerequisites needed to generate this claim
        metadata: Additional claim metadata
        created_at: Creation timestamp
        validated_at: Validation timestamp
    """
    claim_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique claim identifier"
    )
    claim_type: ClaimType = Field(
        ...,
        description="Type of learning claim"
    )
    claim_text: str = Field(
        default="",
        description="The factual claim statement (empty until evidence retrieved)"
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="List of evidence IDs supporting this claim"
    )
    evidence_objects: List[EvidenceItem] = Field(
        default_factory=list,
        description="Full evidence objects (for convenience)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in claim validity (0.0-1.0), computed from evidence + graph metrics"
    )
    status: VerificationStatus = Field(
        default=VerificationStatus.REJECTED,
        description="Verification status"
    )
    rejection_reason: Optional[RejectionReason] = Field(
        default=None,
        description="Coded reason for rejection/low-confidence (NO_EVIDENCE, CONFLICT, etc.)"
    )
    dependency_requests: List[DependencyRequest] = Field(
        default_factory=list,
        description="Prerequisites needed to generate this claim"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional claim metadata: {'source_claim_id': '...', 'agent': 'ConceptAgent', ...}"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When claim was created"
    )
    validated_at: Optional[datetime] = Field(
        None,
        description="When claim was validated"
    )
    
    @validator('claim_text')
    def validate_claim_text(cls, v):
        """Normalize claim text (may be empty prior to evidence-first retrieval)."""
        return v.strip() if v else ""
    
    @validator('evidence_ids')
    def sync_evidence_ids(cls, v, values):
        """Sync evidence_ids with evidence_objects if present."""
        if 'evidence_objects' in values and values['evidence_objects']:
            return [e.evidence_id for e in values['evidence_objects']]
        return v
    
    def add_evidence(self, evidence: EvidenceItem) -> None:
        """Add evidence to this claim."""
        if evidence.evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence.evidence_id)
            self.evidence_objects.append(evidence)
    
    def has_sufficient_evidence(self, min_evidence: int = 1) -> bool:
        """
        Check if claim has sufficient evidence.
        
        Args:
            min_evidence: Minimum number of evidence items required (default: 1)
        
        Returns:
            True if sufficient evidence exists
        """
        return len(self.evidence_ids) >= min_evidence
    
    def calculate_confidence(self, graph_metrics: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate confidence from evidence and optional graph metrics.
        
        Formula:
          base_confidence = weighted_mean(similarity × reliability_prior)
          with_graph_bonus = base_confidence × (1 + redundancy_bonus + diversity_bonus)
        
        Args:
            graph_metrics: Optional dict with 'redundancy', 'diversity' keys
        
        Returns:
            Calculated confidence score (0.0-1.0)
        """
        if not self.evidence_objects:
            return 0.0
        
        # Weighted mean of (similarity × reliability_prior)
        weighted_scores = [
            e.similarity * e.reliability_prior
            for e in self.evidence_objects
        ]
        base_confidence = sum(weighted_scores) / len(weighted_scores)
        
        # Apply graph bonuses if provided
        if graph_metrics:
            # Handle both GraphMetrics object and dict (backward compatibility)
            if isinstance(graph_metrics, dict):
                redundancy = graph_metrics.get("avg_redundancy", 1.0)
                diversity = graph_metrics.get("avg_diversity", 1.0)
            else:
                redundancy = getattr(graph_metrics, 'avg_redundancy', 1.0)
                diversity = getattr(graph_metrics, 'avg_diversity', 1.0)
            
            # Bonus for multiple independent sources
            redundancy_bonus = min(0.15 * (redundancy - 1), 0.15)  # max +0.15
            diversity_bonus = min(0.10 * (diversity - 1), 0.10)    # max +0.10
            
            base_confidence *= (1 + redundancy_bonus + diversity_bonus)
        
        return min(base_confidence, 1.0)  # Clamp to [0, 1]
    
    def should_reject(self, min_confidence: float = 0.2) -> bool:
        """
        Determine if claim should be rejected.
        
        Args:
            min_confidence: Minimum confidence threshold
        
        Returns:
            True if claim should be rejected
        """
        return self.confidence < min_confidence or not self.has_sufficient_evidence()
    
    def update_status(
        self,
        verified_threshold: float = 0.5,
        rejected_threshold: float = 0.2
    ) -> None:
        """
        Update claim status based on confidence and evidence.
        
        Args:
            verified_threshold: confidence >= this → VERIFIED
            rejected_threshold: confidence < this → REJECTED
        """
        if self.confidence >= verified_threshold and self.has_sufficient_evidence():
            self.status = VerificationStatus.VERIFIED
            self.rejection_reason = None
        elif self.confidence < rejected_threshold:
            self.status = VerificationStatus.REJECTED
            if self.rejection_reason is None:
                self.rejection_reason = RejectionReason.INSUFFICIENT_CONFIDENCE
        else:
            self.status = VerificationStatus.LOW_CONFIDENCE
        
        self.validated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export claim to JSON-serializable dict."""
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type.value,
            "claim_text": self.claim_text,
            "evidence_ids": self.evidence_ids,
            "evidence_count": len(self.evidence_ids),
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "dependency_requests": [
                {"concept": dr.concept, "reason": dr.reason, "type": dr.claimed_type.value if dr.claimed_type else None}
                for dr in self.dependency_requests
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "validated_at": self.validated_at.isoformat() if self.validated_at else None
        }
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = False
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123",
                "claim_type": "definition",
                "claim_text": "A derivative represents the instantaneous rate of change of a function",
                "evidence_ids": ["ev_001", "ev_002"],
                "confidence": 0.85,
                "status": "verified",
                "rejection_reason": None
            }
        }


class GraphMetrics(BaseModel):
    """Metrics computed from claim-evidence dependency graph."""
    avg_support_depth: float = Field(..., description="Average path length from evidence to claim")
    avg_redundancy: float = Field(..., description="Average number of independent sources per claim")
    avg_diversity: float = Field(..., description="Average count of distinct source_types per claim")
    conflict_count: int = Field(..., description="Total contradictions detected")
    total_claims: int = Field(..., description="Total claims in graph")
    total_evidence: int = Field(..., description="Total evidence items in graph")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert GraphMetrics to dict for backward compatibility.
        
        Returns:
            Dictionary with all metric fields
        """
        return {
            "avg_support_depth": self.avg_support_depth,
            "avg_redundancy": self.avg_redundancy,
            "avg_diversity": self.avg_diversity,
            "conflict_count": self.conflict_count,
            "total_claims": self.total_claims,
            "total_evidence": self.total_evidence,
            # Aliases for backward compatibility
            "evidence_nodes": self.total_evidence,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like .get() method for backward compatibility.
        
        Args:
            key: Field name
            default: Default value if field not found
        
        Returns:
            Field value or default
        """
        # Handle aliasing
        if key == "evidence_nodes":
            return self.total_evidence
        
        return getattr(self, key, default)


class ClaimCollection(BaseModel):
    """
    Collection of learning claims with metadata.
    
    Attributes:
        session_id: Session identifier
        claims: List of learning claims
        metadata: Collection metadata
        created_at: Creation timestamp
    """
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    claims: List[LearningClaim] = Field(
        default_factory=list,
        description="Learning claims in this collection"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Collection metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When collection was created"
    )
    
    def add_claim(self, claim: LearningClaim) -> None:
        """Add a claim to the collection."""
        self.claims.append(claim)
    
    def get_by_type(self, claim_type: ClaimType) -> List[LearningClaim]:
        """Get all claims of a specific type."""
        return [c for c in self.claims if c.claim_type == claim_type]
    
    def get_by_status(self, status: VerificationStatus) -> List[LearningClaim]:
        """Get all claims with a specific status."""
        return [c for c in self.claims if c.status == status]
    
    def get_verified_claims(self) -> List[LearningClaim]:
        """Get all verified claims."""
        return self.get_by_status(VerificationStatus.VERIFIED)
    
    def get_rejected_claims(self) -> List[LearningClaim]:
        """Get all rejected claims."""
        return self.get_by_status(VerificationStatus.REJECTED)
    
    def get_low_confidence_claims(self) -> List[LearningClaim]:
        """Get all low-confidence claims."""
        return self.get_by_status(VerificationStatus.LOW_CONFIDENCE)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate collection statistics."""
        if not self.claims:
            return {
                "total_claims": 0,
                "verified_count": 0,
                "rejected_count": 0,
                "low_confidence_count": 0,
                "avg_confidence": 0.0,
                "rejection_rate": 0.0,
                "verification_rate": 0.0
            }
        
        verified = len(self.get_verified_claims())
        rejected = len(self.get_rejected_claims())
        low_conf = len(self.get_low_confidence_claims())
        
        return {
            "total_claims": len(self.claims),
            "verified_count": verified,
            "rejected_count": rejected,
            "low_confidence_count": low_conf,
            "avg_confidence": round(sum(c.confidence for c in self.claims) / len(self.claims), 4),
            "rejection_rate": round(rejected / len(self.claims), 4) if self.claims else 0.0,
            "verification_rate": round(verified / len(self.claims), 4) if self.claims else 0.0
        }
    
    def get_rejection_breakdown(self) -> Dict[str, int]:
        """Count rejections by reason."""
        breakdown = {}
        for claim in self.get_rejected_claims():
            if claim.rejection_reason:
                reason_str = claim.rejection_reason.value
                breakdown[reason_str] = breakdown.get(reason_str, 0) + 1
        return breakdown
    
    def to_dict(self) -> Dict[str, Any]:
        """Export collection to JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "claims": [claim.to_dict() for claim in self.claims],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "statistics": self.calculate_statistics()
        }
