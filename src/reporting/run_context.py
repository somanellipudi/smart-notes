"""
RunContext: Unified pipeline context object for carrying data through the verification pipeline.

This class ensures consistency and completeness of data as it flows from ingestion
through extraction, verification, and reporting stages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class InputsReceived:
    """Record of input sources provided."""
    notes_text: bool = False
    audio_file: bool = False
    urls: bool = False
    pdf_files: bool = False
    external_context: bool = False
    
    def to_list(self) -> List[str]:
        """Convert to list of input types."""
        inputs = []
        if self.pdf_files:
            inputs.append("pdf")
        if self.audio_file:
            inputs.append("audio")
        if self.notes_text:
            inputs.append("text")
        if self.urls:
            inputs.append("urls")
        if self.external_context:
            inputs.append("external_context")
        return inputs


@dataclass
class IngestionReportContext:
    """Detailed ingestion diagnostics."""
    # PDF-specific metrics
    total_pages: int = 0
    pages_ocr: int = 0
    headers_removed: int = 0
    footers_removed: int = 0
    watermarks_removed: int = 0
    
    # URL-specific metrics
    url_count: int = 0
    url_fetch_success_count: int = 0
    url_chunks_total: int = 0
    
    # Text-specific metrics
    text_chars_total: int = 0
    text_chunks_total: int = 0
    
    # Audio-specific metrics
    audio_seconds: float = 0.0
    transcript_chars: int = 0
    transcript_chunks_total: int = 0
    
    # Overall metrics
    chunks_total_all_sources: int = 0
    avg_chunk_size_all_sources: Optional[float] = None  # None if no chunks
    extraction_methods: List[str] = field(default_factory=list)
    sources_processed: Dict[str, int] = field(default_factory=dict)  # {'pdf': 3, 'text': 1, 'url': 5, 'audio': 1}
    total_text_length: int = 0  # Total characters extracted
    
    def validate_invariants(self) -> List[str]:
        """Return list of invariant violations, empty if all valid."""
        violations = []
        
        # Invariant: if chunks_total_all_sources > 0, avg_chunk_size_all_sources must be set
        if self.chunks_total_all_sources > 0 and self.avg_chunk_size_all_sources is None:
            violations.append(
                f"avg_chunk_size_all_sources must be set when chunks_total_all_sources > 0 ({self.chunks_total_all_sources} chunks)"
            )
        
        # Invariant: if chunks_total_all_sources == 0, avg_chunk_size_all_sources should be None
        if self.chunks_total_all_sources == 0 and self.avg_chunk_size_all_sources is not None:
            violations.append(
                f"avg_chunk_size_all_sources should be None when chunks_total_all_sources == 0 (was {self.avg_chunk_size_all_sources})"
            )
        
        # Invariant: OCR pages <= total pages
        if self.pages_ocr > self.total_pages:
            violations.append(
                f"pages_ocr ({self.pages_ocr}) cannot exceed total_pages ({self.total_pages})"
            )
        
        # Invariant: url_fetch_success_count <= url_count
        if self.url_fetch_success_count > self.url_count:
            violations.append(
                f"url_fetch_success_count ({self.url_fetch_success_count}) cannot exceed url_count ({self.url_count})"
            )
        
        # Invariant: chunks_total_all_sources should equal sum of individual source chunks
        computed_total = self.url_chunks_total + self.text_chunks_total + self.transcript_chunks_total
        # Note: PDF chunks tracked via sources_processed if available
        if self.chunks_total_all_sources > 0 and computed_total > self.chunks_total_all_sources:
            violations.append(
                f"Computed chunk total ({computed_total}) exceeds chunks_total_all_sources ({self.chunks_total_all_sources})"
            )
        
        return violations


@dataclass
class ExtractionReportContext:
    """Extraction stage diagnostics."""
    chunks_total: int = 0
    chunk_sources: Dict[str, int] = field(default_factory=dict)  # {'notes': 5, 'pdf': 10, 'urls': 3}
    avg_chunk_size: Optional[float] = None
    extraction_methods_used: List[str] = field(default_factory=list)
    
    def _compute_avg_chunk_size(self, total_text_length: int) -> None:
        """Compute average chunk size from known data."""
        if self.chunks_total > 0:
            self.avg_chunk_size = total_text_length / self.chunks_total if total_text_length > 0 else 0
        else:
            self.avg_chunk_size = None


@dataclass
class ClaimReportContext:
    """Claim processing diagnostics."""
    total_claims: int = 0
    question_claims: int = 0
    factual_claims: int = 0  # Non-question claims
    
    def validate_invariants(self) -> List[str]:
        """Return list of invariant violations."""
        violations = []
        
        # Invariant: question + factual <= total
        if self.question_claims + self.factual_claims > self.total_claims:
            violations.append(
                f"question_claims ({self.question_claims}) + factual ({self.factual_claims}) "
                f"cannot exceed total ({self.total_claims})"
            )
        
        return violations


@dataclass
class VerificationReportContext:
    """Verification summary with detailed breakdowns."""
    total_claims: int = 0
    verified: int = 0
    rejected: int = 0
    low_confidence: int = 0
    avg_confidence: float = 0.0
    
    # Rejection reason counts (must sum to rejected)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    
    # Claim type breakdown
    question_claims_count: int = 0
    factual_verified: int = 0  # Verified factual claims
    factual_rejected: int = 0  # Rejected factual claims
    
    def validate_invariants(self) -> List[str]:
        """Return list of invariant violations."""
        violations = []
        
        # Invariant: verified + rejected + low_confidence == total_claims
        count_sum = self.verified + self.rejected + self.low_confidence + self.question_claims_count
        if self.total_claims > 0 and count_sum != self.total_claims:
            violations.append(
                f"Claim count invariant violated: verified ({self.verified}) + rejected ({self.rejected}) + "
                f"low_confidence ({self.low_confidence}) + questions ({self.question_claims_count}) "
                f"= {count_sum}, expected {self.total_claims}"
            )
        
        # Invariant: rejection_reasons must sum to rejected count
        if self.rejected > 0:
            reasons_sum = sum(self.rejection_reasons.values())
            if reasons_sum != self.rejected:
                violations.append(
                    f"Rejection reason invariant violated: reasons sum to {reasons_sum}, "
                    f"but rejected count is {self.rejected}. Missing: {self.rejected - reasons_sum}"
                )
        
        # Invariant: if total_claims > 0, average confidence should be set
        if self.total_claims > 0 and self.avg_confidence == 0.0:
            violations.append(f"avg_confidence should be computed when total_claims > 0")
        
        return violations


@dataclass
class EvidenceSampleContext:
    """Sample of evidence for report."""
    total_evidence_items: int = 0
    evidence_by_source: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    def add_evidence(self, source_id: str, evidence_dict: Dict[str, Any]) -> None:
        """Add evidence sample."""
        if source_id not in self.evidence_by_source:
            self.evidence_by_source[source_id] = []
        self.evidence_by_source[source_id].append(evidence_dict)


@dataclass
class RunContext:
    """
    Unified pipeline context carrying all data through the verification pipeline.
    
    Records what was input, what was extracted, how verification proceeded,
    and maintains invariants to catch data inconsistencies.
    """
    # Identifiers
    session_id: str
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Configuration
    domain_profile: str = "general"
    llm_model: str = ""
    embedding_model: str = ""
    nli_model: str = ""
    
    # Context objects
    inputs_received: InputsReceived = field(default_factory=InputsReceived)
    ingestion_report: IngestionReportContext = field(default_factory=IngestionReportContext)
    extraction_report: ExtractionReportContext = field(default_factory=ExtractionReportContext)
    claim_report: ClaimReportContext = field(default_factory=ClaimReportContext)
    verification_summary: VerificationReportContext = field(default_factory=VerificationReportContext)
    evidence_samples: EvidenceSampleContext = field(default_factory=EvidenceSampleContext)
    
    # Performance metrics
    timings: Dict[str, float] = field(default_factory=dict)
    
    def validate_all_invariants(self) -> List[str]:
        """Validate all invariants across all contexts."""
        violations = []
        
        # Ingestion invariants
        violations.extend(self.ingestion_report.validate_invariants())
        
        # Claim invariants
        violations.extend(self.claim_report.validate_invariants())
        
        # Verification invariants
        violations.extend(self.verification_summary.validate_invariants())
        
        # Cross-context invariant: if claims exist, should have extracted chunks
        if self.verification_summary.total_claims > 0 and self.extraction_report.chunks_total == 0:
            violations.append(
                f"Cannot have {self.verification_summary.total_claims} claims with 0 chunks extracted"
            )
        
        return violations
    
    def get_invariant_violations_str(self) -> Optional[str]:
        """Get human-readable string of all violations, or None if valid."""
        violations = self.validate_all_invariants()
        if not violations:
            return None
        return "\n".join(f"⚠️ {v}" for v in violations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "domain_profile": self.domain_profile,
            "inputs": {
                "note_text": self.inputs_received.notes_text,
                "audio": self.inputs_received.audio_file,
                "urls": self.inputs_received.urls,
                "pdfs": self.inputs_received.pdf_files,
                "external_context": self.inputs_received.external_context,
            },
            "ingestion": {
                "total_pages": self.ingestion_report.total_pages,
                "pages_ocr": self.ingestion_report.pages_ocr,
                "total_chunks": self.ingestion_report.total_chunks,
                "avg_chunk_size": self.ingestion_report.avg_chunk_size,
            },
            "extraction": {
                "chunks_total": self.extraction_report.chunks_total,
                "extraction_methods": self.extraction_report.extraction_methods_used,
            },
            "claims": {
                "total": self.claim_report.total_claims,
                "questions": self.claim_report.question_claims,
                "factual": self.claim_report.factual_claims,
            },
            "verification": {
                "total_claims": self.verification_summary.total_claims,
                "verified": self.verification_summary.verified,
                "rejected": self.verification_summary.rejected,
                "low_confidence": self.verification_summary.low_confidence,
                "avg_confidence": self.verification_summary.avg_confidence,
                "rejection_reasons": self.verification_summary.rejection_reasons,
            },
            "timings": self.timings,
            "invariants": {
                "violations": self.validate_all_invariants(),
            }
        }
