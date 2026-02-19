"""
Research Report Generator for Smart Notes Sessions

Builds comprehensive reports in multiple formats (MD/HTML/JSON) including:
- Session metadata (versions, seed, inputs)
- Ingestion statistics (pages, OCR, headers removed)
- Verification statistics (verified/rejected claims, top rejection reasons)
- Trust guidance (what to trust, what not to trust)
- Claim table with citations (page numbers, snippets)
- Audit trail (JSON format for reproducibility)

See docs/RESEARCH_REPORT.md for format specifications.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import config
    _DEBUG_JSON_ASSERT = bool(getattr(config, "DEBUG_VERIFICATION", False))
except Exception:
    _DEBUG_JSON_ASSERT = False

logger = logging.getLogger(__name__)


def _assert_json_module() -> None:
    """Ensure json refers to the json module (debug safeguard)."""
    assert hasattr(json, "dumps") and callable(json.dumps)


@dataclass
class SessionMetadata:
    """Session metadata for reports."""
    session_id: str
    timestamp: str
    version: str
    seed: int
    language_model: str
    embedding_model: str
    nli_model: str
    inputs_used: List[str]


@dataclass
class IngestionReport:
    """Ingestion statistics."""
    total_pages: int
    pages_ocr: int
    headers_removed: int
    footers_removed: int
    watermarks_removed: int
    total_chunks: int
    avg_chunk_size: Optional[Union[int, float]]  # None if total_chunks == 0
    extraction_methods: List[str]
    
    def __post_init__(self):
        """Enforce invariant: avg_chunk_size should be None if no chunks."""
        if self.total_chunks == 0 and self.avg_chunk_size is not None:
            logger.warning(
                f"Invariant: avg_chunk_size must be None when total_chunks=0. "
                f"Setting to None (was {self.avg_chunk_size})"
            )
            self.avg_chunk_size = None


@dataclass
class VerificationSummary:
    """Verification statistics summary."""
    total_claims: int
    verified_count: int
    low_confidence_count: int
    rejected_count: int
    avg_confidence: float
    top_rejection_reasons: List[Tuple[str, int]]  # (reason, count) pairs
    calibration_metrics: Optional[Dict[str, float]] = None  # ECE, Brier, etc.
    selective_prediction: Optional[Dict[str, Any]] = None  # Risk-coverage analysis
    conformal_prediction: Optional[Dict[str, Any]] = None  # Conformal guarantees
    rejection_reasons_dict: Optional[Dict[str, int]] = None  # Full rejection reason breakdown
    
    def __post_init__(self):
        """Enforce invariants on creation."""
        # Invariant: verified + rejected + low_confidence == total_claims
        count_sum = self.verified_count + self.rejected_count + self.low_confidence_count
        if self.total_claims > 0 and count_sum != self.total_claims:
            logger.warning(
                f"Invariant violation: total_claims={self.total_claims} but "
                f"verified+rejected+low_conf={count_sum}. Adjusting total_claims."
            )
            self.total_claims = count_sum
        
        # Invariant: rejection_reasons must sum to rejected_count
        if self.rejection_reasons_dict and self.rejected_count > 0:
            reasons_sum = sum(self.rejection_reasons_dict.values())
            if reasons_sum != self.rejected_count:
                logger.warning(
                    f"Invariant violation: rejection_reasons sum to {reasons_sum} "
                    f"but rejected_count={self.rejected_count}. Adding UNKNOWN_REASON."
                )
                missing = self.rejected_count - reasons_sum
                if missing > 0:
                    self.rejection_reasons_dict["UNKNOWN_REASON"] = (
                        self.rejection_reasons_dict.get("UNKNOWN_REASON", 0) + missing
                    )


@dataclass
class ClaimEntry:
    """Single claim for table."""
    claim_text: str
    status: str
    confidence: float
    evidence_count: int
    top_evidence: str
    page_num: Optional[int] = None
    span_id: Optional[str] = None
    claim_type: str = "fact_claim"  # From ClaimType enum


class ResearchReportBuilder:
    """Generate comprehensive session reports in multiple formats."""

    def __init__(self):
        """Initialize report builder."""
        self.session_metadata: Optional[SessionMetadata] = None
        self.ingestion_report: Optional[IngestionReport] = None
        self.verification_summary: Optional[VerificationSummary] = None
        self.claims: List[ClaimEntry] = []
        self.performance_metrics: Dict[str, Any] = {}

    def add_session_metadata(self, metadata: SessionMetadata) -> "ResearchReportBuilder":
        """Add session metadata."""
        self.session_metadata = metadata
        return self

    def add_ingestion_report(self, report: IngestionReport) -> "ResearchReportBuilder":
        """Add ingestion statistics."""
        self.ingestion_report = report
        return self

    def add_verification_summary(self, summary: VerificationSummary) -> "ResearchReportBuilder":
        """Add verification statistics with invariant checks.
        
        Validates that the summary metrics are logically consistent:
        - total_claims == verified_count + rejected_count + low_confidence_count
        - If total_claims == 0: avg_confidence should be 0.0, rejection_reasons should be empty
        """
        if summary:
            # Invariant check: counts must sum to total_claims
            count_sum = summary.verified_count + summary.rejected_count + summary.low_confidence_count
            if summary.total_claims != count_sum:
                logger.warning(
                    f"VerificationSummary invariant violation: total_claims={summary.total_claims} "
                    f"but verified+rejected+low_conf={count_sum}. Correcting to match sum."
                )
                summary.total_claims = count_sum
            
            # If total_claims is 0, sanitize metrics
            if summary.total_claims == 0:
                if summary.avg_confidence != 0.0 and summary.avg_confidence is not None:
                    logger.warning(
                        f"VerificationSummary: total_claims=0 but avg_confidence={summary.avg_confidence}. "
                        f"Resetting to 0.0."
                    )
                    summary.avg_confidence = 0.0
                
                # Clear rejection reasons for zero claims
                if summary.top_rejection_reasons:
                    logger.warning(
                        f"VerificationSummary: total_claims=0 but has {len(summary.top_rejection_reasons)} "
                        f"rejection reasons. Clearing."
                    )
                    summary.top_rejection_reasons = []
        
        self.verification_summary = summary
        return self

    def add_claims(self, claims: List[ClaimEntry]) -> "ResearchReportBuilder":
        """Add claim entries."""
        self.claims = claims
        return self

    def add_performance_metrics(self, metrics: Dict[str, Any]) -> "ResearchReportBuilder":
        """Add performance metrics (inference time, memory, etc.)."""
        self.performance_metrics = metrics
        return self

    def build_markdown(self) -> str:
        """Build Markdown report."""
        sections = []

        # Title
        sections.append("# AI Verification Session Report")
        sections.append("")

        if self.session_metadata:
            sections.extend(self._build_md_session_section())

        if self.ingestion_report:
            sections.extend(self._build_md_ingestion_section())

        if self.verification_summary:
            sections.extend(self._build_md_verification_section())
            sections.extend(self._build_md_selective_prediction_section())

        # Trust guidance
        sections.extend(self._build_md_trust_guidance())

        # Claim table
        if self.claims:
            sections.extend(self._build_md_claim_table())

        # Performance
        if self.performance_metrics:
            sections.extend(self._build_md_performance_section())

        return"\n".join(sections)

    def build_html(self) -> str:
        """Build HTML report."""
        md_content = self.build_markdown()
        # Convert markdown to HTML using simple markdown parsing
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '  <meta charset="UTF-8">',
            "  <title>AI Verification Session Report</title>",
            "  <style>",
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
            "            line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; "
            "            color: #1d1d1f; background-color: #f5f5f7; }",
            "    h1 { color: #0a84ff; border-bottom: 3px solid #0a84ff; padding-bottom: 10px; }",
            "    h2 { color: #1d1d1f; margin-top: 30px; border-left: 4px solid #0a84ff; "
            "         padding-left: 15px; }",
            "    h3 { color: #424245; }",
            "    table { width: 100%; border-collapse: collapse; margin: 15px 0; }",
            "    th { background-color: #0a84ff; color: white; padding: 10px; text-align: left; }",
            "    td { border-bottom: 1px solid #e5e5ea; padding: 10px; }",
            "    tr:nth-child(even) { background-color: #f9f9f9; }",
            "    .trust-yes { color: #34c759; font-weight: bold; }",
            "    .trust-no { color: #ff3b30; font-weight: bold; }",
            "    .metric { background-color: white; padding: 15px; margin: 10px 0; "
            "              border-radius: 8px; border-left: 4px solid #34c759; }",
            "    code { background-color: #f0f0f0; padding: 2px 6px; border-radius: 4px; "
            "           font-family: 'Monaco', monospace; }",
            "  </style>",
            "</head>",
            "<body>",
        ]

        # Simple markdown to HTML conversion
        lines = md_content.split("\n")
        in_table = False
        for line in lines:
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_parts.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                if not in_table:
                    html_parts.append("<ul>")
                    in_table = True
                html_parts.append(f"<li>{line[2:]}</li>")
            elif line.strip() == "":
                if in_table:
                    html_parts.append("</ul>")
                    in_table = False
                html_parts.append("<br>")
            elif line.startswith("|"):
                # Simple table handling
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if not in_table:
                    html_parts.append("<table>")
                    in_table = True
                if all(c.startswith("-") for c in cells):
                    continue  # Skip separator row
                row_html = "<tr>"
                for cell in cells:
                    row_html += f"<td>{cell}</td>"
                row_html += "</tr>"
                html_parts.append(row_html)
            else:
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                if line.strip():
                    html_parts.append(f"<p>{line}</p>")

        if in_table:
            html_parts.append("</table>")

        html_parts.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def build_audit_json(self) -> Dict[str, Any]:
        """Build audit trail JSON."""
        return {
            "report_type": "audit",
            "generated_at": datetime.now().isoformat(),
            "session": asdict(self.session_metadata) if self.session_metadata else None,
            "ingestion": asdict(self.ingestion_report) if self.ingestion_report else None,
            "verification": asdict(self.verification_summary) if self.verification_summary else None,
            "claims": [asdict(c) for c in self.claims],
            "performance": self.performance_metrics,
            "metadata": {
                "report_version": "1.0",
                "total_sections": 6,
            }
        }

    def build_report(
        self
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Build complete report in all formats.

        Returns:
            Tuple of (markdown_str, html_str, audit_json_dict)
        """
        return (
            self.build_markdown(),
            self.build_html(),
            self.build_audit_json(),
        )

    # ========================================================================
    # MARKDOWN REPORT SECTIONS
    # ========================================================================

    def _build_md_session_section(self) -> List[str]:
        """Build session metadata section."""
        lines = ["## Session Information", ""]
        if not self.session_metadata:
            return lines

        meta = self.session_metadata
        lines.extend([
            f"- **Session ID**: `{meta.session_id}`",
            f"- **Timestamp**: {meta.timestamp}",
            f"- **Version**: {meta.version}",
            f"- **Random Seed**: {meta.seed}",
            "",
            "### Models Used",
            f"- **Language Model**: {meta.language_model}",
            f"- **Embedding Model**: {meta.embedding_model}",
            f"- **NLI Model**: {meta.nli_model}",
            "",
            "### Inputs",
            *[f"- {inp}" for inp in meta.inputs_used],
            "",
        ])
        return lines

    def _build_md_ingestion_section(self) -> List[str]:
        """Build ingestion statistics section.
        
        Enforces invariant: avg_chunk_size is None/N/A when total_chunks == 0
        """
        lines = ["## Ingestion Statistics", ""]
        if not self.ingestion_report:
            return lines

        ing = self.ingestion_report
        
        # Format avg_chunk_size properly: None or N/A when no chunks
        avg_chunk_display = "N/A"
        if ing.total_chunks > 0 and ing.avg_chunk_size is not None:
            avg_chunk_display = f"{ing.avg_chunk_size:.0f}"
        
        lines.extend([
            f"- **Total Pages**: {ing.total_pages}",
            f"- **Pages OCR'd**: {ing.pages_ocr}",
            f"- **Headers Removed**: {ing.headers_removed}",
            f"- **Footers Removed**: {ing.footers_removed}",
            f"- **Watermarks Removed**: {ing.watermarks_removed}",
            "",
            "### Extraction",
            f"- **Total Chunks**: {ing.total_chunks}",
            f"- **Avg Chunk Size**: {avg_chunk_display} chars",
            f"- **Methods**: {', '.join(ing.extraction_methods) if ing.extraction_methods else 'N/A'}",
            "",
        ])
        return lines

    def _build_md_verification_section(self) -> List[str]:
        """Build verification statistics section.
        
        Enforces logical consistency:
        - Only includes avg_confidence if total_claims > 0
        - Only includes rejection reasons if rejected_count > 0
        - Validates count invariants
        - Ensures rejection_reasons sum to rejected_count
        """
        lines = ["## Verification Results", ""]
        if not self.verification_summary:
            return lines

        ver = self.verification_summary
        
        # Validate invariant: total_claims == verified + rejected + low_confidence
        count_sum = ver.verified_count + ver.rejected_count + ver.low_confidence_count
        if ver.total_claims > 0 and count_sum != ver.total_claims:
            logger.warning(
                f"Invariant violation in report section: total_claims={ver.total_claims} but "
                f"verified={ver.verified_count} + rejected={ver.rejected_count} + "
                f"low_conf={ver.low_confidence_count} = {count_sum}"
            )
        
        # Validate rejection_reasons sum
        if ver.rejection_reasons_dict and ver.rejected_count > 0:
            reasons_sum = sum(ver.rejection_reasons_dict.values())
            if reasons_sum != ver.rejected_count:
                logger.warning(
                    f"Invariant violation: rejection_reasons sum to {reasons_sum} but "
                    f"rejected_count={ver.rejected_count}"
                )
        
        lines.extend([
            f"### Claim Status Distribution",
            f"- **Verified**: {ver.verified_count}/{ver.total_claims} "
            f"({100*ver.verified_count/max(ver.total_claims, 1):.1f}%)",
            f"- **Low Confidence**: {ver.low_confidence_count}/{ver.total_claims} "
            f"({100*ver.low_confidence_count/max(ver.total_claims, 1):.1f}%)",
            f"- **Rejected**: {ver.rejected_count}/{ver.total_claims} "
            f"({100*ver.rejected_count/max(ver.total_claims, 1):.1f}%)",
            "",
            f"### Overall Metrics",
        ])
        
        # Only include avg_confidence if there are claims
        if ver.total_claims > 0:
            lines.append(f"- **Average Confidence**: {ver.avg_confidence:.2f}")
        else:
            lines.append(f"- **Average Confidence**: N/A (no claims)")
        
        lines.append("")

        # Include rejection reasons breakdown (use full dict if available, otherwise top reasons)
        if ver.rejected_count > 0:
            lines.append("### Top Rejection Reasons")
            
            if ver.rejection_reasons_dict:
                # Use the full breakdown from rejection_reasons_dict
                sorted_reasons = sorted(
                    ver.rejection_reasons_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for reason, count in sorted_reasons:
                    percentage = 100 * count / ver.rejected_count if ver.rejected_count > 0 else 0
                    lines.append(f"- {reason}: {count} claims ({percentage:.1f}%)")
            elif ver.top_rejection_reasons:
                # Fallback to top_rejection_reasons tuple list
                for reason, count in ver.top_rejection_reasons:
                    percentage = 100 * count / ver.rejected_count if ver.rejected_count > 0 else 0
                    lines.append(f"- {reason}: {count} claims ({percentage:.1f}%)")
            
            lines.append("")

        if ver.calibration_metrics:
            lines.extend([
                "### Calibration Metrics",
                f"- **ECE (Expected Calibration Error)**: {ver.calibration_metrics.get('ece', 0):.4f}",
                f"- **Brier Score**: {ver.calibration_metrics.get('brier', 0):.4f}",
                "",
            ])

        return lines

    def _build_md_selective_prediction_section(self) -> List[str]:
        """Build selective prediction and conformal guarantees section."""
        lines = []
        
        if not self.verification_summary:
            return lines
        
        ver = self.verification_summary
        has_selective = ver.selective_prediction is not None
        has_conformal = ver.conformal_prediction is not None
        
        if not has_selective and not has_conformal:
            return lines
        
        lines.extend([
            "## Confidence Guarantees",
            "",
            "This section provides statistical guarantees about prediction reliability.",
            "",
        ])
        
        # Selective Prediction (Risk-Coverage Tradeoff)
        if has_selective:
            sp = ver.selective_prediction
            threshold = sp.get('optimal_threshold', 0.0)
            coverage = sp.get('achieved_coverage', 0.0)
            risk = sp.get('achieved_risk', 0.0)
            target_risk = sp.get('target_risk', 0.0)
            
            lines.extend([
                "### 🎯 Selective Prediction (Risk-Coverage Tradeoff)",
                "",
                "**Key Question**: _How many predictions should we accept to control error rate?_",
                "",
                f"**If we accept only claims with confidence ≥ {threshold:.3f}:**",
                f"- **Coverage**: {coverage:.1%} of claims accepted",
                f"- **Risk**: {risk:.1%} expected error rate on accepted claims",
                f"- **Rejected**: {(1-coverage):.1%} of claims (too uncertain)",
                "",
                f"**Interpretation**: To maintain error rate ≤ {target_risk:.1%}, "
                f"we should accept {coverage:.0%} of predictions and reject the rest.",
                "",
            ])
            
            # Add risk-coverage curve summary if available
            if 'auc_rc' in sp:
                auc = sp['auc_rc']
                lines.extend([
                    f"- **AUC-RC**: {auc:.4f} (lower is better, measures overall risk-coverage tradeoff)",
                    "",
                ])
        
        # Conformal Prediction (Distribution-Free Guarantees)
        if has_conformal:
            cp = ver.conformal_prediction
            threshold = cp.get('threshold', 0.0)
            alpha = cp.get('alpha', 0.0)
            confidence = cp.get('coverage_guarantee', 0.0)
            empirical_coverage = cp.get('empirical_coverage', 0.0)
            
            lines.extend([
                "### 🔒 Conformal Prediction (Distribution-Free Guarantees)",
                "",
                "**Key Question**: _With what confidence can we guarantee error control?_",
                "",
                f"**Calibrated Threshold**: {threshold:.3f}",
                f"- **Confidence Level**: {confidence:.1%} (1 - α where α = {alpha:.2f})",
                f"- **Guarantee**: With {confidence:.1%} confidence, predictions with "
                f"score ≥ {threshold:.3f} will have error rate ≤ {alpha:.1%}",
                f"- **Empirical Coverage**: {empirical_coverage:.1%} (observed on calibration set)",
                "",
                "**Interpretation**: This is a **finite-sample guarantee** that holds with high "
                "probability regardless of the true data distribution (exchangeability assumption).",
                "",
            ])
        
        # Combined Recommendation
        if has_selective and has_conformal:
            sp_threshold = ver.selective_prediction.get('optimal_threshold', 0.0)
            cp_threshold = ver.conformal_prediction.get('threshold', 0.0)
            combined_threshold = max(sp_threshold, cp_threshold)
            
            lines.extend([
                "### 💡 Combined Recommendation",
                "",
                f"**Conservative Threshold**: {combined_threshold:.3f} "
                f"(max of selective={sp_threshold:.3f}, conformal={cp_threshold:.3f})",
                "",
                "**Usage Guidance**:",
                f"- ✅ **Accept** claims with confidence ≥ {combined_threshold:.3f}",
                f"- ⚠️ **Review Manually** claims with {min(sp_threshold, cp_threshold):.3f} ≤ confidence < {combined_threshold:.3f}",
                f"- ❌ **Reject** claims with confidence < {min(sp_threshold, cp_threshold):.3f}",
                "",
            ])
        
        return lines

    def _build_md_trust_guidance(self) -> List[str]:
        """Build plain-language trust guidance."""
        lines = [
            "## What to Trust / What Not to Trust",
            "",
            "### ✅ You Can Trust Claims That Are:",
            "- **Verified with high confidence** (status = VERIFIED, confidence > 0.8)",
            "- **Supported by multiple evidence sources** from authoritative materials",
            "- **Not contradicted** by any retrieved evidence",
            "- **Common in domain literature** (not edge cases or disputed claims)",
            "",
            "### ⚠️ Use With Caution:",
            "- **Low-confidence claims** that lack multiple supporting sources",
            "- **Claims from specialized/technical domains** that may have nuanced context",
            "- **Claims where model confidence is 0.5-0.8** (middle range)",
            "",
            "### ❌ Do Not Trust:",
            "- **Rejected claims** - evidence contradicted or insufficient",
            "- **Claims with zero supporting evidence** from course materials",
            "- **Highly technical definitions** without expert review",
            "- **Edge case or controversial claims** marked as low confidence",
            "",
            "### 🔍 How to Verify Further:",
            "1. Check the evidence citations (linked to page numbers and snippets below)",
            "2. Cross-reference low-confidence claims with original texts",
            "3. For critical material, consult with instructor or textbook",
            "4. Review the confidence score and calibration metrics",
            "",
        ]
        return lines

    def _build_md_claim_table(self) -> List[str]:
        """Build claim table with citations.
        
        Enforces invariant:
        - QUESTION claim_type items never appear in rejected/verified/low_conf buckets
        - Questions only appear in "Questions Answered" section
        """
        lines = ["## Verified Claims Table", ""]
        
        if not self.claims:
            lines.append("_No claims to display._")
            return lines

        # Group by status and type
        # Separate QUESTION items from status-based categorization
        questions = [c for c in self.claims if c.claim_type == "question"]
        factual_claims = [c for c in self.claims if c.claim_type != "question"]
        
        # For factual claims, group by verification status
        verified = [c for c in factual_claims if c.status.upper() == "VERIFIED"]
        low_conf = [c for c in factual_claims if c.status.upper() == "LOW_CONFIDENCE"]
        rejected = [c for c in factual_claims if c.status.upper() == "REJECTED"]

        def _add_claim_subtable(
            title: str,
            claims_list: List[ClaimEntry],
            show_evidence: bool = True
        ):
            """Add a subtable of claims.
            
            Args:
                title: Section title
                claims_list: List of claims to display
                show_evidence: Whether to show evidence snippets
            """
            lines.extend([f"### {title}", ""])
            if not claims_list:
                lines.append("_None_\n")
                return

            lines.append("| Claim | Confidence | Evidence | Citation |")
            lines.append("|-------|-----------|----------|----------|")
            
            for claim in claims_list[:10]:  # Show top 10
                claim_short = claim.claim_text[:80].replace("|", "\\|")
                conf = f"{claim.confidence:.2f}"
                evid_count = str(claim.evidence_count)
                
                # Build citation - show when evidence_count > 0
                if claim.evidence_count > 0:
                    citation_parts = []
                    if claim.page_num is not None:
                        citation_parts.append(f"p.{claim.page_num}")
                    if claim.span_id:
                        citation_parts.append(f"`{claim.span_id[:20]}...`")
                    citation = " | ".join(citation_parts) if citation_parts else "Yes"
                else:
                    citation = "N/A"
                
                lines.append(f"| {claim_short}... | {conf} | {evid_count} | {citation} |")
            
            if len(claims_list) > 10:
                lines.append(f"_... and {len(claims_list) - 10} more_")
            lines.append("")

        # Add factual claims grouped by status
        _add_claim_subtable("✅ Verified Claims", verified)
        _add_claim_subtable("⚠️ Low-Confidence Claims", low_conf)
        
        # Add rejected claims (should NOT include questions)
        if rejected:
            _add_claim_subtable("❌ Rejected Claims", rejected)

        # Add questions in separate section (never verified/rejected)
        if questions:
            _add_claim_subtable("❓ Questions Answered", questions, show_evidence=False)
            
            # Log if any questions ended up in rejected/verified buckets (invariant violation)
            rejected_questions = [c for c in questions if c.status.upper() == "REJECTED"]
            verified_questions = [c for c in questions if c.status.upper() == "VERIFIED"]
            if rejected_questions or verified_questions:
                logger.warning(
                    f"Invariant violation: Found {len(rejected_questions)} rejected and "
                    f"{len(verified_questions)} verified questions - questions should not "
                    f"be marked as rejected or verified"
                )

        return lines

    def _build_md_performance_section(self) -> List[str]:
        """Build performance metrics section."""
        lines = ["## Performance Metrics", ""]
        
        for key, value in self.performance_metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.3f}")
            else:
                lines.append(f"- **{key}**: {value}")
        
        lines.append("")
        return lines


def build_report(
    session_metadata: SessionMetadata,
    ingestion_report: IngestionReport,
    verification_summary: VerificationSummary,
    claims: List[ClaimEntry],
    performance_metrics: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Convenience function to build a complete report.

    Returns:
        Tuple of (report_markdown, report_html, audit_json_dict)
    """
    builder = ResearchReportBuilder()
    builder.add_session_metadata(session_metadata)
    builder.add_ingestion_report(ingestion_report)
    builder.add_verification_summary(verification_summary)
    builder.add_claims(claims)
    if performance_metrics:
        builder.add_performance_metrics(performance_metrics)

    return builder.build_report()


def save_reports(
    base_path: Path,
    markdown_content: str,
    html_content: str,
    audit_json: Dict[str, Any],
    prefix: str = "report",
) -> Tuple[Path, Path, Path]:
    """
    Save reports to disk.

    Returns:
        Tuple of (md_path, html_path, json_path)
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    md_path = base_path / f"{prefix}.md"
    md_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Saved Markdown report to {md_path}")

    html_path = base_path / f"{prefix}.html"
    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Saved HTML report to {html_path}")

    json_path = base_path / f"{prefix}_audit.json"
    if _DEBUG_JSON_ASSERT:
        _assert_json_module()
    json_path.write_text(json.dumps(audit_json, indent=2), encoding="utf-8")
    logger.info(f"Saved audit JSON to {json_path}")

    return md_path, html_path, json_path
