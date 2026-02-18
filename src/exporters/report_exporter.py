"""
Export utilities for verifiability reports.
"""

import json
import textwrap
from typing import Any, Dict


def export_report_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


def export_report_pdf(report: Dict[str, Any]) -> bytes:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required to export PDF reports.") from exc

    def wrap_lines(text: str, width: int = 90) -> list[str]:
        return textwrap.wrap(text, width=width) if text else []

    def add_line(page_obj, line: str, y_pos: float) -> float:
        page_obj.insert_text((72, y_pos), line, fontsize=11)
        return y_pos + 14

    doc = fitz.open()
    page = doc.new_page()
    y = 72

    def ensure_space(lines_needed: int = 1):
        nonlocal page, y
        if y + (lines_needed * 14) > page.rect.height - 72:
            page = doc.new_page()
            y = 72

    title = f"Verifiability Report: {report.get('session_id', 'session')}"
    for line in wrap_lines(title, 80):
        ensure_space()
        y = add_line(page, line, y)
    ensure_space()
    y = add_line(page, f"Generated: {report.get('generated_at', '')}", y)
    ensure_space()
    y = add_line(page, "", y)

    metrics = report.get("metrics", {})
    summary_lines = [
        f"Total claims: {metrics.get('total_claims', report.get('claim_count', 0))}",
        f"Verified: {metrics.get('verified_claims', 0)}",
        f"Low confidence: {metrics.get('low_confidence_claims', 0)}",
        f"Rejected: {metrics.get('rejected_claims', 0)}",
        f"Rejection rate: {metrics.get('rejection_rate', 0):.1%}",
        f"Avg confidence: {metrics.get('avg_confidence', 0):.2f}",
    ]
    for line in summary_lines:
        ensure_space()
        y = add_line(page, line, y)

    y = add_line(page, "", y)

    def render_claim_table(title_text: str, claims: list[Dict[str, Any]]):
        nonlocal page, y
        ensure_space(2)
        y = add_line(page, title_text, y)
        if not claims:
            y = add_line(page, "(none)", y)
            return

        for claim in claims:
            claim_text = (claim.get("claim_text") or claim.get("text") or "").strip()
            claim_text = claim_text if len(claim_text) <= 140 else claim_text[:137] + "..."
            evidence = claim.get("evidence", [])
            top_evidence = evidence[0].get("snippet", "") if evidence else ""
            top_evidence = top_evidence if len(top_evidence) <= 160 else top_evidence[:157] + "..."
            line = (
                f"- {claim_text} (conf: {claim.get('confidence', 0):.2f}, "
                f"evidence: {len(evidence)})"
            )
            for wrapped in wrap_lines(line, 92):
                ensure_space()
                y = add_line(page, wrapped, y)
            if top_evidence:
                for wrapped in wrap_lines(f"  Top evidence: {top_evidence}", 92):
                    ensure_space()
                    y = add_line(page, wrapped, y)
        y = add_line(page, "", y)

    claims = report.get("claims", [])
    verified = [c for c in claims if str(c.get("status", "")).lower() == "verified"]
    low_conf = [c for c in claims if str(c.get("status", "")).lower() == "low_confidence"]
    rejected = [c for c in claims if str(c.get("status", "")).lower() == "rejected"]

    render_claim_table("Verified claims", verified)
    render_claim_table("Low-confidence claims", low_conf)
    render_claim_table("Rejected claims", rejected)

    return doc.tobytes()
