"""
Research reporting module for Smart Notes sessions.

Provides comprehensive report generation in multiple formats:
- Markdown (.md) - human-readable summaries
- HTML (.html) - styled web-viewable reports  
- JSON (.json) - structured audit trails
"""

from .research_report import (
    ResearchReportBuilder,
    SessionMetadata,
    IngestionReport,
    VerificationSummary,
    ClaimEntry,
    build_report,
    save_reports,
)

__all__ = [
    "ResearchReportBuilder",
    "SessionMetadata",
    "IngestionReport",
    "VerificationSummary",
    "ClaimEntry",
    "build_report",
    "save_reports",
]
