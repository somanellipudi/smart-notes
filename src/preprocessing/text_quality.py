"""
Text quality assessment module for input validation.

Provides comprehensive quality metrics and assessment for extracted text,
detecting issues like low alphabetic content, CID glyphs (corrupted PDFs),
and other quality problems that indicate extraction failures.
"""

import re
import string
import logging
from typing import Dict, Any
from dataclasses import dataclass, asdict
import config

logger = logging.getLogger(__name__)


@dataclass
class TextQualityReport:
    """
    Quality assessment report for extracted text.
    
    Attributes:
        text_length: Total characters in text
        alphabetic_ratio: Proportion of alphabetic characters (0-1)
        cid_ratio: Proportion of CID glyphs indicating corruption
        printable_ratio: Proportion of printable characters (0-1)
        passes_quality: True if all thresholds pass
        failure_reasons: List of specific quality failures (if any)
        is_unverifiable: True if text fails absolute minimum checks
    """
    text_length: int
    alphabetic_ratio: float
    cid_ratio: float
    printable_ratio: float
    passes_quality: bool
    failure_reasons: list
    is_unverifiable: bool


def compute_text_quality(text: str) -> TextQualityReport:
    """
    Compute comprehensive quality metrics for extracted text.
    
    Args:
        text: Extracted text to assess
        
    Returns:
        TextQualityReport with all quality metrics and pass/fail status
    """
    if not text:
        return TextQualityReport(
            text_length=0,
            alphabetic_ratio=0.0,
            cid_ratio=1.0,
            printable_ratio=0.0,
            passes_quality=False,
            failure_reasons=["Empty text"],
            is_unverifiable=True
        )
    
    text_len = len(text)
    
    # Compute metrics
    alphabetic_count = sum(1 for c in text if c.isalpha())
    alphabetic_ratio = alphabetic_count / text_len if text_len > 0 else 0.0
    
    cid_count = len(re.findall(r"\(cid:", text, re.IGNORECASE))
    cid_ratio = cid_count / max(1, text_len)
    
    printable_count = sum(1 for c in text if c in string.printable)
    printable_ratio = printable_count / text_len if text_len > 0 else 0.0
    
    # Assess quality
    failure_reasons = []
    
    if alphabetic_ratio < config.MIN_ALPHABETIC_RATIO:
        failure_reasons.append(
            f"Low alphabetic ratio: {alphabetic_ratio:.2%} < {config.MIN_ALPHABETIC_RATIO:.2%}"
        )
    
    if cid_ratio > config.MAX_CID_RATIO:
        failure_reasons.append(
            f"High CID glyph ratio: {cid_ratio:.4f} > {config.MAX_CID_RATIO:.4f} "
            "(corrupted PDF - OCR fallback may help)"
        )
    
    if printable_ratio < config.MIN_PRINTABLE_RATIO:
        failure_reasons.append(
            f"Low printable ratio: {printable_ratio:.2%} < {config.MIN_PRINTABLE_RATIO:.2%}"
        )
    
    # Absolute minimum check
    is_unverifiable = text_len < config.MIN_INPUT_CHARS_ABSOLUTE
    if is_unverifiable:
        failure_reasons.append(
            f"Text too short: {text_len} chars < {config.MIN_INPUT_CHARS_ABSOLUTE} absolute minimum"
        )
    
    # Check if passes all thresholds
    passes_quality = (
        not failure_reasons and
        alphabetic_ratio >= config.MIN_ALPHABETIC_RATIO and
        cid_ratio <= config.MAX_CID_RATIO and
        printable_ratio >= config.MIN_PRINTABLE_RATIO and
        text_len >= config.MIN_INPUT_CHARS_ABSOLUTE
    )
    
    return TextQualityReport(
        text_length=text_len,
        alphabetic_ratio=alphabetic_ratio,
        cid_ratio=cid_ratio,
        printable_ratio=printable_ratio,
        passes_quality=passes_quality,
        failure_reasons=failure_reasons,
        is_unverifiable=is_unverifiable
    )


def assess_quality_with_fallback(
    text: str,
    enable_ocr_fallback: bool = None
) -> Dict[str, Any]:
    """
    Assess text quality and recommend OCR fallback if appropriate.
    
    Args:
        text: Extracted text to assess
        enable_ocr_fallback: If True, recommend OCR fallback for PDFs
                            (default: from config)
    
    Returns:
        Dict with:
            - quality_report: TextQualityReport
            - recommend_ocr_fallback: Boolean indicating if OCR fallback recommended
            - status: 'PASS', 'QUALITY_ISSUE', or 'UNVERIFIABLE'
    """
    if enable_ocr_fallback is None:
        enable_ocr_fallback = config.ENABLE_OCR_FALLBACK
    
    report = compute_text_quality(text)
    
    # Determine recommendation
    recommend_ocr = (
        enable_ocr_fallback and
        not report.passes_quality and
        not report.is_unverifiable and
        "CID glyph" in str(report.failure_reasons)
    )
    
    # Determine status
    if report.is_unverifiable:
        status = 'UNVERIFIABLE'
    elif report.passes_quality:
        status = 'PASS'
    else:
        status = 'QUALITY_ISSUE'
    
    return {
        'quality_report': report,
        'recommend_ocr_fallback': recommend_ocr,
        'status': status
    }


def log_quality_report(report: TextQualityReport, context: str = "") -> None:
    """
    Log quality assessment results.
    
    Args:
        report: TextQualityReport to log
        context: Optional context string (e.g., "PDF page 1 OCR")
    """
    prefix = f"[{context}] " if context else ""
    logger.info(f"{prefix}Text Quality Report:")
    logger.info(f"  Length: {report.text_length} chars")
    logger.info(f"  Alphabetic ratio: {report.alphabetic_ratio:.2%}")
    logger.info(f"  CID glyph ratio: {report.cid_ratio:.4f}")
    logger.info(f"  Printable ratio: {report.printable_ratio:.2%}")
    logger.info(f"  Status: {'PASS' if report.passes_quality else 'FAIL'}")
    
    if report.failure_reasons:
        for reason in report.failure_reasons:
            logger.warning(f"  ✗ {reason}")
    
    if report.is_unverifiable:
        logger.error(f"{prefix}Text is UNVERIFIABLE (below absolute minimum)")
