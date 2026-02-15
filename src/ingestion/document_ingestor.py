"""
Comprehensive document ingestion module for Smart Notes.

Supports robust extraction from:
- PDFs (text-based, scanned, corrupted)
- Images (JPG, PNG with OCR)
- URLs (YouTube transcripts, web articles)

Features:
- Multi-strategy fallback (PyMuPDF -> pdfplumber -> OCR)
- Scanned PDF detection
- OCR graceful degradation
- Quality assessment and diagnostics
"""

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF (fitz) not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available for image processing")

# Detection thresholds
MIN_CHARS_EXTRACTED = 100
"""Minimum characters to consider extraction 'successful'"""

MIN_WORDS_EXTRACTED = 10
"""Minimum words to consider extraction 'successful'"""

SCANNED_CHARS_THRESHOLD = 300
"""Below this, likely scanned PDF"""

SCANNED_SPACE_RATIO_THRESHOLD = 0.05
"""Threshold for space character ratio to detect scanned"""

MAX_NONPRINTABLE_RATIO = 0.1
"""Maximum non-printable character ratio before flagging corruption"""

OCR_RENDER_DPI = 200
"""DPI for rendering PDF pages to images for OCR"""


@dataclass
class IngestionDiagnostics:
    """Diagnostics output for ingestion process."""
    extracted_text_length: int
    scanned_detected: bool
    ocr_used: bool
    ocr_error: Optional[str]
    extraction_method: str  # "pymupdf", "pdfplumber", "ocr", "image", "error"
    pages_processed: int
    quality_score: float  # 0-1
    first_300_chars: str
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def detect_scanned_or_low_text(extracted_text: str) -> Tuple[bool, Dict[str, float]]:
    """
    Detect if extracted text is from scanned PDF or corrupted.
    
    Heuristics:
    - Length < threshold
    - High ratio of non-alphabetic characters
    - Too many nonprintable characters
    - Unusual spacing patterns
    
    Args:
        extracted_text: Extracted text from PDF
    
    Returns:
        Tuple of (is_scanned, metrics_dict)
    """
    if not extracted_text:
        return True, {"reason": "empty_text"}
    
    text = extracted_text.strip()
    length = len(text)
    
    # Metric 1: Text too short
    if length < SCANNED_CHARS_THRESHOLD:
        logger.debug(f"Text too short ({length} chars < {SCANNED_CHARS_THRESHOLD}), likely scanned")
        return True, {
            "reason": "text_too_short",
            "length": length,
            "threshold": SCANNED_CHARS_THRESHOLD
        }
    
    # Metric 2: Non-printable characters
    nonprintable = sum(1 for c in text if ord(c) < 32 and c not in '\n\t\r ')
    nonprintable_ratio = nonprintable / length if length > 0 else 0
    
    if nonprintable_ratio > MAX_NONPRINTABLE_RATIO:
        logger.debug(f"High nonprintable ratio ({nonprintable_ratio:.1%}), likely corruption")
        return True, {
            "reason": "high_nonprintable_ratio",
            "ratio": nonprintable_ratio,
            "threshold": MAX_NONPRINTABLE_RATIO
        }
    
    # Metric 3: Space ratio analysis
    spaces = text.count(' ')
    space_ratio = spaces / length if length > 0 else 0
    
    if space_ratio < SCANNED_SPACE_RATIO_THRESHOLD:
        logger.debug(f"Low space ratio ({space_ratio:.1%}), unusual structure")
        return True, {
            "reason": "low_space_ratio",
            "ratio": space_ratio,
            "threshold": SCANNED_SPACE_RATIO_THRESHOLD
        }
    
    # Metrics passed - likely good text-based PDF
    return False, {
        "reason": "looks_good",
        "length": length,
        "nonprintable_ratio": nonprintable_ratio,
        "space_ratio": space_ratio
    }


def extract_text_from_pdf_pymupdf(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
    """
    Extract text from PDF using PyMuPDF (fitz).
    
    Args:
        pdf_bytes: PDF file bytes
    
    Returns:
        Tuple of (text, metadata, exception)
    """
    if not PYMUPDF_AVAILABLE:
        return "", {}, ImportError("PyMuPDF (fitz) not available")
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        extracted_text = ""
        pages_processed = 0
        
        for page_num, page in enumerate(pdf_document):
            try:
                text = page.get_text("text")
                if text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                    pages_processed += 1
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        pdf_document.close()
        
        return extracted_text.strip(), {
            "pages_processed": pages_processed,
            "total_pages": len(pdf_document),
            "method": "pymupdf"
        }, None
    
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return "", {}, e


def extract_text_from_pdf_pdfplumber(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
    """
    Extract text from PDF using pdfplumber.
    
    Fallback strategy when PyMuPDF fails.
    
    Args:
        pdf_bytes: PDF file bytes
    
    Returns:
        Tuple of (text, metadata, exception)
    """
    if not PDFPLUMBER_AVAILABLE:
        return "", {}, ImportError("pdfplumber not available")
    
    try:
        import io
        extracted_text = ""
        pages_processed = 0
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
                        pages_processed += 1
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1} with pdfplumber: {e}")
        
        return extracted_text.strip(), {
            "pages_processed": pages_processed,
            "total_pages": total_pages,
            "method": "pdfplumber"
        }, None
    
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return "", {}, e


def extract_text_from_pdf_ocr(
    pdf_bytes: bytes,
    ocr_instance: Optional[Any] = None,
    max_pages: int = 5
) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
    """
    Extract text from PDF using OCR (render to images + easyocr).
    
    Fallback when text extraction fails. Gracefully handles missing OCR.
    
    Args:
        pdf_bytes: PDF file bytes
        ocr_instance: ImageOCR instance (optional)
        max_pages: Maximum pages to OCR
    
    Returns:
        Tuple of (text, metadata, exception)
    """
    if not PYMUPDF_AVAILABLE:
        error = ImportError("PyMuPDF required for OCR fallback")
        return "", {"method": "ocr", "error": str(error)}, error
    
    if ocr_instance is None:
        error = RuntimeError("OCR instance not provided for PDF OCR fallback")
        return "", {"method": "ocr", "error": str(error)}, error
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_text = ""
        pages_ocr_count = 0
        
        total_pages = len(pdf_document)
        pages_to_process = min(max_pages, total_pages)
        
        for page_num in range(pages_to_process):
            try:
                page = pdf_document[page_num]
                
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72))
                image_bytes = pix.tobytes("png")
                
                # Run OCR
                text = ocr_instance.perform_ocr_bytes(image_bytes)
                
                if text and text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} (OCR) ---\n{text}\n"
                    pages_ocr_count += 1
            
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
        
        pdf_document.close()
        
        return extracted_text.strip(), {
            "pages_ocr_count": pages_ocr_count,
            "total_pages": total_pages,
            "method": "ocr",
            "dpi": OCR_RENDER_DPI
        }, None
    
    except Exception as e:
        logger.error(f"PDF OCR extraction failed: {e}")
        return "", {"method": "ocr", "error": str(e)}, e


def extract_text_from_image(
    image_bytes: bytes,
    ocr_instance: Optional[Any] = None
) -> Tuple[str, Dict[str, Any], Optional[Exception]]:
    """
    Extract text from image using OCR.
    
    Args:
        image_bytes: Image file bytes
        ocr_instance: ImageOCR instance
    
    Returns:
        Tuple of (text, metadata, exception)
    """
    if ocr_instance is None:
        return "", {"method": "image_ocr", "error": "No OCR instance"}, RuntimeError("OCR not available")
    
    try:
        text = ocr_instance.perform_ocr_bytes(image_bytes)
        
        return text or "", {
            "method": "image_ocr",
            "length": len(text or "")
        }, None
    
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return "", {"method": "image_ocr", "error": str(e)}, e


def ingest_document(
    file_bytes: bytes,
    filename: str,
    ocr_instance: Optional[Any] = None,
    file_type: Optional[str] = None
) -> Tuple[str, IngestionDiagnostics]:
    """
    Ingest document with strong fallback chains.
    
    **Multi-strategy PDF extraction**:
    1. PyMuPDF (fast, modern)
    2. pdfplumber (alternative engine)
    3. OCR (render + easyocr)
    
    **Image extraction**:
    1. Direct OCR on image
    
    Args:
        file_bytes: File content bytes
        filename: Original filename
        ocr_instance: Optional ImageOCR instance for fallback
        file_type: Optional file type hint ("pdf", "image", "auto")
    
    Returns:
        Tuple of (extracted_text, diagnostics)
    """
    logger.info(f"Ingesting document: {filename}")
    
    diagnostics = IngestionDiagnostics(
        extracted_text_length=0,
        scanned_detected=False,
        ocr_used=False,
        ocr_error=None,
        extraction_method="error",
        pages_processed=0,
        quality_score=0.0,
        first_300_chars="",
        warnings=[],
        errors=[]
    )
    
    # Determine file type
    if file_type is None:
        file_type = "auto"
    
    if file_type == "auto":
        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            file_type = "pdf"
        elif any(filename_lower.endswith(f".{ext}") for ext in ["jpg", "jpeg", "png", "bmp", "gif"]):
            file_type = "image"
        else:
            diagnostics.errors.append(f"Unknown file type: {filename}")
            return "", diagnostics
    
    extracted_text = ""
    
    # =====================================================================
    # PDF EXTRACTION PIPELINE
    # =====================================================================
    if file_type.lower() == "pdf":
        logger.info(f"Processing PDF: {filename}")
        
        # Strategy 1: PyMuPDF
        if PYMUPDF_AVAILABLE:
            logger.debug("Attempting PyMuPDF extraction...")
            text1, meta1, exc1 = extract_text_from_pdf_pymupdf(file_bytes)
            
            if text1 and len(text1) > MIN_CHARS_EXTRACTED:
                logger.info(f"PyMuPDF extraction successful: {len(text1)} chars")
                extracted_text = text1
                diagnostics.extraction_method = "pymupdf"
                diagnostics.pages_processed = meta1.get("pages_processed", 0)
            elif exc1:
                logger.warning(f"PyMuPDF failed: {exc1}")
                diagnostics.warnings.append(f"PyMuPDF failed: {str(exc1)}")
            else:
                logger.debug(f"PyMuPDF returned too little text: {len(text1)} chars")
                diagnostics.warnings.append("PyMuPDF returned insufficient text")
        
        # Strategy 2: pdfplumber (if PyMuPDF failed or insufficient)
        if not extracted_text and PDFPLUMBER_AVAILABLE:
            logger.debug("Attempting pdfplumber extraction...")
            text2, meta2, exc2 = extract_text_from_pdf_pdfplumber(file_bytes)
            
            if text2 and len(text2) > MIN_CHARS_EXTRACTED:
                logger.info(f"pdfplumber extraction successful: {len(text2)} chars")
                extracted_text = text2
                diagnostics.extraction_method = "pdfplumber"
                diagnostics.pages_processed = meta2.get("pages_processed", 0)
            else:
                logger.debug(f"pdfplumber also insufficient: {len(text2)} chars")
                diagnostics.warnings.append("pdfplumber also insufficient")
        
        # Quality check for OCR fallback decision
        if extracted_text:
            is_scanned, scan_metrics = detect_scanned_or_low_text(extracted_text)
            diagnostics.scanned_detected = is_scanned
            
            if is_scanned:
                logger.info(f"Document appears scanned (metrics: {scan_metrics}), will attempt OCR")
                diagnostics.warnings.append(f"Scanned PDF detected: {scan_metrics}")
        
        # Strategy 3: OCR (if previous failed or scanned detected)
        if (not extracted_text or diagnostics.scanned_detected) and ocr_instance is not None:
            logger.debug("Attempting OCR extraction...")
            text3, meta3, exc3 = extract_text_from_pdf_ocr(file_bytes, ocr_instance)
            
            if text3 and len(text3) > MIN_CHARS_EXTRACTED:
                logger.info(f"OCR extraction successful: {len(text3)} chars")
                # Prefer OCR if it's significantly better
                if not extracted_text or len(text3) > len(extracted_text) * 1.2:
                    extracted_text = text3
                    diagnostics.extraction_method = "ocr"
                    diagnostics.ocr_used = True
                    diagnostics.pages_processed = meta3.get("pages_ocr_count", 0)
            elif exc3:
                diagnostics.ocr_error = str(exc3)
                logger.warning(f"OCR failed: {exc3}")
                diagnostics.warnings.append(f"OCR unavailable: {str(exc3)}")
    
    # =====================================================================
    # IMAGE EXTRACTION PIPELINE
    # =====================================================================
    elif file_type.lower() in ["image", "jpg", "jpeg", "png", "bmp", "gif"]:
        logger.info(f"Processing image: {filename}")
        
        if ocr_instance is None:
            diagnostics.errors.append("OCR instance required for image extraction")
            diagnostics.ocr_error = "OCR not available"
            return "", diagnostics
        
        logger.debug("Attempting image OCR...")
        text, meta, exc = extract_text_from_image(file_bytes, ocr_instance)
        
        if text and len(text) > MIN_CHARS_EXTRACTED:
            logger.info(f"Image OCR successful: {len(text)} chars")
            extracted_text = text
            diagnostics.extraction_method = "image_ocr"
            diagnostics.ocr_used = True
            diagnostics.pages_processed = 1
        else:
            if exc:
                diagnostics.errors.append(f"Image OCR failed: {str(exc)}")
                diagnostics.ocr_error = str(exc)
            else:
                diagnostics.errors.append(f"Image extraction returned insufficient text: {len(text)} chars")
            logger.error(f"Image extraction failed")
    
    else:
        diagnostics.errors.append(f"Unsupported file type: {file_type}")
    
    # =====================================================================
    # FINALIZE DIAGNOSTICS
    # =====================================================================
    
    if extracted_text:
        diagnostics.extracted_text_length = len(extracted_text)
        diagnostics.first_300_chars = extracted_text[:300]
        diagnostics.quality_score = min(1.0, len(extracted_text) / 1000)  # Cap at 1.0
        
        logger.info(
            f"✓ Document ingestion complete: {len(extracted_text)} chars, "
            f"method: {diagnostics.extraction_method}, quality: {diagnostics.quality_score:.1%}"
        )
    else:
        diagnostics.quality_score = 0.0
        logger.error(f"✗ Document ingestion failed: no text extracted")
    
    return extracted_text, diagnostics
