"""
Robust PDF text extraction with page-level processing and OCR fallback.

This module implements intelligent PDF ingestion that:
1. Extracts text page-by-page with quality metrics
2. Applies OCR fallback only to low-quality pages
3. Preserves page-level provenance
4. Returns comprehensive ingestion report with diagnostics
"""

import logging
import re
import tempfile
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field

from src.exceptions import EvidenceIngestError
from src.preprocessing.text_cleaner import clean_extracted_text, CleanDiagnostics
from src.preprocessing.pdf_page_extractor import (
    extract_pages, 
    extract_page_with_ocr,
    PageText,
    get_extraction_summary
)
from src.preprocessing.pdf_layout import reorder_columns
import config

logger = logging.getLogger(__name__)

# Quality thresholds (kept for backward compatibility)
MIN_CHARS_FOR_OCR = 300
"""Minimum characters required before skipping OCR fallback"""

MIN_WORDS_FOR_QUALITY = 20
"""Minimum words to consider extraction successful"""

MIN_ALPHA_RATIO = 0.30
"""Minimum alphabetic character ratio for quality text"""

MIN_UNIQUE_CHARS = 20
"""Minimum unique characters to consider extraction successful"""


@dataclass
class PDFIngestionReport:
    """Comprehensive PDF ingestion report with diagnostics."""
    pages_total: int
    pages_ocr: int
    pages_low_quality: int
    headers_removed_count: int
    watermark_removed_count: int
    removed_lines_count: int
    extraction_method: str
    chars_extracted: int
    words_extracted: int
    alphabetic_ratio: float
    quality_assessment: str
    removed_patterns_hit: Dict[str, int] = field(default_factory=dict)
    removed_by_regex: Dict[str, int] = field(default_factory=dict)
    top_removed_lines: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "pages_total": self.pages_total,
            "pages_ocr": self.pages_ocr,
            "pages_low_quality": self.pages_low_quality,
            "headers_removed_count": self.headers_removed_count,
            "watermark_removed_count": self.watermark_removed_count,
            "removed_lines_count": self.removed_lines_count,
            "extraction_method": self.extraction_method,
            "chars_extracted": self.chars_extracted,
            "words_extracted": self.words_extracted,
            "alphabetic_ratio": self.alphabetic_ratio,
            "quality_assessment": self.quality_assessment,
            "removed_patterns_hit": self.removed_patterns_hit,
            "removed_by_regex": self.removed_by_regex,
            "top_removed_lines": self.top_removed_lines
        }


def _count_letters(text: str) -> int:
    """Count alphabetic characters in text."""
    return sum(1 for c in text if c.isalpha())


def _count_words(text: str) -> int:
    """Count words in text (split by whitespace)."""
    return len(text.split())


def _compute_alphabetic_ratio(text: str) -> float:
    """Compute ratio of alphabetic characters to total length."""
    if not text:
        return 0.0
    letters = _count_letters(text)
    return letters / len(text)


def _assess_extraction_quality(text: str) -> Tuple[bool, str]:
    """
    Assess if extracted text quality is acceptable.
    
    Returns:
        (is_good: bool, reason: str)
    """
    text = text.strip()
    
    if not text:
        return False, "Empty text"
    
    char_count = len(text)
    letters = _count_letters(text)
    words = _count_words(text)
    alpha_ratio = _compute_alphabetic_ratio(text)
    unique_chars = len(set(text))
    
    # Quality thresholds
    if char_count < MIN_CHARS_FOR_OCR:
        return False, f"Too few characters: {char_count} < {MIN_CHARS_FOR_OCR}"
    
    if words < MIN_WORDS_FOR_QUALITY:
        return False, f"Too few words: {words} < {MIN_WORDS_FOR_QUALITY}"
    
    if letters < MIN_CHARS_FOR_OCR:
        return False, f"Too few letters: {letters} < {MIN_CHARS_FOR_OCR}"
    
    if alpha_ratio < MIN_ALPHA_RATIO:
        return False, f"Low alphabetic ratio: {alpha_ratio:.3f} < {MIN_ALPHA_RATIO}"

    if unique_chars < MIN_UNIQUE_CHARS:
        return False, f"Too few unique characters: {unique_chars} < {MIN_UNIQUE_CHARS}"
    
    return True, "Good quality"


def _get_pdf_bytes(uploaded_file: Any) -> bytes:
    """Return raw PDF bytes from various input types."""
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)

    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()

    if hasattr(uploaded_file, "read"):
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return uploaded_file.read()

    raise ValueError("Unsupported PDF input type")


def extract_pdf_text(uploaded_file, ocr: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF using page-level processing with selective OCR fallback.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or file path
        ocr: Optional ImageOCR instance for OCR fallback
    
    Returns:
        Tuple of (extracted_text, metadata_dict with ingestion_report)
    
    Raises:
        EvidenceIngestError: If OCR is unavailable and text quality is insufficient
    
    Metadata includes:
        - extraction_method: primary extraction method used
        - ingestion_report: PDFIngestionReport object with full diagnostics
        - num_pages: number of pages processed
        - chars_extracted: character count
        - words: word count
        - alphabetic_ratio: alphabetic character ratio
        - quality_assessment: success/failure reason
    """
    # Get file bytes
    if hasattr(uploaded_file, 'name'):
        file_path = uploaded_file.name
        file_bytes = _get_pdf_bytes(uploaded_file)
    else:
        file_path = str(uploaded_file)
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    
    logger.info(f"Extracting PDF from: {file_path}")
    
    # Step 1: Extract pages with quality metrics
    logger.info("Step 1: Extracting pages with quality assessment...")
    pages = extract_pages(file_bytes, use_pdfplumber=True)
    
    if not pages:
        raise EvidenceIngestError(
            "EXTRACTION_FAILED",
            "No pages could be extracted from PDF",
            details={"file": file_path}
        )
    
    # Step 2: Apply per-page OCR fallback for low-quality pages
    logger.info("Step 2: Applying OCR fallback to low-quality pages...")
    pages_ocr = 0
    pages_low_quality = 0
    
    for i, page in enumerate(pages):
        if not page.quality_metrics.is_acceptable:
            pages_low_quality += 1
            
            # Try OCR fallback if enabled
            if config.ENABLE_OCR_FALLBACK:
                logger.info(f"  Page {page.page_num}: Low quality, attempting OCR...")
                try:
                    ocr_page = extract_page_with_ocr(file_bytes, page.page_num, ocr)
                    
                    # Only replace if OCR is better
                    if ocr_page.quality_metrics.is_acceptable:
                        logger.info(f"  Page {page.page_num}: OCR successful")
                        pages[i] = ocr_page
                        pages_ocr += 1
                    else:
                        logger.warning(f"  Page {page.page_num}: OCR also poor quality, keeping original")
                except Exception as e:
                    logger.warning(f"  Page {page.page_num}: OCR failed: {e}")
            else:
                logger.warning(f"  Page {page.page_num}: Low quality but OCR disabled")
    
    # Step 3: Combine page texts with page markers
    logger.info("Step 3: Combining page texts...")
    raw_combined_parts = []
    for page in pages:
        text = page.raw_text if page.raw_text else page.cleaned_text
        if text.strip():
            # Apply layout reordering if multi-column detected
            text = reorder_columns(text, safe_mode=True)
            raw_combined_parts.append(f"--- Page {page.page_num} ---\n{text}")
    
    raw_combined = "\n\n".join(raw_combined_parts)
    
    # Step 4: Clean text with diagnostics
    logger.info("Step 4: Cleaning text with header/footer removal...")
    cleaned_text, clean_diag = clean_extracted_text(raw_combined)
    
    # Step 5: Compute final metrics
    chars = len(cleaned_text)
    words = _count_words(cleaned_text)
    letters = _count_letters(cleaned_text)
    alpha_ratio = _compute_alphabetic_ratio(cleaned_text)
    
    # Determine primary extraction method
    extraction_methods = {}
    for page in pages:
        method = page.extraction_method
        extraction_methods[method] = extraction_methods.get(method, 0) + 1
    
    primary_method = max(extraction_methods.items(), key=lambda x: x[1])[0] if extraction_methods else "unknown"
    if pages_ocr > 0:
        primary_method = f"{primary_method}_with_ocr"
    
    # Create ingestion report
    ingestion_report = PDFIngestionReport(
        pages_total=len(pages),
        pages_ocr=pages_ocr,
        pages_low_quality=pages_low_quality,
        headers_removed_count=clean_diag.headers_removed_count,
        watermark_removed_count=clean_diag.watermark_removed_count,
        removed_lines_count=clean_diag.removed_lines_count,
        extraction_method=primary_method,
        chars_extracted=chars,
        words_extracted=words,
        alphabetic_ratio=alpha_ratio,
        quality_assessment=_assess_final_quality(cleaned_text),
        removed_patterns_hit=clean_diag.removed_patterns_hit,
        removed_by_regex=clean_diag.removed_by_regex,
        top_removed_lines=clean_diag.top_removed_lines
    )
    
    # Build metadata dict (backward compatible)
    metadata: Dict[str, Any] = {
        "extraction_method": primary_method,
        "method": primary_method,
        "extraction_method_used": primary_method,  # backward compat
        "num_pages": len(pages),
        "pages": len(pages),
        "ocr_pages": pages_ocr,
        "pages_low_quality": pages_low_quality,
        "chars_extracted": chars,
        "words": words,
        "letters": letters,
        "alphabetic_ratio": alpha_ratio,
        "quality_assessment": ingestion_report.quality_assessment,
        "ingestion_report": ingestion_report,
        "diagnostics": {
            "method": primary_method,
            "chars_extracted": chars,
            "word_count": words,
            "alphabetic_ratio": alpha_ratio,
            "unique_chars": len(set(cleaned_text)),
            "ocr_pages": pages_ocr,
            "pages_low_quality": pages_low_quality,
            "quality_assessment": ingestion_report.quality_assessment,
            "removed_lines_count": clean_diag.removed_lines_count,
            "headers_removed_count": clean_diag.headers_removed_count,
            "watermark_removed_count": clean_diag.watermark_removed_count,
            "removed_by_regex": clean_diag.removed_by_regex,
            "removed_patterns_hit": clean_diag.removed_patterns_hit,
            "removed_repeated_lines_count": clean_diag.removed_repeated_lines_count,
            "top_removed_lines": clean_diag.top_removed_lines,
            "repeat_threshold_used": clean_diag.repeat_threshold_used
        },
        "result": {
            "text": cleaned_text,
            "method": primary_method,
            "diagnostics": ingestion_report.to_dict()
        }
    }
    
    logger.info(
        f"Final extraction: {words} words, {chars} chars, "
        f"alpha_ratio={alpha_ratio:.3f}, method={primary_method}, "
        f"ocr_pages={pages_ocr}/{len(pages)}, low_quality={pages_low_quality}"
    )
    
    if clean_diag.removed_lines_count > 0:
        logger.info(
            f"Text cleaning removed {clean_diag.removed_lines_count} lines "
            f"(headers={clean_diag.headers_removed_count}, watermarks={clean_diag.watermark_removed_count})"
        )
    
    return cleaned_text, metadata


def _assess_final_quality(text: str) -> str:
    """Simple quality assessment for final text."""
    is_good, assessment = _assess_extraction_quality(text)
    return assessment


def extract_pdf_text_legacy(uploaded_file, ocr: Optional[Any] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Legacy extraction method (whole-document OCR).
    Kept for backward compatibility testing.
    
    Use extract_pdf_text() for new code.
    """
    metadata: Dict[str, Any] = {
        "extraction_method": None,
        "num_pages": 0,
        "chars_extracted": 0,
        "words": 0,
        "alphabetic_ratio": 0.0,
        "quality_assessment": "",
        "ocr_pages": 0,
        "diagnostics": {}
    }
    
    # Get file bytes
    if hasattr(uploaded_file, 'name'):
        file_path = uploaded_file.name
        file_bytes = _get_pdf_bytes(uploaded_file)
    else:
        file_path = str(uploaded_file)
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    
    logger.info(f"[LEGACY] Extracting PDF from: {file_path}")
    
    # Count pages once
    num_pages = _count_pdf_pages(file_bytes)
    metadata["num_pages"] = num_pages
    
    # Strategy 1: Try pdfplumber extraction
    logger.info("Strategy 1: Trying pdfplumber extraction...")
    extracted_text = _extract_with_pdfplumber(file_bytes)
    is_good, assessment = _assess_extraction_quality(extracted_text)

    if is_good:
        logger.info(f"✓ pdfplumber extraction succeeded: {assessment}")
        metadata["extraction_method"] = "pdf_text"
        metadata["quality_assessment"] = assessment
    else:
        logger.warning(f"✗ pdfplumber extraction failed: {assessment}")
        logger.info("Strategy 2: Falling back to OCR...")

        if not config.ENABLE_OCR_FALLBACK:
            raise EvidenceIngestError(
                "OCR_DISABLED",
                "OCR fallback is disabled by configuration.",
                details={"chars_extracted": len(extracted_text), "chars_required": MIN_CHARS_FOR_OCR}
            )

        try:
            logger.info("Attempting OCR with PyMuPDF rendering...")
            extracted_text, ocr_method, ocr_pages = _extract_with_ocr_pymupdf(
                file_bytes,
                ocr,
                max_pages=config.OCR_MAX_PAGES,
                dpi=config.OCR_DPI
            )
            metadata["ocr_pages"] = ocr_pages
            is_good, assessment = _assess_extraction_quality(extracted_text)

            if is_good:
                logger.info(f"✓ OCR extraction succeeded: {assessment}")
                metadata["extraction_method"] = ocr_method
                metadata["quality_assessment"] = assessment
            else:
                logger.warning(f"✗ OCR extraction insufficient: {assessment}")
                metadata["extraction_method"] = ocr_method
                metadata["quality_assessment"] = f"OCR incomplete: {assessment}"
        except EvidenceIngestError:
            raise
        except Exception as e:
            logger.error(f"OCR extraction failed with error: {e}")
            raise EvidenceIngestError(
                "OCR_FAILED",
                f"OCR extraction failed: {str(e)}",
                details={"error": str(e)}
            )
    
    # Clean text and compute final metrics
    extracted_text, clean_diag = clean_extracted_text(extracted_text)
    chars = len(extracted_text)
    letters = _count_letters(extracted_text)
    words = _count_words(extracted_text)
    alpha_ratio = _compute_alphabetic_ratio(extracted_text)
    
    metadata["chars_extracted"] = chars
    metadata["words"] = words
    metadata["alphabetic_ratio"] = alpha_ratio
    metadata["method"] = metadata["extraction_method"]
    metadata["diagnostics"] = {
        "method": metadata["extraction_method"],
        "chars_extracted": chars,
        "word_count": words,
        "alphabetic_ratio": alpha_ratio,
        "unique_chars": len(set(extracted_text)),
        "ocr_pages": metadata.get("ocr_pages", 0),
        "quality_assessment": metadata["quality_assessment"],
        "removed_lines_count": clean_diag.removed_lines_count,
        "removed_by_regex": clean_diag.removed_by_regex,
        "removed_repeated_lines_count": clean_diag.removed_repeated_lines_count,
        "top_removed_lines": clean_diag.top_removed_lines,
        "repeat_threshold_used": clean_diag.repeat_threshold_used
    }
    metadata["result"] = {
        "text": extracted_text,
        "method": metadata["extraction_method"],
        "diagnostics": metadata["diagnostics"]
    }
    
    logger.info(
        f"Final extraction: {words} words, {chars} chars, "
        f"alpha_ratio={alpha_ratio:.3f}, method={metadata['extraction_method']}, "
        f"ocr_pages={metadata.get('ocr_pages', 0)}"
    )
    if clean_diag.removed_lines_count > 0:
        logger.info(
            "Text cleaning removed %s lines (repeated=%s).",
            clean_diag.removed_lines_count,
            clean_diag.removed_repeated_lines_count
        )
    
    # Backward compatibility: also include old metadata keys
    metadata["extraction_method_used"] = metadata["extraction_method"]
    metadata["pages"] = metadata["num_pages"]
    metadata["letters"] = letters
    
    return extracted_text, metadata


def _extract_with_pymupdf(file_bytes: bytes) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        
        return "\n".join(text_parts)
    except ImportError:
        logger.debug("PyMuPDF (fitz) not available")
        return ""
    except Exception as e:
        logger.debug(f"PyMuPDF extraction error: {e}")
        return ""


def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    """Extract text using pdfplumber."""
    try:
        import pdfplumber
        from io import BytesIO
        
        pdf_stream = BytesIO(file_bytes)
        text_parts = []
        
        with pdfplumber.open(pdf_stream) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        return "\n".join(text_parts)
    except ImportError:
        logger.debug("pdfplumber not available")
        return ""
    except Exception as e:
        logger.debug(f"pdfplumber extraction error: {e}")
        return ""


def _extract_with_ocr(
    file_bytes: bytes,
    ocr,
    max_pages: int = None,
    dpi: int = 200
) -> Tuple[str, str, int]:
    """Backward-compatible OCR extraction wrapper."""
    return _extract_with_ocr_pymupdf(
        file_bytes=file_bytes,
        ocr=ocr,
        max_pages=max_pages,
        dpi=dpi
    )


def _extract_with_ocr_pymupdf(
    file_bytes: bytes,
    ocr,
    max_pages: int = None,
    dpi: int = 200
) -> Tuple[str, str, int]:
    """
    Extract text using OCR on PDF pages rendered with PyMuPDF.
    
    This method uses PyMuPDF (fitz) to render PDF pages to images,
    avoiding the need for pdf2image and Poppler dependencies.
    
    Args:
        file_bytes: PDF file bytes
        ocr: ImageOCR instance
        max_pages: Maximum pages to process (None = all pages)
    
    Returns:
        Tuple of (extracted_text, method_name, pages_processed)
    """
    doc = None
    try:
        import fitz
        from PIL import Image
        try:
            import pytesseract
        except Exception:
            pytesseract = None

        if pytesseract is None and ocr is None:
            raise EvidenceIngestError(
                "OCR_UNAVAILABLE",
                "OCR fallback requested but neither pytesseract nor EasyOCR is available."
            )
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
        
        text_parts = []
        method_used = "ocr_pymupdf_tesseract" if pytesseract is not None else "ocr_easyocr"
        logger.info(f"OCR: Processing {pages_to_process} pages with PyMuPDF rendering...")
        
        for page_num in range(pages_to_process):
            page = doc[page_num]
            
            try:
                # Render page to pixmap at requested DPI for OCR quality
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert pixmap to PIL Image
                img_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(img_bytes))
                
                # Save to temporary file for OCR when needed
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp, format="PNG")
                    tmp_path = tmp.name
                
                try:
                    logger.info(f"  OCRing page {page_num + 1}/{pages_to_process}...")
                    text = ""
                    method = None

                    if pytesseract is not None:
                        try:
                            text = pytesseract.image_to_string(image)
                            method = "ocr_pymupdf_tesseract"
                        except Exception as e:
                            logger.warning(f"    Tesseract OCR failed on page {page_num + 1}: {e}")

                    if (not text or not text.strip()) and ocr is not None:
                        result = ocr.extract_text_from_image(tmp_path, image_type="notes")
                        if isinstance(result, dict):
                            text = result.get("text", "")
                        else:
                            text = str(result)
                        method = "ocr_easyocr"
                    
                    if text and text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")
                        logger.info(f"    ✓ Extracted {len(text)} chars from page {page_num + 1}")
                        if method:
                            method_used = method
                    else:
                        logger.warning(f"    ⚠ No text found on page {page_num + 1}")
                        
                finally:
                    # Clean up temporary file
                    Path(tmp_path).unlink(missing_ok=True)
                    
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                continue
        
        extracted_text = "\n\n".join(text_parts)
        logger.info(f"OCR (PyMuPDF): Extracted {len(extracted_text)} chars from {len(text_parts)} pages")

        return extracted_text, method_used, pages_to_process
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise EvidenceIngestError(
            "OCR_FAILED",
            f"OCR extraction failed: {str(e)}",
            details={"error": str(e)}
        )
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _clean_text(text: str) -> str:
    """Deprecated: kept for backward compatibility."""
    cleaned, _ = clean_extracted_text(text)
    return cleaned


def _count_pdf_pages(file_bytes: bytes) -> int:
    """Count pages in PDF."""
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = len(doc)
        doc.close()
        return pages
    except Exception:
        try:
            import pdfplumber
            from io import BytesIO
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                return len(pdf.pages)
        except Exception:
            return 0
