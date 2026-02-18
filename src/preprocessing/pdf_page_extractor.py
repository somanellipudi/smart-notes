"""
Page-level PDF extraction with quality metrics and OCR fallback.

This module provides per-page extraction for PDFs, enabling:
- Page-by-page quality assessment
- Selective OCR fallback (only bad pages)
- Page-level provenance tracking
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Text quality metrics for a page."""
    word_count: int = 0
    alpha_ratio: float = 0.0  # Alphabetic chars / total chars
    unique_char_ratio: float = 0.0  # Unique chars / total chars
    nonprintable_ratio: float = 0.0  # Control chars / total chars
    suspicious_glyph_ratio: float = 0.0  # CID patterns, boxes, etc.
    avg_word_length: float = 0.0
    line_count: int = 0
    is_acceptable: bool = False


@dataclass
class PageText:
    """Text extracted from a single PDF page."""
    page_num: int
    raw_text: str
    cleaned_text: str = ""
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    used_ocr: bool = False
    extraction_method: str = "pdf_text"  # "pdf_text" | "ocr_tesseract" | "ocr_easyocr"
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_quality_metrics(text: str) -> QualityMetrics:
    """
    Compute text quality metrics for extracted page.
    
    Args:
        text: Extracted page text
    
    Returns:
        QualityMetrics with computed values
    """
    if not text:
        return QualityMetrics(is_acceptable=False)
    
    text_len = len(text)
    if text_len == 0:
        return QualityMetrics(is_acceptable=False)
    
    # Count character types
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    nonprintable_count = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    unique_chars = len(set(text))
    
    # Check for suspicious glyphs (CID patterns, replacement chars)
    import re
    cid_patterns = len(re.findall(r'\(cid:\d+\)', text))
    box_chars = text.count('□') + text.count('▯') + text.count('�')
    suspicious_count = cid_patterns + box_chars
    
    # Word analysis
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    # Line count
    lines = [line for line in text.split('\n') if line.strip()]
    line_count = len(lines)
    
    # Calculate ratios
    alpha_ratio = alpha_count / text_len if text_len > 0 else 0.0
    unique_char_ratio = unique_chars / text_len if text_len > 0 else 0.0
    nonprintable_ratio = nonprintable_count / text_len if text_len > 0 else 0.0
    suspicious_glyph_ratio = suspicious_count / max(text_len, 1)
    
    # Quality thresholds
    MIN_ALPHA_RATIO = 0.30
    MIN_WORD_COUNT = 20
    MIN_UNIQUE_CHARS = 20
    MAX_NONPRINTABLE_RATIO = 0.10
    MAX_SUSPICIOUS_GLYPH_RATIO = 0.05
    
    is_acceptable = (
        alpha_ratio >= MIN_ALPHA_RATIO and
        word_count >= MIN_WORD_COUNT and
        unique_chars >= MIN_UNIQUE_CHARS and
        nonprintable_ratio <= MAX_NONPRINTABLE_RATIO and
        suspicious_glyph_ratio <= MAX_SUSPICIOUS_GLYPH_RATIO
    )
    
    return QualityMetrics(
        word_count=word_count,
        alpha_ratio=alpha_ratio,
        unique_char_ratio=unique_char_ratio,
        nonprintable_ratio=nonprintable_ratio,
        suspicious_glyph_ratio=suspicious_glyph_ratio,
        avg_word_length=avg_word_length,
        line_count=line_count,
        is_acceptable=is_acceptable
    )


def extract_pages(
    pdf_bytes: bytes,
    use_pdfplumber: bool = True
) -> List[PageText]:
    """
    Extract text from PDF page-by-page.
    
    Args:
        pdf_bytes: PDF file bytes
        use_pdfplumber: Try pdfplumber if available
    
    Returns:
        List of PageText objects (one per page)
    """
    pages = []
    
    # Try pdfplumber first
    if use_pdfplumber:
        try:
            import pdfplumber
            from io import BytesIO
            
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        raw_text = page.extract_text() or ""
                        quality = compute_quality_metrics(raw_text)
                        
                        pages.append(PageText(
                            page_num=i + 1,
                            raw_text=raw_text,
                            quality_metrics=quality,
                            extraction_method="pdfplumber",
                            metadata={"width": page.width, "height": page.height}
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to extract page {i+1} with pdfplumber: {e}")
                        pages.append(PageText(
                            page_num=i + 1,
                            raw_text="",
                            quality_metrics=QualityMetrics(is_acceptable=False),
                            extraction_method="failed",
                            metadata={"error": str(e)}
                        ))
                
                logger.info(f"Extracted {len(pages)} pages with pdfplumber")
                return pages
        
        except ImportError:
            logger.debug("pdfplumber not available, trying PyMuPDF")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
    
    # Fallback to PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(len(doc)):
            try:
                page = doc[i]
                raw_text = page.get_text()
                quality = compute_quality_metrics(raw_text)
                
                pages.append(PageText(
                    page_num=i + 1,
                    raw_text=raw_text,
                    quality_metrics=quality,
                    extraction_method="pymupdf",
                    metadata={
                        "width": page.rect.width,
                        "height": page.rect.height
                    }
                ))
            except Exception as e:
                logger.warning(f"Failed to extract page {i+1} with PyMuPDF: {e}")
                pages.append(PageText(
                    page_num=i + 1,
                    raw_text="",
                    quality_metrics=QualityMetrics(is_acceptable=False),
                    extraction_method="failed",
                    metadata={"error": str(e)}
                ))
        
        doc.close()
        logger.info(f"Extracted {len(pages)} pages with PyMuPDF")
        return pages
    
    except ImportError:
        raise RuntimeError("No PDF library available (need pdfplumber or PyMuPDF)")
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise


def extract_page_with_ocr(
    pdf_bytes: bytes,
    page_num: int,
    ocr_instance: Optional[Any] = None
) -> PageText:
    """
    Extract a single page using OCR.
    
    Args:
        pdf_bytes: PDF file bytes
        page_num: Page number (1-indexed)
        ocr_instance: Optional ImageOCR instance
    
    Returns:
        PageText with OCR-extracted text
    """
    try:
        import fitz
        from io import BytesIO
        from PIL import Image
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Invalid page_num {page_num} (PDF has {len(doc)} pages)")
        
        page = doc[page_num - 1]
        
        # Render page to image
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(BytesIO(img_bytes))
        
        # OCR the image
        if ocr_instance:
            ocr_text = ocr_instance.extract_text(img)
            extraction_method = "ocr_custom"
        else:
            # Try pytesseract
            try:
                import pytesseract
                ocr_text = pytesseract.image_to_string(img)
                extraction_method = "ocr_tesseract"
            except Exception:
                # Fallback to empty
                ocr_text = ""
                extraction_method = "ocr_failed"
        
        doc.close()
        
        quality = compute_quality_metrics(ocr_text)
        
        return PageText(
            page_num=page_num,
            raw_text=ocr_text,
            quality_metrics=quality,
            used_ocr=True,
            extraction_method=extraction_method,
            metadata={"dpi": 300}
        )
    
    except Exception as e:
        logger.error(f"OCR extraction failed for page {page_num}: {e}")
        return PageText(
            page_num=page_num,
            raw_text="",
            quality_metrics=QualityMetrics(is_acceptable=False),
            used_ocr=True,
            extraction_method="ocr_failed",
            metadata={"error": str(e)}
        )


def get_extraction_summary(pages: List[PageText]) -> Dict[str, Any]:
    """
    Get summary statistics for page extraction.
    
    Args:
        pages: List of PageText objects
    
    Returns:
        Summary dict with counts and metrics
    """
    total_pages = len(pages)
    pages_with_ocr = sum(1 for p in pages if p.used_ocr)
    pages_low_quality = sum(1 for p in pages if not p.quality_metrics.is_acceptable)
    total_chars = sum(len(p.raw_text) for p in pages)
    total_words = sum(p.quality_metrics.word_count for p in pages)
    
    extraction_methods = {}
    for page in pages:
        method = page.extraction_method
        extraction_methods[method] = extraction_methods.get(method, 0) + 1
    
    return {
        "total_pages": total_pages,
        "pages_with_ocr": pages_with_ocr,
        "pages_low_quality": pages_low_quality,
        "total_chars": total_chars,
        "total_words": total_words,
        "extraction_methods": extraction_methods,
        "avg_words_per_page": total_words / max(total_pages, 1),
        "avg_chars_per_page": total_chars / max(total_pages, 1)
    }
