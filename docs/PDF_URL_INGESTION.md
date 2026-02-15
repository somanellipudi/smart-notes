# PDF and URL Ingestion System - Complete Documentation

## Overview

This document describes the comprehensive PDF and URL ingestion system that has been integrated into Smart Notes. The system provides robust extraction of content from multiple sources with intelligent fallback strategies and quality validation.

---

## Key Features

### 1. **Robust PDF Extraction** (src/preprocessing/pdf_ingest.py)

#### Multi-Strategy Fallback
The PDF extraction system uses a 3-level fallback strategy to handle various PDF formats and corruption:

1. **PyMuPDF (fitz)** - Fast, modern PDF library with better support for complex PDFs
2. **pdfplumber** - Alternative extraction engine with table and layout awareness  
3. **OCR Fallback** - pdf2image + pytesseract for scanned or heavily corrupted PDFs

#### Quality Assessment
All extracted text is validated against multiple quality heuristics:
- **Minimum words**: 80 words required
- **Minimum letters**: 400 alphabetic characters required
- **Alphabetic ratio**: At least 30% alphabetic characters (prevents garbage text and CID glyphs)

#### Usage
```python
from src.preprocessing.pdf_ingest import extract_pdf_text

# Extract with automatic OCR fallback
text, metadata = extract_pdf_text(pdf_file, ocr=ocr_instance)

# Returns:
# - text: Extracted text content or empty string if fails quality check
# - metadata: {
#     "extraction_method": "pymupdf|pdfplumber|ocr",
#     "pages": number_of_pages,
#     "quality_metrics": {...},
#     "error": "error message if applicable"
#   }
```

---

### 2. **Intelligent URL Ingestion** (src/preprocessing/url_ingest.py)

#### YouTube Support
- Fetches video transcripts using youtube-transcript-api
- Supports standard YouTube URLs, short URLs (youtu.be), and timestamped links
- Graceable handles disabled captions with fallback

#### Article Extraction
Multi-strategy article extraction with progressive fallback:
1. **trafilatura** - Specialized for news/article extraction (preferred)
2. **readability-lxml** - General content extraction with layout awareness
3. **BeautifulSoup** - Basic HTML parsing fallback

Features:
- Extracts main content with automatic noise removal
- Preserves article title and metadata
- 10-second timeout to prevent hangs
- User-agent spoofing for blocked sites

#### Usage
```python
from src.preprocessing.url_ingest import fetch_url_text

# Fetch and extract content
text, metadata = fetch_url_text(url)

# Returns:
# - text: Extracted content or empty string if fails
# - metadata: {
#     "source_type": "youtube|article",
#     "title": "Article title or video name",
#     "extraction_method": "youtube_api|trafilatura|readability|beautifulsoup|error",
#     "word_count": number_of_words,
#     "video_id": "video_id" (for YouTube),
#     "error": "error message if applicable"
#   }
```

---

## Integration Points

### 1. **Streamlit UI (app.py)**

#### File Upload
PDF files uploaded via the "Upload Images & PDFs" section are now processed with the new robust extraction:
```
✓ PDF extraction complete: [character count] chars, ~[word count] words
```

The extraction method used is logged and includes fallback information.

#### URL Input
URL support is available via the "🌐 URL Sources (Beta)" expander:
- Enter YouTube URLs or web article URLs (one per line)
- URLs are automatically validated and ingested
- Accessible only when `config.ENABLE_URL_SOURCES = True`

### 2. **Verifiable Pipeline** (src/reasoning/verifiable_pipeline.py)

The pipeline receives URLs and:
1. Ingests content from each URL using existing `ingest_urls()` function from `src/retrieval/url_ingest.py`
2. Chunks extracted content for evidence store indexing
3. Uses content for claim verification and evidence matching

### 3. **Evidence Store** (src/retrieval/evidence_store.py)

Graceful validation fallback:
- **Before**: Hard-fail with RuntimeError for insufficient input
- **After**: Falls back to baseline mode with explanatory metadata
- Returns `INSUFFICIENT_EVIDENCE` status instead of crashing
- Provides feedback on character count vs. requirements

---

## Configuration

### Required Settings (config.py)

```python
# PDF/URL Processing
ENABLE_OCR_FALLBACK = True          # Enable OCR when PDF extraction fails
ENABLE_URL_SOURCES = True           # Enable URL ingestion feature

# Input Validation Thresholds
MIN_INPUT_CHARS_ABSOLUTE = 100      # Hard minimum for any input
MIN_INPUT_CHARS_FOR_VERIFICATION = 500  # Minimum for verifiable mode

# PDF Quality Thresholds
PDF_QUALITY_MIN_WORDS = 80          # Minimum words extracted from PDF
PDF_QUALITY_MIN_LETTERS = 400       # Minimum alphabetic characters
PDF_QUALITY_MIN_ALPHA_RATIO = 0.30  # Minimum alphabetic character ratio
```

### Optional Settings

```python
# OCR Configuration (if using Tesseract)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# TESSERACT_CMD = "/usr/bin/tesseract"  # Linux

# LLM Correction for OCR Output
CORRECT_OCR_WITH_LLM = True  # Use LLM to fix OCR errors
```

---

## Error Handling

### PDF Extraction Errors

1. **Empty PDF**
   - Returns empty string and error metadata
   - Falls back to baseline mode if in verification pipeline

2. **Corrupted PDF (CID glyphs)**
   - Quality assessment rejects text with <30% alphabetic characters
   - Automatically triggers OCR fallback
   - If OCR also fails, returns empty with error metadata

3. **OCR Failure**
   - When available, uses existing ImageOCR instance
   - If ImageOCR not available, logs warning and returns empty

### URL Ingestion Errors

1. **Network Error**
   - Timeout after 10 seconds
   - Returns empty text with error metadata
   - Pipeline continues with other URLs

2. **YouTube Captions Disabled**
   - Returns empty text with "captions disabled" error
   - No fallback available (transcripts are YouTube-specific)

3. **Invalid/Malformed URL**
   - Validated before attempting fetch
   - Returns error metadata with validation reason

---

## Quality Validation

### PDF Quality Heuristics

The system detects and rejects:

- **CID Glyphs**: Non-Unicode placeholders (cid:123) in corrupted PDFs
- **Excessive Symbols**: Text with <30% alphabetic characters
- **Insufficient Length**: Extracted text <80 words or <400 letters

Example of rejected low-quality extraction:
```
"(cid:1) (cid:2) (cid:3) (cid:4) 123 ###@@@ $$$%%% ..."
```

### Text Cleaning

Automatic processing includes:
- Removal of CID glyphs and unicode control characters
- Normalization of whitespace (multiple spaces → single space)
- Trimming of empty lines
- Preservation of actual content

---

## Performance Considerations

### Timeouts and Limits

- **URL Fetch Timeout**: 10 seconds per URL
- **HTTP Download Limit**: 2 MB maximum
- **PDF Processing**: Varies by strategy (PyMuPDF fastest, OCR slowest)

### Optimization Tips

1. **For Educational Content**
   - PDFs are typically well-structured; PyMuPDF handles most cases
   - OCR fallback adds ~2-5 seconds per page for scanned PDFs

2. **For URLs**
   - YouTube transcripts fetch instantly (no video streaming)
   - Article extraction depends on page complexity (typically <2s)

3. **For Better Results**
   - Avoid scanned PDFs when original digital files available
   - Prefer direct YouTube URLs over article links mentioning videos

---

## Testing

### Unit Tests
Located in `tests/test_pdf_url_ingest.py`:
- Quality assessment heuristics
- Text cleaning and CID glyph removal
- YouTube URL identification and video ID extraction
- Article extraction with mocked HTTP
- Error handling for network failures

### Integration Tests
Located in `tests/test_integration_pdf_url.py`:
- End-to-end PDF extraction with mocked file operations
- Complete URL ingestion pipeline
- Quality threshold enforcement
- Error handling in data flow

### Running Tests

```bash
# Run all PDF/URL tests
pytest tests/test_pdf_url_ingest.py tests/test_integration_pdf_url.py -v

# Run specific test class
pytest tests/test_pdf_url_ingest.py::TestPDFIngestQuality -v

# Run with coverage
pytest tests/test_pdf_url_ingest.py --cov=src.preprocessing
```

---

## Known Limitations

### PDF Extraction

1. **Scanned PDFs Performance**
   - OCR fallback is slow for large scanned documents (>50 pages)
   - Recommend extracting single chapters or sections

2. **Multilingual PDFs**
   - PDF text extraction works well for English
   - OCR supports multiple languages but requires language data
   - Non-Latin scripts may have lower quality

3. **Complex Layouts**
   - Tabular content may not extract perfectly from PDFs
   - Equations/formulas may extract as raw unicode instead of LaTeX

### URL Ingestion

1. **Protected Content**
   - Paywalled articles cannot be accessed
   - Bot-protected sites may reject requests

2. **Dynamic Content**
   - JavaScript-rendered content not supported
   - Extraction only works on server-rendered HTML

3. **Large Documents**
   - Downloads are limited to 2 MB
   - Very large PDFs linked from URLs may be truncated

---

## Migration from Old System

### What Changed

**Old PDF System**:
```python
from PyPDF2 import PdfReader
reader = PdfReader(file)
text = "".join(page.extract_text() for page in reader.pages)
# Returns garbage text for corrupted PDFs, no quality validation
```

**New PDF System**:
```python
from src.preprocessing.pdf_ingest import extract_pdf_text
text, metadata = extract_pdf_text(file, ocr=ocr_instance)
# Returns clean text with quality validation or empty string if corrupted
```

### Existing Code Compatibility

- **New code uses new PDF extraction**: Better quality, automatic OCR fallback
- **Graceful degradation**: If insufficient input, falls back to baseline instead of crashing
- **URL support**: New feature, previously unsupported

---

## Future Enhancements

### Planned Improvements

1. **PDF Extraction**
   - Add support for extracting metadata (author, creation date)
   - Better equation/formula preservation
   - Language detection for multilingual PDFs

2. **URL Ingestion**
   - Support for academic repository URLs (arXiv, ResearchGate)
   - Podcast transcript extraction
   - Video platform support beyond YouTube (Vimeo, etc.)

3. **Performance**
   - Parallel PDF page extraction
   - Caching of URL content to avoid re-fetching
   - GPU acceleration for OCR

---

## Troubleshooting

### PDF Extraction Returns Empty String

**Symptom**: "PDF extraction complete: 0 characters"

**Causes**:
1. PDF is corrupted or heavily scanned
2. OCR is disabled (`ENABLE_OCR_FALLBACK = False`)
3. Content doesn't meet quality thresholds

**Solution**: 
1. Check PDF with external reader (verify it's readable)
2. Enable OCR if available: `ENABLE_OCR_FALLBACK = True`
3. Lower quality thresholds if needed (but validates content first)

### URL Ingestion Times Out

**Symptom**: "Connection timeout" or "Failed to fetch URL"

**Cause**: Website is slow or blocking requests

**Solution**:
1. Try again (may be temporary network issue)
2. Check URL validity in browser first
3. Consider providing alternative URLs

### Quality Assessment Too Strict

**Symptom**: Good PDFs rejected as low quality

**Solution**:
1. Adjust thresholds in `config.py`:
   - Increase `PDF_QUALITY_MIN_LETTERS` if consistently too high
   - Lower `PDF_QUALITY_MIN_ALPHA_RATIO` for non-English content

---

## References

### Libraries Used

- **PDF Extraction**: PyMuPDF (fitz), pdfplumber
- **OCR**: pdf2image, pytesseract
- **URL Processing**: youtube-transcript-api, trafilatura, readability-lxml, requests
- **HTML Parsing**: BeautifulSoup4

### Configuration Reference

See [config.py](../config.py) for all available settings

---

## Support

For issues or questions about PDF/URL ingestion:
1. Check the Troubleshooting section above
2. Review test files for usage examples
3. Check application logs for detailed error messages
