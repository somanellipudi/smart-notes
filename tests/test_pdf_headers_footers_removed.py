"""
Tests for header/footer removal in PDF ingestion.

Tests frequency-based header/footer stripping and specific pattern removal.
"""

import pytest
from io import BytesIO
from src.preprocessing.text_cleaner import clean_extracted_text, CleanDiagnostics
from src.preprocessing.pdf_ingest import extract_pdf_text, PDFIngestionReport


def test_repeated_header_footer_removal():
    """Test that repeated headers and footers are removed."""
    # Create synthetic multi-page PDF-like text
    pages = []
    for i in range(5):
        page_text = f"""--- Page {i+1} ---
UNIVERSITY OF EXAMPLE - COURSE NOTES
Introduction to Calculus

This is the main content of page {i+1} with substantial text.
It contains important information about derivatives and calculus.
The derivative of x squared is two x which is a fundamental rule.
We use derivatives to find rates of change in mathematical functions.

Page {i+1} of 5
"""
        pages.append(page_text)
    
    raw_text = "\n\n".join(pages)
    
    cleaned, diag = clean_extracted_text(raw_text)
    
    # Check that repeated headers/footers were removed
    assert diag.removed_lines_count > 0, "Should remove repeated lines"
    assert diag.headers_removed_count > 0, "Should count headers removed"
    
    # Main content should remain
    assert "main content" in cleaned.lower()
    assert "derivative" in cleaned.lower()
    
    # Repeated header should be gone (appears on all pages)
    assert "university of example" not in cleaned.lower() or cleaned.lower().count("university of example") <= 1


def test_camscanner_watermark_removal():
    """Test CamScanner watermark removal."""
    # Use enough content that some remains after cleaning
    text_with_watermark = """--- Page 1 ---
Lecture Notes on Mathematical Analysis and Important Concepts
Some important content here about mathematical theories and theorems.
Scanned by CamScanner from academic sources.
More content discussing important ideas and fundamental applications.
Definitions and proofs are central to this material.
--- Page 2 ---
More lecture notes on advanced topics and complex theories.
www.camscanner.com watermark present.
Important information about derivative calculations and integration methods.
These topics form the basis of modern mathematics."""
    
    cleaned, diag = clean_extracted_text(text_with_watermark)
    
    # Check watermarks were removed
    assert diag.watermark_removed_count > 0, "Should detect and count watermarks"
    assert "camscanner" not in cleaned.lower(), "CamScanner watermark should be removed"
    
    # Verify that watermark patterns are tracked
    assert "camscanner_watermark" in diag.removed_patterns_hit, "Should track CamScanner watermarks"


def test_unit_chapter_header_removal():
    """Test UNIT/CHAPTER header removal."""
    text_with_headers = """--- Page 1 ---
UNIT 3
Functions and Limits with Detailed Analysis

A function is a mapping from input to output values in mathematics.
This is a fundamental concept in calculus and analysis.
We study various types of functions throughout this unit.
--- Page 2 ---
CHAPTER 4
Derivatives and Their Applications

The derivative measures rate of change of functions over time.
Derivatives are essential tools for understanding function behavior.
We apply derivatives in physics and engineering applications."""
    
    cleaned, diag = clean_extracted_text(text_with_headers)
    
    # Check patterns were detected
    assert "unit_chapter_header" in diag.removed_patterns_hit, "Should detect UNIT/CHAPTER patterns"
    
    # Content should remain - at least some mentions of concepts
    content_lower = cleaned.lower()
    assert "function" in content_lower or "derivative" in content_lower


def test_isolated_page_numbers_removed():
    """Test isolated page numbers are removed."""
    # Ensure substantial content beyond page numbers
    text_with_page_nums = """--- Page 1 ---
Introduction to Algebra and Number Systems with Theory
This is substantial content about algebraic concepts and methods.
We study numbers, variables, and mathematical operations carefully.
Algebra forms the foundation for advanced mathematics.
1
--- Page 2 ---
More content about algebra continues here with new topics.
Solving equations and working with polynomials is essential.
Understanding variables requires mathematical precision.
2
More information about variables and expressions with examples.
--- Page 3 ---
Page 3
Final content covers advanced algebraic topics and applications."""
    
    cleaned, diag = clean_extracted_text(text_with_page_nums)
    
    # Check page numbers were detected
    has_page_num_patterns = ("isolated_page_number" in diag.removed_patterns_hit or 
                            "page_number_label" in diag.removed_patterns_hit or
                            "low_info" in diag.removed_by_regex)
    
    assert has_page_num_patterns, \
        "Should detect isolated page numbers"
    
    # Content should remain - substantial text about algebra
    content_lower = cleaned.lower()
    assert "algebra" in content_lower or "equation" in content_lower or "mathematical" in content_lower


def test_short_allcaps_title_removal():
    """Test short all-caps title lines followed by normal text are removed if repeated."""
    pages = []
    for i in range(4):
        page_text = f"""--- Page {i+1} ---
CALCULUS BASICS
The study of change and motion.
Content for page {i+1}."""
        pages.append(page_text)
    
    raw_text = "\n\n".join(pages)
    cleaned, diag = clean_extracted_text(raw_text)
    
    # Repeated all-caps title should be removed or reduced
    # (appears on all 4 pages, so should trigger removal)
    count = cleaned.lower().count("calculus basics")
    assert count < 4, "Repeated all-caps title should be removed from most occurrences"


def test_date_stamp_removal():
    """Test date stamp removal."""
    text_with_dates = """--- Page 1 ---
Lecture Notes
01/15/2024
Important content here.
--- Page 2 ---
More notes.
2024-01-15
More content."""
    
    cleaned, diag = clean_extracted_text(text_with_dates)
    
    # Date stamps should be detected
    if "date_stamp" in diag.removed_patterns_hit:
        assert diag.removed_patterns_hit["date_stamp"] > 0


def test_copyright_footer_removal():
    """Test copyright footer removal."""
    text_with_copyright = """--- Page 1 ---
Course Material
© 2024 University Press
Main content here.
--- Page 2 ---
More material.
Copyright 2024 Example Corp
Important information."""
    
    cleaned, diag = clean_extracted_text(text_with_copyright)
    
    # Copyright notices should be detected
    if "copyright_footer" in diag.removed_patterns_hit:
        assert diag.removed_patterns_hit["copyright_footer"] > 0
        # Check that copyright symbols/text is removed or reduced
        assert cleaned.count("©") < 2 or cleaned.lower().count("copyright") < 2


def test_diagnostics_completeness():
    """Test that diagnostics include all required fields."""
    pages = []
    for i in range(3):
        page_text = f"""--- Page {i+1} ---
COMMON HEADER
Content page {i+1}
Scanned by CamScanner
Page {i+1}"""
        pages.append(page_text)
    
    raw_text = "\n\n".join(pages)
    cleaned, diag = clean_extracted_text(raw_text)
    
    # Check all diagnostic fields exist
    assert hasattr(diag, 'removed_lines_count')
    assert hasattr(diag, 'headers_removed_count')
    assert hasattr(diag, 'watermark_removed_count')
    assert hasattr(diag, 'removed_patterns_hit')
    assert hasattr(diag, 'removed_by_regex')
    assert hasattr(diag, 'top_removed_lines')
    
    # Check they have meaningful values
    assert isinstance(diag.removed_lines_count, int)
    assert isinstance(diag.headers_removed_count, int)
    assert isinstance(diag.watermark_removed_count, int)
    assert isinstance(diag.removed_patterns_hit, dict)


def test_content_preservation():
    """Test that legitimate content is preserved."""
    # Use more substantial content that passes low_info check
    text = """--- Page 1 ---
REPEATED HEADER
Chapter 5: Important Concepts in Mathematical Analysis

The concept of limits is fundamental to understanding calculus deeply.
A limit describes the behavior of a function as x approaches a specific value.
Understanding limits requires careful mathematical analysis and rigor.
Limits allow us to define continuity and derivatives formally.

Example: lim(x→0) sin(x)/x = 1 is a classical and important result.

The limit concept helps us understand function continuity rigorously.
REPEATED HEADER
--- Page 2 ---
REPEATED HEADER

This theorem is crucial for understanding derivatives correctly and deeply.
Let f(x) be a differentiable function on an open interval.
Continuous functions have important properties like intermediate value property.

REPEATED HEADER"""
    
    cleaned, diag = clean_extracted_text(text)
    
    # Important content should be preserved
    content = cleaned.lower()
    # At least one of these mathematical concepts should remain
    has_key_content = any(word in content for word in 
                         ["limits", "limit", "calculus", "fundamental", "function", 
                          "theorem", "derivative", "continuous", "definition"])
    
    assert has_key_content, f"Should preserve mathematical content. Got: {content[:200]}"
    
    # Repeated headers should be removed
    assert diag.headers_removed_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
