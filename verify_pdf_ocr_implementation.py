#!/usr/bin/env python3
"""
Quick verification script for PDF OCR fallback implementation.

Validates:
1. All imports work
2. Quality heuristics function correctly
3. PDF extraction module is wired into app.py
4. Requirements include necessary dependencies
"""

import sys
from pathlib import Path

def check_imports():
    """Verify all necessary imports are available."""
    print("\n[1] Checking imports...")
    
    try:
        from src.preprocessing.pdf_ingest import extract_pdf_text
        print("  ✓ extract_pdf_text imported")
    except ImportError as e:
        print(f"  ✗ Failed to import extract_pdf_text: {e}")
        return False
    
    try:
        import fitz
        print("  ✓ PyMuPDF (fitz) available")
    except ImportError:
        print("  ⚠ PyMuPDF not installed (optional, will use pdfplumber)")
    
    try:
        import pdfplumber
        print("  ✓ pdfplumber available")
    except ImportError:
        print("  ⚠ pdfplumber not installed (optional fallback)")
    
    try:
        from pdf2image import convert_from_bytes
        print("  ✓ pdf2image available (for OCR)")
    except ImportError:
        print("  ⚠ pdf2image not installed (optional OCR fallback)")
    
    try:
        from src.audio.image_ocr import ImageOCR
        print("  ✓ ImageOCR available")
    except ImportError as e:
        print(f"  ✗ Failed to import ImageOCR: {e}")
        return False
    
    return True


def check_quality_heuristics():
    """Verify quality heuristics work correctly."""
    print("\n[2] Testing quality heuristics...")
    
    from src.preprocessing.pdf_ingest import (
        _count_letters,
        _count_words,
        _compute_alphabetic_ratio,
        _assess_extraction_quality
    )
    
    # Test 1: Count letters
    assert _count_letters("Hello") == 5
    assert _count_letters("Hello123") == 5
    assert _count_letters("123") == 0
    print("  ✓ Letter counting works")
    
    # Test 2: Count words
    assert _count_words("hello world") == 2
    assert _count_words("") == 0
    print("  ✓ Word counting works")
    
    # Test 3: Alphabetic ratio
    assert _compute_alphabetic_ratio("hello") == 1.0
    assert abs(_compute_alphabetic_ratio("hello123") - 5/8) < 0.01
    assert _compute_alphabetic_ratio("") == 0.0
    print("  ✓ Alphabetic ratio calculation works")
    
    # Test 4: Quality assessment - good text
    good_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
    is_good, reason = _assess_extraction_quality(good_text)
    assert is_good, f"Good text rejected: {reason}"
    print("  ✓ Good quality text accepted")
    
    # Test 5: Quality assessment - garbage/numeric text
    # CID glyphs contain letters, so use pure numeric text instead
    garbage_text = "123 456 789 " * 50
    is_good, reason = _assess_extraction_quality(garbage_text)
    assert not is_good, f"Garbage text not rejected: {reason}"
    print("  ✓ Garbage text (numeric only) rejected")
    
    # Test 6: Quality assessment - few words
    few_words = " ".join(["word"] * 40)
    is_good, reason = _assess_extraction_quality(few_words)
    assert not is_good, "Low word count not rejected"
    print("  ✓ Low word count rejected")
    
    return True


def check_app_integration():
    """Verify app.py has been properly updated."""
    print("\n[3] Checking app.py integration...")
    
    app_path = Path(__file__).parent / "app.py"
    
    with open(app_path, 'r', encoding='utf-8', errors='ignore') as f:
        app_content = f.read()
    
    # Check for import
    if "from src.preprocessing.pdf_ingest import extract_pdf_text" in app_content:
        print("  ✓ extract_pdf_text imported in app.py")
    else:
        print("  ✗ extract_pdf_text import not found in app.py")
        return False
    
    # Check for OCR parameter in function
    if "def _extract_text_from_pdf(pdf_file, ocr=None)" in app_content:
        print("  ✓ _extract_text_from_pdf accepts ocr parameter")
    else:
        print("  ✗ _extract_text_from_pdf doesn't have ocr parameter")
        return False
    
    # Check for OCR instance initialization
    if "ocr_instance = initialize_ocr()" in app_content:
        print("  ✓ OCR initialized before PDF processing")
    else:
        print("  ⚠ OCR not explicitly initialized (may be implicit)")
    
    # Check for passing OCR to extraction function
    if "_extract_text_from_pdf(pdf_file, ocr=ocr_instance)" in app_content:
        print("  ✓ OCR passed to extraction function")
    else:
        print("  ✗ OCR not passed to extraction function")
        return False
    
    return True


def check_requirements():
    """Verify requirements.txt includes necessary packages."""
    print("\n[4] Checking requirements.txt...")
    
    req_path = Path(__file__).parent / "requirements.txt"
    
    with open(req_path, 'r') as f:
        req_content = f.read()
    
    required_packages = [
        ("PyMuPDF", "PyMuPDF"),
        ("pdfplumber", "pdfplumber"),
        ("pdf2image", "pdf2image"),
    ]
    
    for package_name, pattern in required_packages:
        if pattern in req_content:
            print(f"  ✓ {package_name} in requirements.txt")
        else:
            print(f"  ⚠ {package_name} not in requirements.txt (optional)")
    
    return True


def check_test_file():
    """Verify test file exists and compiles."""
    print("\n[5] Checking test file...")
    
    test_path = Path(__file__).parent / "tests" / "test_pdf_ocr_fallback.py"
    
    if test_path.exists():
        print(f"  ✓ Test file exists: {test_path.name}")
        
        # Try to compile it
        try:
            import py_compile
            py_compile.compile(str(test_path), doraise=True)
            print("  ✓ Test file compiles without errors")
        except Exception as e:
            print(f"  ✗ Test file compilation failed: {e}")
            return False
    else:
        print(f"  ✗ Test file not found: {test_path}")
        return False
    
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("PDF OCR FALLBACK IMPLEMENTATION - VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Quality Heuristics", check_quality_heuristics),
        ("App Integration", check_app_integration),
        ("Requirements", check_requirements),
        ("Test File", check_test_file),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ✗ Check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {check_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Run: pip install -r requirements.txt")
        print("2. Test with: pytest tests/test_pdf_ocr_fallback.py -v")
        print("3. Test in app: streamlit run app.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Review errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
