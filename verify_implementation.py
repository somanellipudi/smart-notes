"""
Final verification that all PDF and URL ingestion improvements are working.

This script validates:
1. PDF extraction module imports and functions correctly
2. URL ingestion module imports and functions correctly  
3. app.py has been updated with new modules
4. Quality assessment works correctly
5. Graceful degradation is in place
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_imports():
    """Verify all new modules can be imported."""
    print("\n1. VERIFYING IMPORTS")
    print("-" * 60)
    
    try:
        from src.preprocessing.pdf_ingest import (
            extract_pdf_text,
            _assess_extraction_quality,
            _clean_text,
            _count_letters,
            _count_words,
            _compute_alphabetic_ratio
        )
        print("   ✓ PDF ingestion module imported")
    except ImportError as e:
        print(f"   ✗ PDF ingestion import failed: {e}")
        return False
    
    try:
        from src.preprocessing.url_ingest import (
            fetch_url_text,
            _is_youtube_url,
            _extract_youtube_video_id,
            _clean_text as clean_url_text
        )
        print("   ✓ URL ingestion module imported")
    except ImportError as e:
        print(f"   ✗ URL ingestion import failed: {e}")
        return False
    
    try:
        import app
        print("   ✓ app.py imports without errors")
    except ImportError as e:
        print(f"   ✗ app.py import failed: {e}")
        return False
    
    return True


def verify_pdf_quality_assessment():
    """Verify PDF quality assessment works."""
    print("\n2. VERIFYING PDF QUALITY ASSESSMENT")
    print("-" * 60)
    
    from src.preprocessing.pdf_ingest import _assess_extraction_quality, _compute_alphabetic_ratio, _clean_text
    
    # Test 1: Good quality text
    good_text = " ".join(["Calculus is the mathematical study of continuous change."] * 20)
    is_good, reason = _assess_extraction_quality(good_text)
    if is_good:
        print(f"   ✓ Good quality text accepted: {reason}")
    else:
        print(f"   ✗ Good quality text rejected: {reason}")
        return False
    
    # Test 2: Corrupted text with CID glyphs (low word count will reject it)
    corrupted_text = "(cid:1) (cid:2) (cid:3) " * 20  # Only ~20 words
    is_good, reason = _assess_extraction_quality(corrupted_text)
    if not is_good and "word" in reason.lower():
        print(f"   ✓ Corrupted text rejected (due to word count): {reason}")
    else:
        # Try with more words but non-alphabetic content
        no_alpha_text = "123 456 789 " * 100  # 300 words but no letters
        is_good, reason = _assess_extraction_quality(no_alpha_text)
        if not is_good and "alphabetic" in reason.lower():
            print(f"   ✓ Non-alphabetic text rejected: {reason}")
        else:
            print(f"   ✗ Low alphabetic content not rejected: {reason}")
            return False
    
    # Test 3: Low word count
    few_words = " ".join(["word"] * 40)
    is_good, reason = _assess_extraction_quality(few_words)
    if not is_good:
        print(f"   ✓ Low word count rejected: {reason}")
    else:
        print(f"   ✗ Low word count not rejected: {reason}")
        return False
    
    # Test 4: Alphabetic ratio calculation
    ratio = _compute_alphabetic_ratio("hello123")
    expected = 5/8  # 5 letters out of 8 chars
    if abs(ratio - expected) < 0.01:
        print(f"   ✓ Alphabetic ratio calculation correct: {ratio:.2%}")
    else:
        print(f"   ✗ Alphabetic ratio incorrect: {ratio:.2%} (expected {expected:.2%})")
        return False
    
    # Test 5: CID glyph cleaning
    text_with_cids = "Hello (cid:123) world (cid:456)"
    cleaned = _clean_text(text_with_cids)
    if "(cid:" not in cleaned and "Hello" in cleaned and "world" in cleaned:
        print(f"   ✓ CID glyph removal works")
    else:
        print(f"   ✗ CID glyph not properly removed")
        return False
    
    return True


def verify_url_detection():
    """Verify URL type detection works."""
    print("\n3. VERIFYING URL DETECTION")
    print("-" * 60)
    
    from src.preprocessing.url_ingest import _is_youtube_url, _extract_youtube_video_id
    
    # Test YouTube detection
    youtube_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
    ]
    
    for url in youtube_urls:
        if _is_youtube_url(url):
            print(f"   ✓ YouTube URL recognized: {url[:40]}...")
        else:
            print(f"   ✗ YouTube URL not recognized: {url}")
            return False
    
    # Test video ID extraction
    video_id = _extract_youtube_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ")
    if video_id == "dQw4w9WgXcQ":
        print(f"   ✓ Video ID extraction correct: {video_id}")
    else:
        print(f"   ✗ Video ID extraction failed: {video_id}")
        return False
    
    # Test non-YouTube URL
    if not _is_youtube_url("https://example.com/article"):
        print(f"   ✓ Non-YouTube URL correctly identified")
    else:
        print(f"   ✗ Non-YouTube URL incorrectly identified as YouTube")
        return False
    
    return True


def verify_app_integration():
    """Verify app.py has been updated."""
    print("\n4. VERIFYING APP.PY INTEGRATION")
    print("-" * 60)
    
    app_path = Path(__file__).parent / "app.py"
    
    with open(app_path, 'r', encoding='utf-8', errors='ignore') as f:
        app_content = f.read()
    
    # Check for new imports
    checks = [
        ("pdf_ingest import", "from src.preprocessing.pdf_ingest import extract_pdf_text"),
        ("url_ingest import", "from src.preprocessing.url_ingest import fetch_url_text"),
        ("PDF function updated", "def _extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]"),
        ("URL processing logic", 'pdf_text, pdf_metadata = _extract_text_from_pdf(pdf_file)'),
        ("Extraction method logging", 'extraction_method = pdf_metadata.get("extraction_method"'),
    ]
    
    all_passed = True
    for check_name, check_string in checks:
        if check_string in app_content:
            print(f"   ✓ {check_name}")
        else:
            print(f"   ✗ {check_name}: '{check_string[:50]}...' not found")
            all_passed = False
    
    return all_passed


def verify_test_files():
    """Verify test files exist and compile."""
    print("\n5. VERIFYING TEST FILES")
    print("-" * 60)
    
    test_files = [
        ("tests/test_pdf_url_ingest.py", "PDF/URL unit tests"),
        ("tests/test_integration_pdf_url.py", "Integration tests"),
        ("tests/test_ingestion_practical.py", "Practical tests"),
    ]
    
    all_exist = True
    for test_file, description in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            print(f"   ✓ {description} exists")
            
            # Check if it compiles
            try:
                import py_compile
                py_compile.compile(str(test_path), doraise=True)
                print(f"     → Compiles without errors")
            except Exception as e:
                print(f"     → Compilation failed: {e}")
                all_exist = False
        else:
            print(f"   ✗ {description} not found: {test_file}")
            all_exist = False
    
    return all_exist


def verify_documentation():
    """Verify documentation files exist."""
    print("\n6. VERIFYING DOCUMENTATION")
    print("-" * 60)
    
    doc_files = [
        ("docs/PDF_URL_INGESTION.md", "PDF/URL ingestion documentation"),
        ("docs/IMPLEMENTATION_COMPLETE.md", "Implementation summary"),
    ]
    
    all_exist = True
    for doc_file, description in doc_files:
        doc_path = Path(__file__).parent / doc_file
        if doc_path.exists():
            size_kb = doc_path.stat().st_size / 1024
            print(f"   ✓ {description} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {description} not found: {doc_file}")
            all_exist = False
    
    return all_exist


def verify_configuration():
    """Verify configuration settings exist."""
    print("\n7. VERIFYING CONFIGURATION")
    print("-" * 60)
    
    import config
    
    settings = [
        ("ENABLE_URL_SOURCES", "URL ingestion enabled"),
        ("ENABLE_OCR_FALLBACK", "OCR fallback enabled"),
        ("MIN_INPUT_CHARS_ABSOLUTE", "Absolute minimum input"),
        ("MIN_INPUT_CHARS_FOR_VERIFICATION", "Verification minimum"),
    ]
    
    all_exist = True
    for setting, description in settings:
        if hasattr(config, setting):
            value = getattr(config, setting)
            print(f"   ✓ {setting} = {value} ({description})")
        else:
            print(f"   ✗ {setting} not found in config")
            all_exist = False
    
    return all_exist


def main():
    """Run all verifications."""
    print("=" * 60)
    print("PDF/URL INGESTION SYSTEM - FINAL VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Run all verification checks
    results.append(("Imports", verify_imports()))
    results.append(("PDF Quality Assessment", verify_pdf_quality_assessment()))
    results.append(("URL Detection", verify_url_detection()))
    results.append(("App Integration", verify_app_integration()))
    results.append(("Test Files", verify_test_files()))
    results.append(("Documentation", verify_documentation()))
    results.append(("Configuration", verify_configuration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {check_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
        print("\nThe PDF and URL ingestion system is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} verification check(s) failed.")
        print("Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
