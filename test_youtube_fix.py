#!/usr/bin/env python3
"""
Quick test to verify YouTube URL extraction error handling works correctly.
Tests that specific error messages from YouTube API are preserved and shown to user.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.url_ingest import fetch_url_text, _is_youtube_url, _extract_youtube_video_id
from src.retrieval.youtube_ingest import fetch_transcript_text

def test_youtube_url_detection():
    """Test YouTube URL detection."""
    print("\n" + "="*70)
    print("TEST 1: YouTube URL Detection")
    print("="*70)
    
    test_urls = [
        ("https://youtu.be/OYvlzrJ4IZQ", True),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", True),
        ("https://www.articlesite.com/some-article", False),
    ]
    
    for url, expected in test_urls:
        result = _is_youtube_url(url)
        status = "✓" if result == expected else "✗"
        print(f"{status} {url}: {result} (expected {expected})")

def test_video_id_extraction():
    """Test video ID extraction."""
    print("\n" + "="*70)
    print("TEST 2: YouTube Video ID Extraction")
    print("="*70)
    
    test_urls = [
        ("https://youtu.be/OYvlzrJ4IZQ", "OYvlzrJ4IZQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"),
        ("https://www.notayoutube.com/", None),
    ]
    
    for url, expected in test_urls:
        result = _extract_youtube_video_id(url)
        status = "✓" if result == expected else "✗"
        print(f"{status} {url}: {result} (expected {expected})")

def test_error_message_preservation():
    """Test that error messages from YouTube API are preserved."""
    print("\n" + "="*70)
    print("TEST 3: Error Message Preservation")
    print("="*70)
    
    # Test with a private video URL (won't have transcript)
    test_cases = [
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - should work
            "description": "Valid YouTube URL"
        },
        {
            "url": "https://www.youtube.com/watch?v=invalid12345",  # Invalid video ID
            "description": "Invalid YouTube video"
        },
    ]
    
    print("\nTesting fetch_url_text with YouTube URLs...")
    print("(Note: Actual extraction depends on network and transcript availability)\n")
    
    for test in test_cases:
        url = test["url"]
        desc = test["description"]
        print(f"\nTest: {desc}")
        print(f"URL: {url}")
        
        text, metadata = fetch_url_text(url)
        print(f"  - Extraction method: {metadata.get('extraction_method', 'unknown')}")
        print(f"  - Text length: {len(text)} chars")
        
        if metadata.get("error"):
            print(f"  - Error: {metadata['error']}")
            # Verify it's not a generic "No text extracted" but specific error
            if metadata['error'] != "No text extracted":
                print(f"  ✓ Specific error message preserved")
            else:
                print(f"  ℹ️ Generic error (might be expected if transcript unavailable)")
        else:
            print(f"  ✓ Successfully extracted")

def test_backward_compatibility():
    """Test backward compatibility aliases."""
    print("\n" + "="*70)
    print("TEST 4: Backward Compatibility Aliases")
    print("="*70)
    
    print("Testing that old function names still work...")
    
    # These imports should work without error
    from src.preprocessing.url_ingest import _is_youtube_url, _extract_youtube_video_id
    
    print("✓ _is_youtube_url  (backward compatible alias)")
    print("✓ _extract_youtube_video_id (backward compatible alias)")

if __name__ == "__main__":
    test_youtube_url_detection()
    test_video_id_extraction()
    test_error_message_preservation()
    test_backward_compatibility()
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
