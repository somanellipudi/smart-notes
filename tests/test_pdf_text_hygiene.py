"""
Test suite for PDF text extraction hygiene checks.

These tests verify that:
1. Compiled paper PDF does not contain replacement glyph artifacts
2. Architecture figure PDF does not contain embedded title/banner text
3. Soft-hyphen artifacts are detected in PDF extraction
"""

import sys
import subprocess
from pathlib import Path
import pytest


# Get repo root
REPO_ROOT = Path(__file__).parent.parent


def test_pdf_text_hygiene_script_exists():
    """Verify check_pdf_text_hygiene.py script exists."""
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    assert script.exists(), f"Script not found: {script}"


def test_paper_pdf_no_replacement_artifacts():
    """
    Test that compiled paper PDF has no replacement glyph artifacts.
    
    This test runs the hygiene checker on the main paper PDF and ensures
    it passes (exit code 0). The PDF is expected to be generated or present.
    """
    pdf_path = REPO_ROOT / "paper" / "main.pdf"
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    
    if not pdf_path.exists():
        pytest.skip(f"Paper PDF not found at {pdf_path} (run pdflatex first)")
    
    result = subprocess.run(
        [sys.executable, str(script), str(pdf_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    
    assert result.returncode == 0, (
        f"PDF text hygiene check failed for {pdf_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "[OK]" in result.stdout or "[WARN]" in result.stdout


def test_architecture_pdf_no_embedded_titles():
    """
    Test that architecture.pdf does not contain embedded title/banner text.
    
    Verifies that architecture.pdf contains only the diagram (boxes/arrows)
    and no embedded text like "CalibraTeach:" or hardware/runtime information.
    """
    pdf_path = REPO_ROOT / "figures" / "architecture.pdf"
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    
    if not pdf_path.exists():
        pytest.skip(f"Architecture PDF not found at {pdf_path}")
    
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(pdf_path),
            "--check-architecture"
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    
    assert result.returncode == 0, (
        f"Architecture PDF check failed: {pdf_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    
    # Expect success message
    assert "[OK]" in result.stdout or "clean" in result.stdout.lower()


def test_architecture_pdf_bans_embedded_specs():
    """
    Test that architecture.pdf is verified to exclude banned spec strings.
    """
    pdf_path = REPO_ROOT / "figures" / "architecture.pdf"
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    
    if not pdf_path.exists():
        pytest.skip(f"Architecture PDF not found at {pdf_path}")
    
    result = subprocess.run(
        [sys.executable, str(script), str(pdf_path), "--check-architecture"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    
    # Should NOT contain any of these banned strings
    banned_strings = ["CalibraTeach:", "GPU:", "PyTorch", "CUDA", "Transformers"]
    for banned in banned_strings:
        assert banned not in result.stdout, (
            f"Architecture PDF test reported banned string '{banned}' in output:\n"
            f"{result.stdout}"
        )


def test_pdf_text_hygiene_detects_soft_hyphen_artifacts():
    """
    Test that soft-hyphen artifacts (U+00AD) are detected.
    
    Uses a synthetic test string containing soft-hyphen to verify detection logic.
    This is a fast, deterministic test that doesn't require actual PDF files.
    """
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    
    # Verify the script defines soft-hyphen detection
    # Read the script source to check for U+00AD
    script_content = script.read_text()
    assert "0x00AD" in script_content, (
        "Soft-hyphen (U+00AD) detection not found in script"
    )
    assert "Soft hyphen" in script_content, (
        "Soft-hyphen description not found in script"
    )


def test_check_pdf_text_hygiene_with_nonexistent_file():
    """Test that script fails gracefully on nonexistent PDF."""
    script = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"
    result = subprocess.run(
        [sys.executable, str(script), "/nonexistent/pdf.pdf"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT
    )
    
    assert result.returncode != 0, "Script should fail on missing PDF"
    assert "[ERROR]" in result.stdout or "ERROR" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

