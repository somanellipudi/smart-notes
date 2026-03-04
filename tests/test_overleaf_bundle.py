"""
Tests for build_overleaf_bundle.py

Validates that the Overleaf bundle builder correctly validates paper assets
and creates a compliant ZIP archive for IEEE Access submission.
"""

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
import pytest


@pytest.mark.paper
def test_bundle_builder_help():
    """Test that --help flag works."""
    result = subprocess.run(
        ["python", "scripts/build_overleaf_bundle.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    assert result.returncode == 0
    assert "overleaf" in result.stdout.lower()
    assert "--out" in result.stdout
    assert "--paper_dir" in result.stdout


@pytest.mark.paper
def test_validate_only_mode():
    """Test that --validate-only runs without creating ZIP."""
    repo_root = Path(__file__).parent.parent
    
    result = subprocess.run(
        ["python", "scripts/build_overleaf_bundle.py", "--validate-only"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    
    # Should succeed if paper/ directory is valid
    assert result.returncode == 0, f"Validation failed: {result.stdout}\n{result.stderr}"
    assert "Validating paper assets" in result.stdout
    assert "validate-only mode" in result.stdout.lower()


@pytest.mark.paper
def test_bundle_creation():
    """Test that bundle is created successfully with default settings."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip = Path(tmpdir) / "test_bundle.zip"
        
        result = subprocess.run(
            [
                "python",
                "scripts/build_overleaf_bundle.py",
                "--out",
                str(temp_zip),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        # Should succeed
        assert result.returncode == 0, f"Bundle creation failed: {result.stdout}\n{result.stderr}"
        assert temp_zip.exists(), "ZIP file was not created"
        assert temp_zip.stat().st_size > 0, "ZIP file is empty"


@pytest.mark.paper
def test_bundle_contents():
    """Test that bundle contains required files and excludes unwanted ones."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip = Path(tmpdir) / "test_bundle.zip"
        
        result = subprocess.run(
            [
                "python",
                "scripts/build_overleaf_bundle.py",
                "--out",
                str(temp_zip),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        # Extract and check contents
        with zipfile.ZipFile(temp_zip, 'r') as zf:
            filenames = set(zf.namelist())
            
            # Required files
            assert 'paper/main.tex' in filenames, "Missing main.tex"
            assert 'paper/metrics_values.tex' in filenames, "Missing metrics_values.tex"
            assert 'paper/SUBMISSION.md' in filenames, "Missing SUBMISSION.md"
            
            # Check for figures (at least one should exist)
            figure_files = [f for f in filenames if f.startswith('paper/figures/') and f.endswith('.pdf')]
            assert len(figure_files) >= 1, f"No figures found in bundle. Files: {filenames}"
            
            # Unwanted files should NOT be present
            unwanted_patterns = [
                'src/', 'tests/', 'artifacts/', 'docs/',
                '.pyc', '__pycache__', '.git',
                '.aux', '.log', '.out', '.synctex.gz'
            ]
            
            for pattern in unwanted_patterns:
                matching = [f for f in filenames if pattern in f]
                assert len(matching) == 0, f"Bundle contains unwanted files matching '{pattern}': {matching}"


@pytest.mark.paper
def test_bundle_main_tex_valid():
    """Test that main.tex in bundle is valid LaTeX."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip = Path(tmpdir) / "test_bundle.zip"
        
        result = subprocess.run(
            [
                "python",
                "scripts/build_overleaf_bundle.py",
                "--out",
                str(temp_zip),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        # Extract main.tex and check structure
        with zipfile.ZipFile(temp_zip, 'r') as zf:
            main_tex_content = zf.read('paper/main.tex').decode('utf-8')
            
            # Check for essential LaTeX structure
            assert r'\documentclass' in main_tex_content
            assert r'\begin{document}' in main_tex_content
            assert r'\end{document}' in main_tex_content
            
            # Check for metrics guard
            assert r'\IfFileExists{metrics_values.tex}' in main_tex_content
            
            # Check for fallback macros
            assert r'\AccuracyValue' in main_tex_content


@pytest.mark.paper
def test_bundle_figures_referenced():
    """Test that all figures referenced in main.tex exist in bundle."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip = Path(tmpdir) / "test_bundle.zip"
        
        result = subprocess.run(
            [
                "python",
                "scripts/build_overleaf_bundle.py",
                "--out",
                str(temp_zip),
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        import re
        
        with zipfile.ZipFile(temp_zip, 'r') as zf:
            # Extract main.tex
            main_tex_content = zf.read('paper/main.tex').decode('utf-8')
            
            # Find all \includegraphics references
            pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            figure_refs = re.findall(pattern, main_tex_content)
            
            # Get list of files in bundle
            bundle_files = set(zf.namelist())
            
            # Check each reference exists (with common extensions)
            for ref in figure_refs:
                ref = ref.strip()
                
                # Construct possible paths
                possible_paths = [
                    f'paper/{ref}',
                    f'paper/{ref}.pdf',
                    f'paper/{ref}.png',
                ]
                
                found = any(p in bundle_files for p in possible_paths)
                assert found, f"Figure reference '{ref}' not found in bundle. Bundle files: {bundle_files}"


@pytest.mark.paper
def test_bundle_deterministic():
    """Test that bundle creation is deterministic (same inputs = same output)."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip1 = Path(tmpdir) / "bundle1.zip"
        temp_zip2 = Path(tmpdir) / "bundle2.zip"
        
        # Create bundle twice
        for zip_path in [temp_zip1, temp_zip2]:
            result = subprocess.run(
                [
                    "python",
                    "scripts/build_overleaf_bundle.py",
                    "--out",
                    str(zip_path),
                ],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            assert result.returncode == 0
        
        # Both bundles should exist
        assert temp_zip1.exists() and temp_zip2.exists()
        
        # Compare file lists (timestamps may differ, but file lists should match)
        with zipfile.ZipFile(temp_zip1, 'r') as zf1, zipfile.ZipFile(temp_zip2, 'r') as zf2:
            files1 = sorted(zf1.namelist())
            files2 = sorted(zf2.namelist())
            
            assert files1 == files2, "Bundle file lists differ between runs"


@pytest.mark.paper
def test_missing_paper_dir():
    """Test that script fails gracefully when paper/ directory is missing."""
    repo_root = Path(__file__).parent.parent
    
    result = subprocess.run(
        [
            "python",
            "scripts/build_overleaf_bundle.py",
            "--paper_dir",
            "nonexistent_dir",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    
    # Should fail with clear error
    assert result.returncode == 1
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
