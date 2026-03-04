"""
Tests for verify_paper_artifacts.py

Validates that the verification script correctly validates artifacts.
"""

import json
import subprocess
import tempfile
from pathlib import Path
import pytest


@pytest.mark.paper
def test_verify_help():
    """Test that --help flag works."""
    result = subprocess.run(
        ["python", "scripts/verify_paper_artifacts.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    assert result.returncode == 0
    assert "verify" in result.stdout.lower()
    assert "--quickstart" in result.stdout
    assert "--report" in result.stdout


@pytest.mark.paper
def test_verify_missing_quickstart():
    """Test that verifier fails gracefully when quickstart output missing."""
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        temp_report = f.name
    
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/verify_paper_artifacts.py",
                "--quickstart",
                "nonexistent.json",
                "--report",
                temp_report,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        # Should fail with exit code 1
        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
        
    finally:
        Path(temp_report).unlink(missing_ok=True)


@pytest.mark.paper
def test_verify_valid_quickstart():
    """Test that verifier passes with valid quickstart output."""
    # Create valid quickstart output
    valid_output = {
        "run_id": "20260304_120000",
        "smoke": True,
        "n": 2,
        "tau": 0.90,
        "examples": [
            {
                "claim": "Test claim 1",
                "pred_label": "SUPPORTED",
                "confidence": 0.95,
                "abstained": False,
                "top_evidence": ["Evidence 1", "Evidence 2", "Evidence 3"],
                "stage_latency_ms": {
                    "retrieval": 10.0,
                    "filtering": 2.0,
                    "nli": 5.0,
                    "aggregation": 1.0,
                    "calibration": 0.5,
                    "selective": 0.1,
                    "explanation": 0.1,
                    "total": 18.7,
                },
            },
            {
                "claim": "Test claim 2",
                "pred_label": "REFUTED",
                "confidence": 0.85,
                "abstained": False,
                "top_evidence": ["Evidence 1", "Evidence 2", "Evidence 3"],
                "stage_latency_ms": {
                    "retrieval": 12.0,
                    "filtering": 2.5,
                    "nli": 6.0,
                    "aggregation": 1.2,
                    "calibration": 0.6,
                    "selective": 0.1,
                    "explanation": 0.1,
                    "total": 22.5,
                },
            },
        ],
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_quickstart = f.name
        json.dump(valid_output, f)
    
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        temp_report = f.name
    
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/verify_paper_artifacts.py",
                "--quickstart",
                temp_quickstart,
                "--report",
                temp_report,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        # Should pass
        assert result.returncode == 0, f"Verification failed: {result.stdout}"
        
        # Check report was created
        report_path = Path(temp_report)
        assert report_path.exists()
        
        # Check report content
        report_content = report_path.read_text(encoding='utf-8')
        assert "Verification Report" in report_content
        assert "PASS" in report_content or "Valid" in report_content
        
    finally:
        Path(temp_quickstart).unlink(missing_ok=True)
        Path(temp_report).unlink(missing_ok=True)


@pytest.mark.paper
def test_verify_invalid_schema():
    """Test that verifier catches schema violations."""
    # Invalid output - missing required fields
    invalid_output = {
        "run_id": "20260304_120000",
        "smoke": True,
        # Missing 'n' and 'tau'
        "examples": [
            {
                "claim": "Test claim",
                # Missing 'pred_label', 'confidence', etc.
            }
        ],
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_quickstart = f.name
        json.dump(invalid_output, f)
    
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        temp_report = f.name
    
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/verify_paper_artifacts.py",
                "--quickstart",
                temp_quickstart,
                "--report",
                temp_report,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        # Should fail
        assert result.returncode == 1
        assert "invalid" in result.stdout.lower() or "missing" in result.stdout.lower()
        
    finally:
        Path(temp_quickstart).unlink(missing_ok=True)
        Path(temp_report).unlink(missing_ok=True)


@pytest.mark.paper
def test_verify_creates_directories():
    """Test that verifier creates missing artifact directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create valid quickstart output
        valid_output = {
            "run_id": "20260304_120000",
            "smoke": True,
            "n": 1,
            "tau": 0.90,
            "examples": [
                {
                    "claim": "Test",
                    "pred_label": "SUPPORTED",
                    "confidence": 0.95,
                    "abstained": False,
                    "top_evidence": ["E1", "E2", "E3"],
                    "stage_latency_ms": {
                        "retrieval": 10.0,
                        "filtering": 2.0,
                        "nli": 5.0,
                        "aggregation": 1.0,
                        "calibration": 0.5,
                        "selective": 0.1,
                        "explanation": 0.1,
                        "total": 18.7,
                    },
                }
            ],
        }
        
        quickstart_path = temp_path / "quickstart.json"
        with open(quickstart_path, 'w', encoding='utf-8') as f:
            json.dump(valid_output, f)
        
        report_path = temp_path / "subdir" / "report.md"
        
        result = subprocess.run(
            [
                "python",
                "scripts/verify_paper_artifacts.py",
                "--quickstart",
                str(quickstart_path),
                "--report",
                str(report_path),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        # Should succeed and create directory
        assert result.returncode == 0
        assert report_path.exists()
        assert report_path.parent.exists()


@pytest.mark.paper
def test_verify_with_end_to_end():
    """Integration test: run quickstart then verify."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        quickstart_out = temp_path / "quickstart.json"
        report_out = temp_path / "report.md"
        
        # Run quickstart
        result1 = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2", "--out", str(quickstart_out)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result1.returncode == 0, f"Quickstart failed: {result1.stderr}"
        assert quickstart_out.exists()
        
        # Run verification
        result2 = subprocess.run(
            [
                "python",
                "scripts/verify_paper_artifacts.py",
                "--quickstart",
                str(quickstart_out),
                "--report",
                str(report_out),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result2.returncode == 0, f"Verification failed: {result2.stdout}"
        assert report_out.exists()
        
        # Check report indicates success
        report_content = report_out.read_text(encoding='utf-8')
        assert "PASS" in report_content or "passed" in report_content.lower()


@pytest.mark.paper
def test_verify_default_report_location():
    """Test default verification output location is artifacts/verification/VerificationReport.md."""
    repo_root = Path(__file__).parent.parent
    quickstart_default = repo_root / "artifacts" / "quickstart" / "output.json"
    report_default = repo_root / "artifacts" / "verification" / "VerificationReport.md"

    original_quickstart = quickstart_default.read_text(encoding="utf-8") if quickstart_default.exists() else None
    original_report = report_default.read_text(encoding="utf-8") if report_default.exists() else None

    try:
        quickstart_result = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert quickstart_result.returncode == 0, f"Quickstart failed: {quickstart_result.stderr}"

        verify_result = subprocess.run(
            ["python", "scripts/verify_paper_artifacts.py"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert verify_result.returncode == 0, f"Verification failed: {verify_result.stdout}\n{verify_result.stderr}"
        assert report_default.exists(), "Default verification report was not created"
    finally:
        if original_quickstart is not None:
            quickstart_default.write_text(original_quickstart, encoding="utf-8")
        else:
            quickstart_default.unlink(missing_ok=True)

        if original_report is not None:
            report_default.write_text(original_report, encoding="utf-8")
        else:
            report_default.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
