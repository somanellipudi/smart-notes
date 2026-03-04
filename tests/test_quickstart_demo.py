"""
Tests for quickstart_demo.py

Validates that quickstart demo runs successfully and produces correct output schema.
"""

import json
import subprocess
import tempfile
from pathlib import Path
import pytest


@pytest.mark.paper
def test_quickstart_smoke_mode_runs():
    """Test that quickstart demo runs in smoke mode without errors."""
    result = subprocess.run(
        ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2", "--out", "test_output.json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    # Clean up
    output_file = Path(__file__).parent.parent / "test_output.json"
    if output_file.exists():
        output_file.unlink()
    
    assert result.returncode == 0, f"Quickstart failed: {result.stderr}"


@pytest.mark.paper
def test_quickstart_output_schema():
    """Test that quickstart output matches required schema."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_output = f.name
    
    try:
        result = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2", "--out", temp_output],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0, f"Quickstart failed: {result.stderr}"
        
        # Load and validate JSON
        with open(temp_output, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check top-level fields
        assert "run_id" in data
        assert "smoke" in data
        assert "n" in data
        assert "tau" in data
        assert "examples" in data
        
        # Check types
        assert isinstance(data["run_id"], str)
        assert isinstance(data["smoke"], bool)
        assert isinstance(data["n"], int)
        assert isinstance(data["tau"], (int, float))
        assert isinstance(data["examples"], list)
        
        # Check smoke mode flag
        assert data["smoke"] is True
        
        # Check n matches
        assert data["n"] == 2
        assert len(data["examples"]) == 2
        
    finally:
        Path(temp_output).unlink(missing_ok=True)


@pytest.mark.paper
def test_quickstart_example_fields():
    """Test that each example has required fields."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_output = f.name
    
    try:
        result = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "3", "--out", temp_output],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        
        with open(temp_output, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, example in enumerate(data["examples"]):
            # Required fields
            assert "claim" in example, f"Example {i}: missing 'claim'"
            assert "pred_label" in example, f"Example {i}: missing 'pred_label'"
            assert "confidence" in example, f"Example {i}: missing 'confidence'"
            assert "abstained" in example, f"Example {i}: missing 'abstained'"
            assert "top_evidence" in example, f"Example {i}: missing 'top_evidence'"
            assert "stage_latency_ms" in example, f"Example {i}: missing 'stage_latency_ms'"
            
            # Field types
            assert isinstance(example["claim"], str)
            assert isinstance(example["pred_label"], str)
            assert isinstance(example["confidence"], (int, float))
            assert isinstance(example["abstained"], bool)
            assert isinstance(example["top_evidence"], list)
            assert isinstance(example["stage_latency_ms"], dict)
            
            # Valid pred_label
            assert example["pred_label"] in ["SUPPORTED", "REFUTED", "ABSTAIN"]
            
            # Confidence range
            assert 0.0 <= example["confidence"] <= 1.0
            
    finally:
        Path(temp_output).unlink(missing_ok=True)


@pytest.mark.paper
def test_quickstart_latency_fields():
    """Test that latency breakdown has all required fields."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_output = f.name
    
    try:
        result = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "1", "--out", temp_output],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        
        assert result.returncode == 0
        
        with open(temp_output, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        latency = data["examples"][0]["stage_latency_ms"]
        
        required_fields = [
            "retrieval",
            "filtering",
            "nli",
            "aggregation",
            "calibration",
            "selective",
            "explanation",
            "total",
        ]
        
        for field in required_fields:
            assert field in latency, f"Missing latency field: {field}"
            assert isinstance(latency[field], (int, float)), f"Latency field '{field}' must be numeric"
            assert latency[field] >= 0, f"Latency field '{field}' must be non-negative"
        
    finally:
        Path(temp_output).unlink(missing_ok=True)


@pytest.mark.paper
def test_quickstart_determinism():
    """Test that smoke mode produces deterministic output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
        temp_output1 = f1.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        temp_output2 = f2.name
    
    try:
        # Run twice
        for output in [temp_output1, temp_output2]:
            result = subprocess.run(
                ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2", "--out", output],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            assert result.returncode == 0
        
        # Load both outputs
        with open(temp_output1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(temp_output2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        # Compare examples (excluding run_id which has timestamp)
        assert len(data1["examples"]) == len(data2["examples"])
        
        for ex1, ex2 in zip(data1["examples"], data2["examples"]):
            assert ex1["claim"] == ex2["claim"]
            assert ex1["pred_label"] == ex2["pred_label"]
            assert ex1["confidence"] == ex2["confidence"]
            assert ex1["abstained"] == ex2["abstained"]
            assert ex1["top_evidence"] == ex2["top_evidence"]
            assert ex1["stage_latency_ms"] == ex2["stage_latency_ms"]
        
    finally:
        Path(temp_output1).unlink(missing_ok=True)
        Path(temp_output2).unlink(missing_ok=True)


@pytest.mark.paper
def test_quickstart_default_output_location():
    """Test default output location is artifacts/quickstart/output.json."""
    repo_root = Path(__file__).parent.parent
    default_output = repo_root / "artifacts" / "quickstart" / "output.json"

    original_content = None
    had_existing = default_output.exists()
    if had_existing:
        original_content = default_output.read_text(encoding="utf-8")

    try:
        result = subprocess.run(
            ["python", "scripts/quickstart_demo.py", "--smoke", "--n", "2"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

        assert result.returncode == 0, f"Quickstart failed: {result.stderr}"
        assert default_output.exists(), "Default quickstart output was not created"

        with open(default_output, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["smoke"] is True
        assert data["n"] == 2
    finally:
        if had_existing and original_content is not None:
            default_output.write_text(original_content, encoding="utf-8")
        else:
            default_output.unlink(missing_ok=True)


@pytest.mark.paper
def test_quickstart_help():
    """Test that --help flag works."""
    result = subprocess.run(
        ["python", "scripts/quickstart_demo.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    
    assert result.returncode == 0
    assert "quickstart demo" in result.stdout.lower()
    assert "--smoke" in result.stdout
    assert "--n" in result.stdout
    assert "--out" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
