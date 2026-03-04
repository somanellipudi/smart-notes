"""
Tests for real claims data loading and mock retrieval mode in leakage_scan.py.

Tests:
1. Create temp JSONL with real claims
2. Load with --retrieval_mode mock (deterministic passages)
3. Verify retrieval_mode field in JSON output
4. Verify claims_path is recorded correctly
5. Verify determinism with --seed
"""

import json
import subprocess
import tempfile
from pathlib import Path
import pytest


@pytest.mark.paper
def test_leakage_real_loader_jsonl_mock():
    """Test loading real claims from temp JSONL with mock retrieval mode."""
    repo_root = Path(__file__).parent.parent
    
    # Create temp JSONL with real claims
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_claims_file = f.name
        # Write 3 real claims
        f.write(json.dumps({"id": "claim_1", "text": "Paris is the capital of France"}) + "\n")
        f.write(json.dumps({"id": "claim_2", "text": "Water boils at 100 degrees Celsius"}) + "\n")
        f.write(json.dumps({"id": "claim_3", "text": "The Earth orbits the Sun"}) + "\n")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "scripts/leakage_scan.py",
                 "--claims", temp_claims_file,
                 "--retrieval_mode", "mock",
                 "--max_claims", "2",
                 "--outdir", tmpdir],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            
            # Check outputs exist
            json_file = Path(tmpdir) / "leakage_report.json"
            csv_file = Path(tmpdir) / "leakage_report.csv"
            
            assert json_file.exists(), "JSON report not created"
            assert csv_file.exists(), "CSV report not created"
            
            # Check JSON schema
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify retrieval_mode field
            assert "retrieval_mode" in data
            assert data["retrieval_mode"] == "mock", f"Expected retrieval_mode='mock', got {data['retrieval_mode']}"
            
            # Verify claims_path is the temp file
            assert data["claims_path"] == temp_claims_file, f"Expected claims_path={temp_claims_file}, got {data['claims_path']}"
            
            # Verify NOT synthetic
            assert "synthetic" not in data["claims_path"].lower()
            
            # Verify claims were loaded (should be 2 due to --max_claims 2)
            assert data["n_claims_scanned"] == 2
            
            # Verify per_claim records exist
            assert len(data["per_claim"]) == 22  # 2 claims × 11 k values
            
            # Verify first record has expected claim_id from JSONL
            first_claim_id = data["per_claim"][0]["claim_id"]
            assert "claim_1" in first_claim_id or "claim_2" in first_claim_id
            
    finally:
        # Clean up temp file
        Path(temp_claims_file).unlink(missing_ok=True)


@pytest.mark.paper
def test_leakage_real_loader_json_mock():
    """Test loading real claims from temp JSON with mock retrieval mode."""
    repo_root = Path(__file__).parent.parent
    
    # Create temp JSON with real claims
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        temp_claims_file = f.name
        json.dump([
            {"id": "c1", "text": "The sky is blue"},
            {"id": "c2", "text": "Grass is green"},
        ], f)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "scripts/leakage_scan.py",
                 "--claims", temp_claims_file,
                 "--retrieval_mode", "mock",
                 "--outdir", tmpdir],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            
            json_file = Path(tmpdir) / "leakage_report.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["retrieval_mode"] == "mock"
            assert data["claims_path"] == temp_claims_file
            assert data["n_claims_scanned"] == 2
            
    finally:
        Path(temp_claims_file).unlink(missing_ok=True)


@pytest.mark.paper
def test_leakage_mock_deterministic():
    """Test that mock mode is deterministic with same seed."""
    repo_root = Path(__file__).parent.parent
    
    # Create temp JSONL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_claims_file = f.name
        f.write(json.dumps({"text": "Machine learning is powerful"}) + "\n")
        f.write(json.dumps({"text": "Deep learning uses neural networks"}) + "\n")
    
    try:
        outputs = []
        
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["python", "scripts/leakage_scan.py",
                     "--claims", temp_claims_file,
                     "--retrieval_mode", "mock",
                     "--seed", "42",
                     "--outdir", tmpdir],
                    capture_output=True,
                    text=True,
                    cwd=repo_root,
                )
                
                assert result.returncode == 0
                
                json_file = Path(tmpdir) / "leakage_report.json"
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                outputs.append(data)
        
        # Verify per_claim records are identical
        assert len(outputs[0]["per_claim"]) == len(outputs[1]["per_claim"])
        
        for r0, r1 in zip(outputs[0]["per_claim"], outputs[1]["per_claim"]):
            assert r0["claim_id"] == r1["claim_id"]
            assert r0["max_lco"] == r1["max_lco"]
            assert r0["max_lcs"] == r1["max_lcs"]
            assert r0["max_substring"] == r1["max_substring"]
        
    finally:
        Path(temp_claims_file).unlink(missing_ok=True)


@pytest.mark.paper
def test_leakage_fail_no_dataset():
    """Test that non-smoke mode fails when no real dataset found."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py",
             "--outdir", tmpdir,
             "--max_claims", "1"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        # Should fail with non-zero exit code
        assert result.returncode != 0, "Should fail when no dataset found"
        
        # Should print error message
        assert "[ERROR]" in result.stderr
        assert "No claims file found" in result.stderr
        
        # Should NOT create a report
        json_file = Path(tmpdir) / "leakage_report.json"
        assert not json_file.exists(), "Should not create report when no dataset found"


@pytest.mark.paper
def test_leakage_mock_passages_are_deterministic():
    """Test that mock passages are consistent across runs."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_claims_file = f.name
        f.write(json.dumps({"text": "Test claim"}) + "\n")
    
    try:
        outputs = []
        
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["python", "scripts/leakage_scan.py",
                     "--claims", temp_claims_file,
                     "--retrieval_mode", "mock",
                     "--outdir", tmpdir],
                    capture_output=True,
                    text=True,
                    cwd=repo_root,
                )
                
                assert result.returncode == 0
                
                json_file = Path(tmpdir) / "leakage_report.json"
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                outputs.append(data)
        
        # Mock passages should produce consistent metrics
        for i, (metric_name, _) in enumerate(outputs[0]["summary"].items()):
            if i >= 3:  # Only check the 3 metrics
                break
            # Summary statistics should be identical
            assert outputs[0]["summary"][metric_name]["max"] == outputs[1]["summary"][metric_name]["max"]
            
    finally:
        Path(temp_claims_file).unlink(missing_ok=True)


@pytest.mark.paper
def test_leakage_mock_mode_labeled_in_json():
    """Test that mock mode is clearly labeled in output JSON."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
        temp_claims_file = f.name
        f.write(json.dumps({"text": "Sample claim"}) + "\n")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "scripts/leakage_scan.py",
                 "--claims", temp_claims_file,
                 "--retrieval_mode", "mock",
                 "--outdir", tmpdir],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            
            assert result.returncode == 0
            
            json_file = Path(tmpdir) / "leakage_report.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Must have retrieval_mode field
            assert "retrieval_mode" in data
            # Must be labeled mock
            assert data["retrieval_mode"] == "mock"
            # README warning should be clear on this
            readme_file = Path(tmpdir) / "README.md"
            readme_content = readme_file.read_text(encoding='utf-8')
            assert "mock" in readme_content.lower()
            
    finally:
        Path(temp_claims_file).unlink(missing_ok=True)
