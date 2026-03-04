"""
Tests for leakage_scan.py smoke mode and full pipeline.

Validates that the script produces correct outputs with exact schema matching
and deterministic behavior.
"""

import json
import csv
import subprocess
import tempfile
from pathlib import Path
import pytest


@pytest.mark.paper
def test_leakage_scan_smoke_mode_runs():
    """Test that leakage_scan runs successfully in smoke mode."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "3", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"


@pytest.mark.paper
def test_leakage_scan_smoke_mode_creates_files():
    """Test that smoke mode creates all required output files."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "3", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        tmpdir_path = Path(tmpdir)
        json_file = tmpdir_path / "leakage_report.json"
        csv_file = tmpdir_path / "leakage_report.csv"
        readme_file = tmpdir_path / "README.md"
        
        assert json_file.exists(), "JSON report not created"
        assert csv_file.exists(), "CSV report not created"
        assert readme_file.exists(), "README not created"


@pytest.mark.paper
def test_leakage_scan_json_schema():
    """Test that JSON output matches expected schema."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "2", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        json_file = Path(tmpdir) / "leakage_report.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check top-level keys
        assert "run_id" in data
        assert "seed" in data
        assert "k" in data, "Missing 'k' field (canonical k value)"
        assert "k2" in data, "Missing 'k2' field (canonical k2 value)"
        assert "k_values" in data
        assert "claim_count_definition" in data, "Missing 'claim_count_definition' field"
        assert "claims_path" in data
        assert "n_claims_scanned" in data
        assert "thresholds" in data
        assert "summary" in data
        assert "per_claim" in data
        
        # Check types
        assert isinstance(data["seed"], int)
        assert isinstance(data["k"], int), "k should be integer"
        assert isinstance(data["k2"], int), "k2 should be integer"
        assert isinstance(data["k_values"], list), "k_values should be list"
        assert len(data["k_values"]) == 2, "k_values should contain exactly [k, k2]"
        assert data["k_values"] == [data["k"], data["k2"]], "k_values should be [k, k2]"
        assert data["claim_count_definition"] == "max_over_k", "claim_count_definition should be 'max_over_k'"
        assert isinstance(data["n_claims_scanned"], int)
        assert isinstance(data["thresholds"], list)
        assert isinstance(data["summary"], dict)
        assert isinstance(data["per_claim"], list)
        
        # Check seed defaults to 0 in smoke mode when not specified
        assert data["seed"] == 0, "Seed should default to 0"


@pytest.mark.paper
def test_leakage_scan_json_summary_schema():
    """Test that summary statistics have correct schema with row vs claim counts."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "2", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        json_file = Path(tmpdir) / "leakage_report.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data["summary"]
        
        # Check for all three metrics
        assert "lco" in summary
        assert "lcs" in summary
        assert "substring" in summary
        
        # Check each metric's fields
        for metric_name in ["lco", "lcs", "substring"]:
            metric_stats = summary[metric_name]
            assert "max" in metric_stats
            assert "p95" in metric_stats
            assert "mean" in metric_stats
            
            # Check for row_count_* fields (counts across all claim×k rows)
            assert "row_count_ge_0.10" in metric_stats, f"Missing row_count_ge_0.10 in {metric_name}"
            assert "row_count_ge_0.15" in metric_stats, f"Missing row_count_ge_0.15 in {metric_name}"
            assert "row_count_ge_0.20" in metric_stats, f"Missing row_count_ge_0.20 in {metric_name}"
            
            # Check for claim_count_* fields (counts across unique claims, max-over-k per claim)
            assert "claim_count_ge_0.10" in metric_stats, f"Missing claim_count_ge_0.10 in {metric_name}"
            assert "claim_count_ge_0.15" in metric_stats, f"Missing claim_count_ge_0.15 in {metric_name}"
            assert "claim_count_ge_0.20" in metric_stats, f"Missing claim_count_ge_0.20 in {metric_name}"
            
            # Check types
            assert isinstance(metric_stats["max"], (int, float))
            assert isinstance(metric_stats["p95"], (int, float))
            assert isinstance(metric_stats["mean"], (int, float))
            assert isinstance(metric_stats["row_count_ge_0.10"], int)
            assert isinstance(metric_stats["row_count_ge_0.15"], int)
            assert isinstance(metric_stats["row_count_ge_0.20"], int)
            assert isinstance(metric_stats["claim_count_ge_0.10"], int)
            assert isinstance(metric_stats["claim_count_ge_0.15"], int)
            assert isinstance(metric_stats["claim_count_ge_0.20"], int)
            
            # Sanity check: claim_count should be <= n_claims_scanned
            assert metric_stats["claim_count_ge_0.10"] <= data["n_claims_scanned"]
            assert metric_stats["claim_count_ge_0.15"] <= data["n_claims_scanned"]
            assert metric_stats["claim_count_ge_0.20"] <= data["n_claims_scanned"]
            
            # Sanity check: row_count should be <= n_claims_scanned * (k2 - k + 1)
            expected_total_rows = data["n_claims_scanned"] * (data["k2"] - data["k"] + 1)
            assert metric_stats["row_count_ge_0.10"] <= expected_total_rows
            assert metric_stats["row_count_ge_0.15"] <= expected_total_rows
            assert metric_stats["row_count_ge_0.20"] <= expected_total_rows


@pytest.mark.paper
def test_leakage_scan_json_per_claim_schema():
    """Test that per-claim records have correct schema."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "1", "--outdir", tmpdir, "--k", "5", "--k2", "5"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        json_file = Path(tmpdir) / "leakage_report.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data["per_claim"]) > 0
        
        record = data["per_claim"][0]
        
        # Check required fields
        assert "claim_id" in record
        assert "claim" in record
        assert "k" in record
        assert "max_lco" in record
        assert "max_lcs" in record
        assert "max_substring" in record
        assert "max_source" in record
        
        # Check max_source structure
        max_source = record["max_source"]
        assert "passage" in max_source
        assert "doc_id" in max_source
        assert "rank" in max_source
        
        # Check types
        assert isinstance(record["claim_id"], str)
        assert isinstance(record["claim"], str)
        assert isinstance(record["k"], int)
        assert isinstance(record["max_lco"], (int, float))
        assert isinstance(record["max_lcs"], (int, float))
        assert isinstance(record["max_substring"], (int, float))


@pytest.mark.paper
def test_leakage_scan_csv_format():
    """Test that CSV has correct headers and format."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "2", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        csv_file = Path(tmpdir) / "leakage_report.csv"
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0, "CSV is empty"
        
        # Check header
        expected_headers = {"claim_id", "k", "max_lco", "max_lcs", "max_substring", "doc_id", "rank"}
        actual_headers = set(rows[0].keys())
        assert expected_headers == actual_headers, f"Headers mismatch: {actual_headers}"
        
        # Check first row has valid data
        first_row = rows[0]
        assert first_row["claim_id"]
        assert first_row["k"]
        assert float(first_row["max_lco"]) >= 0.0
        assert float(first_row["max_lcs"]) >= 0.0
        assert float(first_row["max_substring"]) >= 0.0


@pytest.mark.paper
def test_leakage_scan_deterministic():
    """Test that running twice with same seed produces identical output."""
    repo_root = Path(__file__).parent.parent
    
    outputs = []
    
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "3", "--seed", "42", "--outdir", tmpdir],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            
            assert result.returncode == 0
            
            json_file = Path(tmpdir) / "leakage_report.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            outputs.append(data)
    
    # Compare outputs (excluding run_id which has timestamp)
    for key in ["seed", "k_values", "n_claims_scanned", "thresholds"]:
        assert outputs[0][key] == outputs[1][key], f"Mismatch in {key}"
    
    # Compare per_claim records (should be identical)
    assert len(outputs[0]["per_claim"]) == len(outputs[1]["per_claim"])
    
    for i, (record0, record1) in enumerate(zip(outputs[0]["per_claim"], outputs[1]["per_claim"])):
        assert record0["claim_id"] == record1["claim_id"], f"Mismatch in claim_id at index {i}"
        assert record0["claim"] == record1["claim"], f"Mismatch in claim at index {i}"
        assert record0["k"] == record1["k"], f"Mismatch in k at index {i}"
        assert record0["max_lco"] == record1["max_lco"], f"Mismatch in max_lco at index {i}"
        assert record0["max_lcs"] == record1["max_lcs"], f"Mismatch in max_lcs at index {i}"
        assert record0["max_substring"] == record1["max_substring"], f"Mismatch in max_substring at index {i}"


@pytest.mark.paper
def test_leakage_scan_readme_exists():
    """Test that README is generated and has expected content."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "1", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        readme_file = Path(tmpdir) / "README.md"
        assert readme_file.exists()
        
        content = readme_file.read_text(encoding='utf-8')
        
        # Check for key sections
        assert "Lexical Leakage" in content
        assert "LCO" in content
        assert "LCS" in content
        assert "SUBSTRING" in content or "Longest Common Substring" in content
        assert "Thresholds" in content
        assert "Examples" in content or "Example" in content


@pytest.mark.paper
def test_leakage_scan_with_k_range():
    """Test that k and k2 are correctly recorded in output."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "1", "--k", "5", "--k2", "7", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        json_file = Path(tmpdir) / "leakage_report.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check that k and k2 are recorded correctly
        assert data["k"] == 5
        assert data["k2"] == 7
        
        # k_values should be [k, k2] pair only
        assert data["k_values"] == [5, 7]
        
        # Per-claim records should still range from k to k2
        # For 1 claim with k in [5, 6, 7], should have 3 records
        assert len(data["per_claim"]) == 3
        
        # Each record should have a different k
        ks = [record["k"] for record in data["per_claim"]]
        assert ks == [5, 6, 7]


@pytest.mark.paper
def test_leakage_scan_claims_path_recorded():
    """Test that claims_path is recorded in output."""
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["python", "scripts/leakage_scan.py", "--smoke", "--max_claims", "1", "--outdir", tmpdir],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        
        assert result.returncode == 0
        
        json_file = Path(tmpdir) / "leakage_report.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # In smoke mode, should record synth path
        assert "smoke_mode" in data["claims_path"] or data["claims_path"] != ""


@pytest.mark.paper
def test_leakage_scan_smoke_help():
    """Test that --help works."""
    repo_root = Path(__file__).parent.parent
    
    result = subprocess.run(
        ["python", "scripts/leakage_scan.py", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    
    assert result.returncode == 0
    assert "smoke" in result.stdout or "smoke" in result.stderr
    assert "outdir" in result.stdout or "outdir" in result.stderr
