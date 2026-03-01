from src.evaluation.runner import run
from pathlib import Path


def test_runner_creates_outputs(tmp_path, monkeypatch):
    out = tmp_path / "res"
    metrics = run(mode="baseline_retriever", output_dir=str(out))
    assert (out / "metrics.json").exists()
    assert (out / "metrics.md").exists()
    assert (out / "figures" / "reliability.png").exists()
    assert (out / "figures" / "risk_coverage.png").exists()
    # risk-coverage data should be present in metrics.json
    assert "risk_coverage" in metrics
    rc = metrics["risk_coverage"]
    assert isinstance(rc.get("coverage"), list) and isinstance(rc.get("accuracy"), list)
    assert metrics["mode"] == "baseline_retriever"
