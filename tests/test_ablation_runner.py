import pytest
from src.evaluation.ablation import run_ablations
from pathlib import Path


@pytest.mark.timeout(30)
def test_ablation_runner_creates_csv(tmp_path, monkeypatch):
    # monkeypatch runner.run to avoid expensive model loads
    import types
    fake_metrics = {"mode": "verifiable_full", "n": 10, "accuracy": 0.5, "macro_f1": 0.5, "ece": 0.1, "brier_score": 0.2}
    def fake_run(mode, output_dir):
        return fake_metrics.copy()
    monkeypatch.setattr("src.evaluation.ablation.run", fake_run)

    out = tmp_path / "abl"
    ablations = run_ablations(output_base=str(out))
    assert (out / "ablations_summary.csv").exists()
    assert isinstance(ablations, list)
    # ensure stub was used
    assert ablations and ablations[0]["n"] == 10
