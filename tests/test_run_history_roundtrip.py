"""Tests for run history save/load roundtrip."""

from src.persistence.run_history import load_run_history, save_run_history


def test_run_history_roundtrip(tmp_path):
    path = tmp_path / "run_history.json"
    history = [
        {"run_id": "run_1", "session_id": "session_a"},
        {"run_id": "run_2", "session_id": "session_b"},
    ]

    save_run_history(path, history)
    loaded = load_run_history(path)

    assert loaded == history
