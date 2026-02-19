"""Tests for run history rotation."""

from src.persistence.run_history import append_run


def test_run_history_rotation():
    history = [
        {"run_id": "run_1"},
        {"run_id": "run_2"},
        {"run_id": "run_3"},
    ]
    new_run = {"run_id": "run_4"}

    updated = append_run(history, new_run, max_runs=3)

    assert [item["run_id"] for item in updated] == ["run_2", "run_3", "run_4"]
