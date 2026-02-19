"""Run history persistence for the last N runs."""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_run_history(index_path: Path) -> List[Dict[str, Any]]:
    """Load run history list from disk.

    Returns an empty list if the file does not exist or is invalid.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        return []

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    return data


def append_run(
    history: List[Dict[str, Any]],
    run_summary: Dict[str, Any],
    max_runs: int = 3
) -> List[Dict[str, Any]]:
    """Append a run summary and keep only the most recent max_runs."""
    if history is None:
        history = []

    run_id = run_summary.get("run_id")
    if run_id:
        history = [item for item in history if item.get("run_id") != run_id]

    history.append(run_summary)

    if len(history) > max_runs:
        history = history[-max_runs:]

    return history


def save_run_history(index_path: Path, history: List[Dict[str, Any]]) -> None:
    """Save run history list to disk."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
