"""Persistence utilities for Smart Notes."""

from .run_history import load_run_history, append_run, save_run_history

__all__ = [
    "load_run_history",
    "append_run",
    "save_run_history",
]
