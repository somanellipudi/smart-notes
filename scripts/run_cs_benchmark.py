#!/usr/bin/env python3
"""Thin CLI wrapper for CS benchmark ablation runner."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.cs_benchmark import AblationRunner, main, run_cs_benchmark


if __name__ == "__main__":
    raise SystemExit(main())
