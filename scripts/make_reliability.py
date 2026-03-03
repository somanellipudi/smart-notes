#!/usr/bin/env python3
"""Deprecated wrapper.

Use scripts/generate_paper_figures.py to generate reliability figure from
artifacts/metrics_summary.json (single source of truth).
"""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    print("[DEPRECATED] scripts/make_reliability.py -> scripts/generate_paper_figures.py")
    cmd = [sys.executable, "scripts/generate_paper_figures.py"]
    raise SystemExit(subprocess.call(cmd))
