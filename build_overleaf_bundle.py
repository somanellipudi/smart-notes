#!/usr/bin/env python3
"""Backward-compatible wrapper for Overleaf bundle validation/build.

This preserves support for the historical root-level command:
    python build_overleaf_bundle.py --validate-only

Canonical entrypoint remains:
    python scripts/build_overleaf_bundle.py ...
"""

from pathlib import Path
import runpy
import sys


def main() -> int:
    script_path = Path(__file__).parent / "scripts" / "build_overleaf_bundle.py"
    if not script_path.exists():
        print(f"[ERROR] Missing script: {script_path}")
        return 1

    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
