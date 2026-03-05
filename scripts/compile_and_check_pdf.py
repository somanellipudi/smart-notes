#!/usr/bin/env python3
"""
Compile bundled paper and verify hygiene on COMPILED PDF.

Flow:
1) Extract dist/overleaf_submission.zip to temporary directory
2) Compile main.tex with pdflatex (if available)
3) Run scripts/check_pdf_text_hygiene.py on compiled PDF
4) Fail on banned architecture strings or replacement artifacts

Exit codes:
  0 = PASS (or pdflatex unavailable, skipped)
  1 = FAIL
  2 = ERROR
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_ZIP = REPO_ROOT / "dist" / "overleaf_submission.zip"
HYGIENE = REPO_ROOT / "scripts" / "check_pdf_text_hygiene.py"


def _safe(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def find_pdflatex() -> str | None:
    cmd = ["where", "pdflatex"] if sys.platform == "win32" else ["which", "pdflatex"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if sys.platform == "win32":
            candidates = [
                Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
                Path(r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"),
                Path(r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def resolve_tex_location(extract_dir: Path) -> tuple[Path, Path] | None:
    candidates = [
        extract_dir / "paper" / "main.tex",
        extract_dir / "main.tex",
    ]
    for tex in candidates:
        if tex.exists():
            return tex.parent, tex
    return None


def _run_process(
    cmd: list[str],
    cwd: Path | None = None,
    timeout_seconds: int = 90,
) -> subprocess.CompletedProcess | None:
    env = dict(__import__("os").environ)
    env["MIKTEX_NONINTERACTIVE"] = "1"
    env["MIKTEX_AUTO_INSTALL"] = "1"
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(_safe(f"[ERROR] Command timed out after {timeout_seconds}s: {' '.join(cmd)}"))
        return None


def _extract_missing_sty_package(log_text: str) -> str | None:
    match = re.search(r"File `([^`]+)\\.sty' not found", log_text)
    if not match:
        return None
    return match.group(1).strip()


def _install_miktex_package(package_name: str) -> bool:
    install_commands = [
        ["mpm", f"--install={package_name}"],
        ["miktex", "packages", "install", package_name],
    ]

    for cmd in install_commands:
        result = _run_process(cmd, timeout_seconds=120)
        if result and result.returncode == 0:
            print(f"[OK] Installed MiKTeX package: {package_name} via {' '.join(cmd)}")
            return True

    print(f"[ERROR] Failed to auto-install MiKTeX package: {package_name}")
    return False


def compile_pdf(pdflatex: str, workdir: Path, tex_file: Path) -> Path | None:
    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file.name]

    installed_once: set[str] = set()

    # Two passes for references/tables, with one missing-package recovery per pass
    for _ in range(2):
        result = _run_process(cmd, cwd=workdir, timeout_seconds=120)
        if result is None:
            return None

        if result.returncode != 0:
            combined = result.stdout + "\n" + result.stderr
            missing_pkg = _extract_missing_sty_package(combined)

            if missing_pkg and missing_pkg not in installed_once:
                print(_safe(f"[WARN] Missing LaTeX package detected: {missing_pkg}.sty"))
                if _install_miktex_package(missing_pkg):
                    installed_once.add(missing_pkg)
                    retry = _run_process(cmd, cwd=workdir, timeout_seconds=120)
                    if retry and retry.returncode == 0:
                        continue
                    if retry is None:
                        return None
                    combined = retry.stdout + "\n" + retry.stderr

            print("[ERROR] pdflatex failed")
            print(_safe(combined[-2000:]))
            return None

    pdf_path = tex_file.with_suffix(".pdf")
    if not pdf_path.exists():
        print(_safe(f"[ERROR] Compiled PDF not found: {pdf_path}"))
        return None

    return pdf_path


def run_hygiene_check(compiled_pdf: Path) -> int:
    if not HYGIENE.exists():
        print(f"[ERROR] Missing hygiene checker: {HYGIENE}")
        return 2

    cmd = [
        sys.executable,
        str(HYGIENE),
        str(compiled_pdf),
        "--check-compiled",
    ]

    result = _run_process(cmd, cwd=REPO_ROOT, timeout_seconds=60)
    if result is None:
        return 2

    if result.stdout:
        print(_safe(result.stdout.strip()))
    if result.stderr:
        print(_safe(result.stderr.strip()))

    if result.returncode != 0:
        print("[ERROR] Compiled PDF hygiene check failed")
        return 1

    print("[OK] Compiled PDF hygiene check passed")
    return 0


def main() -> int:
    if not DIST_ZIP.exists():
        print(_safe(f"[ERROR] Missing bundle: {DIST_ZIP}"))
        return 1

    pdflatex = find_pdflatex()
    if not pdflatex:
        print("[WARN] pdflatex not found; skipping compiled PDF check")
        return 0

    print(_safe(f"[OK] pdflatex found: {pdflatex}"))

    with tempfile.TemporaryDirectory(prefix="compiled_pdf_check_") as temp_root:
        temp_dir = Path(temp_root)

        try:
            with zipfile.ZipFile(DIST_ZIP, "r") as archive:
                archive.extractall(temp_dir)
            print(_safe(f"[OK] Extracted bundle to {temp_dir}"))
        except Exception as exc:
            print(_safe(f"[ERROR] Failed to extract {DIST_ZIP}: {exc}"))
            return 2

        resolved = resolve_tex_location(temp_dir)
        if resolved is None:
            print("[ERROR] main.tex not found in extracted bundle")
            return 1

        workdir, tex_file = resolved
        print(_safe(f"[OK] Compiling {tex_file}"))

        compiled_pdf = compile_pdf(pdflatex, workdir, tex_file)
        if compiled_pdf is None:
            return 1

        print(_safe(f"[OK] Compiled PDF created: {compiled_pdf}"))
        return run_hygiene_check(compiled_pdf)


if __name__ == "__main__":
    raise SystemExit(main())
