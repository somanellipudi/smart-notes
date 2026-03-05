#!/usr/bin/env python3
"""
Build Overleaf submission bundle for CalibraTeach IEEE Access paper.

Creates a deterministic, validated ZIP archive containing only the required
paper compilation files. Fails fast with clear errors if any assets are missing.

Usage:
    python scripts/build_overleaf_bundle.py
    python scripts/build_overleaf_bundle.py --out dist/custom.zip
    python scripts/build_overleaf_bundle.py --paper_dir paper --validate-only
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Set, Tuple


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Overleaf submission bundle for IEEE Access paper"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dist/overleaf_submission.zip"),
        help="Output ZIP file path (default: dist/overleaf_submission.zip)",
    )
    parser.add_argument(
        "--paper_dir",
        type=Path,
        default=Path("paper"),
        help="Paper source directory (default: paper)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate paper assets without creating ZIP",
    )
    return parser.parse_args()


def extract_figure_paths(tex_content: str) -> Set[str]:
    """
    Extract all \\includegraphics paths from LaTeX content.
    
    Handles variations:
    - \\includegraphics{file.pdf}
    - \\includegraphics[options]{file.pdf}
    - Paths with/without extensions
    """
    # Match \includegraphics with optional arguments
    pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    matches = re.findall(pattern, tex_content)
    
    # Normalize paths and add common extensions if missing
    figure_paths = set()
    for match in matches:
        path = match.strip()
        # If no extension, check for .pdf, .png, .jpg
        if not Path(path).suffix:
            for ext in ['.pdf', '.png', '.jpg', '.eps']:
                figure_paths.add(f"{path}{ext}")
        else:
            figure_paths.add(path)
    
    return figure_paths


def validate_main_tex(paper_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate main.tex exists and has required structure.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    main_tex = paper_dir / "main.tex"
    
    if not main_tex.exists():
        errors.append(f"ERROR: main.tex not found at {main_tex}")
        return False, errors
    
    content = main_tex.read_text(encoding='utf-8')
    
    # Check for essential LaTeX structure
    if r'\documentclass' not in content:
        errors.append("ERROR: main.tex missing \\documentclass declaration")
    
    if r'\begin{document}' not in content:
        errors.append("ERROR: main.tex missing \\begin{document}")
    
    if r'\end{document}' not in content:
        errors.append("ERROR: main.tex missing \\end{document}")
    
    # Check for metrics_values.tex guard
    if r'\IfFileExists{metrics_values.tex}' not in content:
        errors.append(
            "WARNING: main.tex should include \\IfFileExists guard for metrics_values.tex"
        )
    
    return len(errors) == 0, errors


def validate_figures(paper_dir: Path, main_tex_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate all \\includegraphics references point to existing files.
    Also validates architecture.pdf has no embedded specs (canonical-only, clean).
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    content = main_tex_path.read_text(encoding='utf-8')

    # Enforce single architecture includegraphics reference in main.tex with line reporting
    arch_matches = []
    pattern = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*(?:^|/)architecture\.pdf)\}')
    for line_number, line in enumerate(content.splitlines(), start=1):
        for match in pattern.finditer(line):
            arch_matches.append((line_number, match.group(1).strip()))

    if len(arch_matches) != 1:
        errors.append(
            f"ERROR: Expected exactly 1 architecture includegraphics reference in "
            f"{main_tex_path}, found {len(arch_matches)}"
        )
        for line_number, ref in arch_matches:
            errors.append(f"  - {main_tex_path}:{line_number}: {ref}")
        return False, errors

    if arch_matches[0][1] != "figures/architecture.pdf":
        errors.append(
            "ERROR: Architecture include path in main.tex must be figures/architecture.pdf "
            f"(found: {arch_matches[0][1]})"
        )
        return False, errors

    figure_paths = extract_figure_paths(content)
    
    if not figure_paths:
        errors.append("WARNING: No figures found via \\includegraphics in main.tex")
        return True, errors  # Not a fatal error
    
    print(f"Found {len(figure_paths)} figure references in main.tex")
    
    missing_figures = []
    for fig_path in sorted(figure_paths):
        full_path = paper_dir / fig_path
        if not full_path.exists():
            missing_figures.append(fig_path)
        else:
            print(f"  [OK] {fig_path}")
    
    if missing_figures:
        errors.append("ERROR: Missing figure files:")
        for fig in missing_figures:
            errors.append(f"  - {fig}")
        return False, errors
    
    # Architecture PDF hygiene check (canonical only)
    print("\n[HYGIENE CHECK] Validating architecture.pdf for embedded specs...")
    canonical_arch = paper_dir / "figures" / "architecture.pdf"
    if canonical_arch.exists():
        import subprocess
        repo_root = paper_dir.parent
        scripts_dir = repo_root / "scripts"
        
        hygiene_result = subprocess.run(
            [__import__("sys").executable, str(scripts_dir / "check_pdf_text_hygiene.py"),
             str(canonical_arch), "--check-architecture"],
            capture_output=True,
            text=True,
            cwd=str(repo_root)
        )
        
        if hygiene_result.returncode != 0:
            errors.append("ERROR: Architecture PDF hygiene check failed (embedded specs detected)")
            errors.append(hygiene_result.stdout)
            return False, errors
        
        print("  [OK] architecture.pdf is clean (no embedded specs)")
    
    return True, errors


def validate_metrics_values(paper_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate metrics_values.tex exists or main.tex has fallback.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    metrics_file = paper_dir / "metrics_values.tex"
    main_tex = paper_dir / "main.tex"
    
    if not metrics_file.exists():
        # Check if main.tex has fallback
        if not main_tex.exists():
            errors.append("ERROR: Neither metrics_values.tex nor main.tex found")
            return False, errors
        
        content = main_tex.read_text(encoding='utf-8')
        if r'\AccuracyValue' not in content:
            errors.append(
                "ERROR: metrics_values.tex missing and main.tex has no \\AccuracyValue fallback"
            )
            return False, errors
        
        print("  [WARN] metrics_values.tex missing, but main.tex has fallback macros")
        return True, errors
    
    # Validate metrics_values.tex content
    content = metrics_file.read_text(encoding='utf-8')
    required_macros = [r'\AccuracyValue', r'\ECEValue', r'\AUCACValue']
    missing_macros = [m for m in required_macros if m not in content]
    
    if missing_macros:
        errors.append(f"ERROR: metrics_values.tex missing required macros: {missing_macros}")
        return False, errors
    
    print("  [OK] metrics_values.tex exists with required macros")
    return True, errors


def validate_references(paper_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate references.bib exists (if using BibTeX).
    
    Note: Current paper uses embedded bibliography, so this is optional.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    bib_file = paper_dir / "references.bib"
    
    if not bib_file.exists():
        print("  [WARN] references.bib not found (OK if using embedded bibliography)")
        return True, errors  # Not fatal
    
    print("  [OK] references.bib exists")
    return True, errors


def validate_paper_assets(paper_dir: Path) -> bool:
    """
    Comprehensive validation of all paper assets.
    
    Returns:
        True if all required assets are valid, False otherwise
    """
    print(f"\nValidating paper assets in {paper_dir}/\n")
    print("=" * 60)
    
    all_valid = True
    all_errors = []
    
    # 1. Validate main.tex
    print("\n[1/4] Validating main.tex...")
    valid, errors = validate_main_tex(paper_dir)
    all_valid &= valid
    all_errors.extend(errors)
    if valid and not errors:
        print("  [OK] main.tex structure valid")
    
    # 2. Validate figures
    print("\n[2/4] Validating figures...")
    main_tex = paper_dir / "main.tex"
    if main_tex.exists():
        valid, errors = validate_figures(paper_dir, main_tex)
        all_valid &= valid
        all_errors.extend(errors)
    
    # 3. Validate metrics_values.tex
    print("\n[3/4] Validating metrics_values.tex...")
    valid, errors = validate_metrics_values(paper_dir)
    all_valid &= valid
    all_errors.extend(errors)
    
    # 4. Validate references.bib
    print("\n[4/4] Validating references.bib...")
    valid, errors = validate_references(paper_dir)
    all_valid &= valid
    all_errors.extend(errors)
    
    # Print summary
    print("\n" + "=" * 60)
    if all_valid:
        print("\n[OK] All paper assets validated successfully\n")
    else:
        print("\n[ERROR] Validation failed with errors:\n")
        for error in all_errors:
            print(f"  {error}")
        print()
    
    return all_valid


def create_overleaf_bundle(paper_dir: Path, output_path: Path) -> bool:
    """
    Create ZIP archive containing only paper compilation files.
    
    Includes:
    - main.tex
    - metrics_values.tex
    - figures/*.pdf
    - references.bib (if exists)
    - SUBMISSION.md
    
    Excludes:
    - src/, tests/, artifacts/, docs/ (repo code)
    - LaTeX build artifacts (*.aux, *.log, etc.)
    - Python caches
    
    Returns:
        True if bundle created successfully, False otherwise
    """
    print(f"\nCreating Overleaf bundle: {output_path}\n")
    print("=" * 60)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extensions to exclude
    exclude_extensions = {
        '.aux', '.log', '.out', '.toc', '.bbl', '.blg',
        '.synctex.gz', '.fdb_latexmk', '.fls', '.pyc',
        '.pyo', '.pyd', '.so', '.egg-info'
    }
    
    # Directories to exclude
    exclude_dirs = {'__pycache__', '.git', '.venv', 'node_modules'}
    
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            file_count = 0
            
            for item in sorted(paper_dir.rglob('*')):
                # Skip directories themselves
                if item.is_dir():
                    continue
                
                # Skip excluded extensions
                if item.suffix in exclude_extensions:
                    continue
                
                # Skip excluded directories
                if any(exc_dir in item.parts for exc_dir in exclude_dirs):
                    continue
                
                # Compute archive name (relative to paper_dir parent)
                arcname = item.relative_to(paper_dir.parent)
                
                # Add to archive
                zf.write(item, arcname)
                print(f"  + {arcname}")
                file_count += 1
            
            print(f"\n[OK] Added {file_count} files to {output_path}")
    
    except Exception as e:
        print(f"\n[ERROR] Error creating bundle: {e}")
        return False
    
    # Verify ZIP is readable
    try:
        with zipfile.ZipFile(output_path, 'r') as zf:
            bad_files = zf.testzip()
            if bad_files:
                print(f"\n[ERROR] ZIP verification failed: corrupt file {bad_files}")
                return False

            names = set(zf.namelist())
            required_arch = "paper/figures/architecture.pdf"
            if required_arch not in names:
                print(f"\n[ERROR] ZIP missing canonical architecture file: {required_arch}")
                return False

            arch_entries = sorted(name for name in names if name.lower().endswith("architecture.pdf"))
            non_canonical = [name for name in arch_entries if name != required_arch]
            if non_canonical:
                print("\n[ERROR] ZIP contains non-canonical architecture.pdf entries:")
                for entry in non_canonical:
                    print(f"  - {entry}")
                return False
        print(f"[OK] ZIP integrity verified")
    except Exception as e:
        print(f"\n[ERROR] ZIP verification failed: {e}")
        return False
    
    print(f"\n{'=' * 60}")
    print(f"\n[OK] Overleaf bundle created successfully:")
    print(f"  {output_path.resolve()}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB\n")
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate paper directory exists
    if not args.paper_dir.exists():
        print(f"\n[ERROR] Paper directory not found: {args.paper_dir}")
        print("  Run this script from the repository root.")
        return 1
    
    # Validate all assets
    if not validate_paper_assets(args.paper_dir):
        print("\n[ERROR] Validation failed. Fix errors above and try again.\n")
        return 1
    
    # Exit if validate-only mode
    if args.validate_only:
        print("\n[OK] Validation complete (--validate-only mode)\n")
        return 0
    
    # Create bundle
    if not create_overleaf_bundle(args.paper_dir, args.out):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
