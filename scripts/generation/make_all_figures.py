"""
Master script to generate all three CalibraTeach figures.
Runs: make_architecture.py, make_reliability.py, make_acc_coverage.py
Outputs: figures/architecture.pdf, figures/reliability.pdf, figures/acc_coverage.pdf
"""

import subprocess
import sys
import os

def run_figure_scripts():
    """Run all figure generation scripts in sequence."""
    
    scripts = [
        'scripts/make_architecture.py',
        'scripts/make_reliability.py',
        'scripts/make_acc_coverage.py',
    ]
    
    print("=" * 70)
    print("CalibraTeach Figure Generation Suite")
    print("=" * 70)
    print()
    
    failed = []
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"[FAIL] Script not found: {script}")
            failed.append(script)
            continue
        
        print(f"Running: {script}")
        print("-" * 70)
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)
            
            if result.returncode != 0:
                print(f"[FAIL] Script failed with exit code {result.returncode}")
                failed.append(script)
            else:
                print(f"[OK] {script} completed successfully")
        except subprocess.TimeoutExpired:
            print(f"[FAIL] Script timed out: {script}")
            failed.append(script)
        except Exception as e:
            print(f"[FAIL] Error running {script}: {e}")
            failed.append(script)
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if os.path.exists('figures/architecture.pdf'):
        print("[OK] figures/architecture.pdf")
    else:
        print("[MISS] figures/architecture.pdf (missing)")
    
    if os.path.exists('figures/reliability.pdf'):
        print("[OK] figures/reliability.pdf")
    else:
        print("[MISS] figures/reliability.pdf (missing)")
    
    if os.path.exists('figures/acc_coverage.pdf'):
        print("[OK] figures/acc_coverage.pdf")
    else:
        print("[MISS] figures/acc_coverage.pdf (missing)")
    
    print()
    
    if not failed:
        print("[OK] All figures generated successfully!")
        print()
        print("Next steps:")
        print("1. Update submission_bundle/OVERLEAF_TEMPLATE.tex to use \\includegraphics")
        print("2. Compile with: pdflatex OVERLEAF_TEMPLATE.tex")
        return 0
    else:
        print(f"[FAIL] {len(failed)} script(s) failed or had issues")
        return 1

if __name__ == '__main__':
    sys.exit(run_figure_scripts())
