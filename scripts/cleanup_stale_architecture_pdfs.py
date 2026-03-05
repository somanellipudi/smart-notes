#!/usr/bin/env python3
"""
Delete stale architecture.pdf copies, keeping only the canonical one.

Canonical path: paper/figures/architecture.pdf
(Matches: paper/main.tex includes figures/architecture.pdf)

Safe cleanup:
- Only deletes if canonical file exists
- After deletion, asserts exactly ONE architecture.pdf remains
"""

import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent
    canonical_path = repo_root / "paper" / "figures" / "architecture.pdf"
    
    # Step 1: Verify canonical exists
    if not canonical_path.exists():
        print(f"[ERROR] Canonical architecture.pdf not found at: {canonical_path}")
        print(f"[ERROR] Cannot safely delete stale copies without canonical present")
        return False
    
    print(f"[OK] Canonical architecture.pdf found at: {canonical_path}")
    
    # Step 2: Find all architecture.pdf files
    all_copies = list(repo_root.rglob("architecture.pdf"))
    print(f"[INFO] Found {len(all_copies)} total copies of architecture.pdf")
    
    # Step 3: Delete stale copies (all except canonical)
    stale_copies = [p for p in all_copies if p != canonical_path]
    
    for stale in stale_copies:
        try:
            print(f"[DELETE] Removing stale copy: {stale}")
            stale.unlink()
            print(f"[OK] Deleted: {stale}")
        except Exception as e:
            print(f"[ERROR] Failed to delete {stale}: {e}", file=sys.stderr)
            return False
    
    # Step 4: Verify exactly ONE remains
    remaining = list(repo_root.rglob("architecture.pdf"))
    print(f"\n[INFO] After cleanup: {len(remaining)} copy/copies remaining")
    
    if len(remaining) != 1:
        print(f"[ERROR] Expected exactly 1 architecture.pdf, found {len(remaining)}", file=sys.stderr)
        for r in remaining:
            print(f"       {r}")
        return False
    
    if remaining[0] != canonical_path:
        print(f"[ERROR] Remaining file is not canonical:", file=sys.stderr)
        print(f"       Expected: {canonical_path}")
        print(f"       Found:    {remaining[0]}")
        return False
    
    print(f"[OK] Cleanup successful: exactly ONE canonical architecture.pdf remains")
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
