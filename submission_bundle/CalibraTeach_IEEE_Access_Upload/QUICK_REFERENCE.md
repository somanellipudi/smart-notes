# 🚀 Quick Reference: IEEE Access Submission Verification

**One command to verify everything**:
```bash
python verify.py
```

---

## Essential Commands

| What | Command | When to Use |
|------|---------|-------------|
| **Verify all** | `python verify.py` | Before submission |
| **Auto-fix + verify** | `python verify.py --fix` | After copy/paste from Word/Docs |
| **Quick check** | `python verify.py --skip-build` | Before committing code |
| **Check Unicode** | `python scripts/sanitize_unicode.py --check` | After editing .tex files |
| **Fix Unicode** | `python scripts/sanitize_unicode.py --fix` | When Unicode issues found |
| **Verify PDF** | `python scripts/verify_pdf_text.py` | After manual PDF build |
| **Check integrity** | `python scripts/verify_submission.py OVERLEAF_TEMPLATE.tex` | Verify refs/metrics |

---

## Make Commands (Alternative)

```bash
make verify       # Run all checks
make verify-fix   # Auto-fix + verify
make clean        # Remove temp files
make help         # Show all commands
```

---

## Expected Success Output

```
======================================================================
VERIFICATION SUMMARY
======================================================================
✓ ALL CHECKS PASSED (4/4)

✅ READY FOR IEEE ACCESS SUBMISSION
```

---

## If Checks Fail

1. **Unicode issues**: Run `python verify.py --fix`
2. **Missing files**: Check `figures/` directory exists
3. **PDF artifacts**: Run `python scripts/sanitize_unicode.py --fix` then rebuild
4. **Reference errors**: Check all `\label{}` exist for every `\ref{}`

---

## Dependencies

**Linux**: `sudo apt-get install texlive-full poppler-utils`  
**macOS**: `brew install mactex poppler`  
**Windows**: Install MiKTeX + poppler-windows

---

## Documentation

- **Complete guide**: See `VERIFICATION_INFRASTRUCTURE.md`
- **Submission guide**: See `SUBMISSION.md`
- **Implementation summary**: See `IMPLEMENTATION_SUMMARY.md`

---

**Status**: ✅ All scripts tested and working  
**Next**: Run `python verify.py` → Upload to Overleaf → Submit
