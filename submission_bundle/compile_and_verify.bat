
@echo off
REM FINAL IEEE ACCESS MANUSCRIPT COMPILATION SCRIPT
REM Purpose: Compile LaTeX with verification of all fixes

setlocal enabledelayedexpansion

set TEXFILE=submission_bundle\CalibraTeach_IEEE_Access_Upload\OVERLEAF_TEMPLATE.tex
set OUTDIR=submission_bundle\CalibraTeach_IEEE_Access_Upload
set PDFFILE=submission_bundle\CalibraTeach_IEEE_Access_Upload\OVERLEAF_TEMPLATE.pdf

REM Try different LaTeX engines
for %%E in (pdflatex latexmk lualatex xelatex) do (
    where /q %%E
    if !errorlevel! equ 0 (
        echo Found LaTeX engine: %%E
        cd /d %OUTDIR%
        echo Compiling LaTeX source...
        %%E -interaction=nonstopmode -halt-on-error "%TEXFILE%"
        if !errorlevel! equ 0 (
            echo Compilation successful - PDF created
            goto :verify
        ) else (
            echo Compilation failed with %%E
        )
    )
)

echo.
echo ERROR: No working LaTeX engine found
echo.
echo REQUIRED: Install one of these:
echo   - MikTeX: choco install miktex
echo   - TeX Live: choco install texlive
echo   - Tectonic: cargo install tectonic
echo.
exit /b 1

:verify
echo.
echo VERIFICATION CHECKS:
echo.
REM Check 1: Source grep verification
echo [TEST 1] Searching for forbidden strings in TeX source...
findstr /M "Sec. III-B" "%TEXFILE%" >nul
if !errorlevel! equ 0 (
    echo FAIL: Found "Sec. III-B" in source
    exit /b 1
) else (
    echo PASS: No "Sec. III-B" in source
)

findstr /M "Eq. (6), Section III-B" "%TEXFILE%" >nul
if !errorlevel! equ 0 (
    echo FAIL: Found "Eq. (6), Section III-B" in source
    exit /b 1
) else (
    echo PASS: No "Eq. (6), Section III-B" in source
)

echo.
echo SUCCESS: All verifications passed
