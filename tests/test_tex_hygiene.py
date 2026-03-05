from pathlib import Path

from scripts.check_tex_artifacts_hygiene import scan_file


def test_detects_banned_codepoint_soft_hyphen(tmp_path: Path):
    target = tmp_path / "bad.tex"
    target.write_text("value \u00adhere", encoding="utf-8")

    issues = scan_file(target)

    assert any(issue.get("codepoint") == "U+00AD" for issue in issues)


def test_flags_unescaped_percent(tmp_path: Path):
    target = tmp_path / "percent.tex"
    target.write_text("value 80.0%", encoding="utf-8")

    issues = scan_file(target)

    assert any(issue.get("detail") == "unescaped %" for issue in issues)


def test_allows_escaped_percent(tmp_path: Path):
    target = tmp_path / "clean.tex"
    target.write_text(r"\newcommand{\Test}{80.00\%}", encoding="utf-8")

    issues = scan_file(target)

    assert issues == []
