from pathlib import Path

import pytest

from src.parser import ParseError, discover_resumes, parse_resume

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_resume_extracts_text():
    parsed = parse_resume(FIXTURES / "eligible_resume.pdf")
    assert "Jane Doe" in parsed.text
    assert parsed.source_file == "eligible_resume.pdf"
    assert parsed.page_count == 1


def test_parse_missing_file_raises_parse_error():
    with pytest.raises(ParseError):
        parse_resume(FIXTURES / "does_not_exist.pdf")


def test_parse_non_pdf_raises_parse_error(tmp_path):
    fake = tmp_path / "resume.txt"
    fake.write_text("not a pdf")
    with pytest.raises(ParseError):
        parse_resume(fake)


def test_parse_blank_pdf_raises_parse_error_no_extractable_text():
    with pytest.raises(ParseError):
        parse_resume(FIXTURES / "blank_resume.pdf")


def test_parse_corrupt_pdf_raises_parse_error(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 this is not a real pdf structure")
    with pytest.raises(ParseError):
        parse_resume(corrupt)


def test_discover_resumes_finds_only_pdfs(tmp_path):
    (tmp_path / "candidate_01.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "candidate_02.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "subdir").mkdir()

    found = discover_resumes(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["candidate_01.pdf", "candidate_02.pdf"]


def test_discover_resumes_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        discover_resumes(Path("/nonexistent/directory/for/sure"))


def test_discover_resumes_no_naming_pattern_assumed(tmp_path):
    # Filenames are arbitrary and must not be assumed to follow a pattern.
    (tmp_path / "resume_final_v2_ACTUAL.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "cv.pdf").write_bytes(b"%PDF-1.4")
    found = discover_resumes(tmp_path)
    assert len(found) == 2
