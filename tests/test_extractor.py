from src.extractor import extract_deterministic


def test_extracts_email():
    text = "Jane Doe\njane.doe@example.com\nExperience\nPython developer."
    result = extract_deterministic(text)
    assert result.email == "jane.doe@example.com"


def test_extracts_github_url():
    text = "John Smith\nhttps://github.com/johnsmith\nPython developer."
    result = extract_deterministic(text)
    assert result.github_url == "https://github.com/johnsmith"


def test_extracts_candidate_name_from_first_line():
    text = "Priya Sharma\npriya.sharma@example.com\nExperience\nBuilt Python APIs."
    result = extract_deterministic(text)
    assert result.candidate_name == "Priya Sharma"


def test_skips_heading_lines_when_looking_for_name():
    text = "Resume\nPriya Sharma\npriya.sharma@example.com\nExperience"
    result = extract_deterministic(text)
    assert result.candidate_name == "Priya Sharma"


def test_does_not_mistake_email_or_url_for_name():
    text = "priya.sharma@example.com\nhttps://github.com/priyasharma\nPriya Sharma\nExperience"
    result = extract_deterministic(text)
    assert result.candidate_name == "Priya Sharma"


def test_does_not_mistake_phone_number_for_name():
    text = "+1 (555) 123-4567\nPriya Sharma\nExperience"
    result = extract_deterministic(text)
    assert result.candidate_name == "Priya Sharma"


def test_returns_none_when_no_plausible_name_line_found():
    text = "PROFESSIONAL SUMMARY\nExperienced backend engineer with 5 years in industry."
    result = extract_deterministic(text)
    assert result.candidate_name is None


def test_python_and_ai_evidence_extraction():
    text = (
        "Jane Doe\n"
        "Built REST APIs using Python and FastAPI.\n"
        "Implemented a RAG pipeline using LangChain.\n"
        "Managed retail inventory."
    )
    result = extract_deterministic(text)
    assert any("python" in line.lower() for line in result.python_evidence)
    assert any("rag" in line.lower() or "langchain" in line.lower() for line in result.ai_evidence)
