from src.eligibility import evaluate_eligibility


def test_eligible_with_both_python_and_ai_evidence():
    result = evaluate_eligibility(
        python_evidence=["Built REST APIs with Python and FastAPI"],
        ai_evidence=["Implemented a RAG pipeline using LangChain"],
    )
    assert result.eligible is True
    assert any("Meets minimum requirement" in r for r in result.reasons)


def test_ineligible_missing_python_evidence():
    result = evaluate_eligibility(python_evidence=[], ai_evidence=["Trained a neural network"])
    assert result.eligible is False
    assert any("No Python evidence" in r for r in result.reasons)


def test_ineligible_missing_ai_evidence():
    result = evaluate_eligibility(python_evidence=["Wrote Python scripts for ETL"], ai_evidence=[])
    assert result.eligible is False
    assert any("No AI/LLM/Agentic evidence" in r for r in result.reasons)


def test_ineligible_missing_both():
    result = evaluate_eligibility(python_evidence=[], ai_evidence=[])
    assert result.eligible is False
    assert len(result.reasons) == 2


def test_eligibility_ignores_evidence_source_only_cares_about_presence():
    # Whether evidence came from deterministic regex extraction or from
    # Cohere is irrelevant to this function -- it just needs non-empty lists.
    result = evaluate_eligibility(python_evidence=["a"], ai_evidence=["b", "c", "d"])
    assert result.eligible is True
