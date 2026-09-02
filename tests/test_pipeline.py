from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.github import GitHubClient
from src.llm.cohere_provider import CohereProvider, LLMError
from src.models import GitHubProfile, ProjectEvidence, ResumeAnalysis, Status
from src.pipeline import process_single_resume, run_batch

FIXTURES = Path(__file__).parent / "fixtures"


def _disabled_cohere() -> CohereProvider:
    """
    A CohereProvider that is force-disabled regardless of environment state.

    CohereProvider(api_key=None) alone is NOT enough here: __post_init__
    falls back to whatever COHERE_API_KEY is set in the environment/.env,
    so if a real key is configured (e.g. for real runs against main.py),
    these unit tests would otherwise start making live API calls. We
    override api_key after construction so `enabled` is reliably False.
    """
    provider = CohereProvider(api_key=None)
    provider.api_key = None
    return provider


def _disabled_github() -> GitHubClient:
    return GitHubClient(token=None)


# ---------------------------------------------------------------------
# process_single_resume — no LLM configured
# ---------------------------------------------------------------------

def test_process_single_resume_without_cohere_uses_deterministic_extraction():
    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", _disabled_cohere(), _disabled_github(), {}
    )
    assert result.parse_status == Status.SUCCESS
    assert result.llm_status == Status.NOT_AVAILABLE
    assert result.eligible is True
    assert result.github_url == "https://github.com/janedoe"


def test_process_single_resume_without_cohere_populates_fallback_narrative():
    # Regression test: strengths/concerns/summary must not be silently
    # empty just because Cohere isn't configured -- they should be
    # populated from deterministic evidence instead.
    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", _disabled_cohere(), _disabled_github(), {}
    )
    assert result.strengths  # non-empty
    assert result.concerns  # non-empty (at least the "LLM unavailable" note)
    assert result.summary  # non-empty
    assert any("Python" in s for s in result.strengths)
    assert any("AI/LLM" in s for s in result.strengths)


def test_process_single_resume_ineligible_when_missing_ai_evidence():
    result = process_single_resume(
        FIXTURES / "python_only_resume.pdf", _disabled_cohere(), _disabled_github(), {}
    )
    assert result.eligible is False
    assert any("AI" in reason for reason in result.eligibility_reasons)


def test_process_single_resume_ineligible_when_no_relevant_evidence():
    result = process_single_resume(
        FIXTURES / "unrelated_resume.pdf", _disabled_cohere(), _disabled_github(), {}
    )
    assert result.eligible is False
    assert len(result.eligibility_reasons) == 2


# ---------------------------------------------------------------------
# Parser failure isolation
# ---------------------------------------------------------------------

def test_process_single_resume_parse_failure_is_isolated():
    result = process_single_resume(
        FIXTURES / "blank_resume.pdf", _disabled_cohere(), _disabled_github(), {}
    )
    assert result.parse_status == Status.FAILED
    assert result.eligible is False
    assert result.errors  # error recorded, no exception raised


# ---------------------------------------------------------------------
# Cohere failure isolation
# ---------------------------------------------------------------------

def test_process_single_resume_cohere_failure_falls_back_to_deterministic():
    mock_provider = MagicMock(spec=CohereProvider)
    mock_provider.enabled = True
    mock_provider.analyze_resume.side_effect = LLMError("simulated Cohere outage")

    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", mock_provider, _disabled_github(), {}
    )

    assert result.llm_status == Status.FAILED
    assert any("LLM processing failure" in e for e in result.errors)
    # Deterministic extraction still found Python + AI evidence -> still eligible.
    assert result.eligible is True
    # Regression check: candidate_name must come from the resume text itself
    # (deterministic extraction) when Cohere fails -- NOT from the filename.
    # eligible_resume.pdf's first line is "Jane Doe".
    assert result.candidate_name == "Jane Doe"
    assert result.candidate_name != "Eligible Resume"
    # Regression check: strengths/concerns/summary must still be populated
    # (from deterministic evidence) even though the LLM call itself failed --
    # this mirrors a real Cohere outage/404/429 scenario, not just the
    # "never configured" case.
    assert result.strengths
    assert result.concerns
    assert result.summary


def test_process_single_resume_cohere_success_enriches_result():
    mock_provider = MagicMock(spec=CohereProvider)
    mock_provider.enabled = True
    mock_provider.analyze_resume.return_value = ResumeAnalysis(
        candidate_name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python", "LangChain"],
        python_evidence=["Extra python evidence from LLM"],
        ai_evidence=["Extra ai evidence from LLM"],
        projects=[ProjectEvidence(name="RAG bot", depth="deep", shallow_wrapper=False)],
        strengths=["Strong Python background"],
        concerns=["Limited production experience"],
        summary="Solid backend + AI candidate.",
    )

    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", mock_provider, _disabled_github(), {}
    )

    assert result.llm_status == Status.SUCCESS
    assert result.candidate_name == "Jane Doe"
    assert result.strengths == ["Strong Python background"]
    assert result.score_breakdown.project_quality_score > 0


# ---------------------------------------------------------------------
# GitHub failure isolation
# ---------------------------------------------------------------------

def test_process_single_resume_github_failure_is_isolated_not_fatal():
    mock_github = MagicMock(spec=GitHubClient)
    mock_github.fetch_profile.return_value = GitHubProfile(
        status=Status.FAILED, username="janedoe", error="simulated network error"
    )

    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", _disabled_cohere(), mock_github, {}
    )

    assert result.github_status == Status.FAILED
    assert any("GitHub lookup failed" in e for e in result.errors)
    # GitHub failure must not affect eligibility.
    assert result.eligible is True


def test_process_single_resume_github_exception_does_not_propagate():
    mock_github = MagicMock(spec=GitHubClient)
    mock_github.fetch_profile.side_effect = RuntimeError("boom")

    # Must not raise -- failure is caught and recorded.
    result = process_single_resume(
        FIXTURES / "eligible_resume.pdf", _disabled_cohere(), mock_github, {}
    )
    assert result.github_status == Status.FAILED
    assert any("Unexpected GitHub error" in e for e in result.errors)


# ---------------------------------------------------------------------
# Batch-level: per-resume isolation + ranking
# ---------------------------------------------------------------------

def test_run_batch_processes_all_and_isolates_failures(tmp_path):
    for fixture in ["eligible_resume.pdf", "python_only_resume.pdf", "unrelated_resume.pdf", "blank_resume.pdf"]:
        (tmp_path / fixture).write_bytes((FIXTURES / fixture).read_bytes())

    output_path = tmp_path / "output" / "results.json"
    batch = run_batch(
        tmp_path,
        output_path,
        concurrency=2,
        cohere_provider=_disabled_cohere(),
        github_client=_disabled_github(),
    )

    assert batch.summary.total_discovered == 4
    assert batch.summary.failed_count == 1  # blank_resume.pdf
    assert batch.summary.successfully_parsed == 3
    assert batch.summary.eligible_count == 1  # only eligible_resume.pdf
    assert output_path.exists()

    filenames = {c.source_file for c in batch.candidates}
    assert filenames == {
        "eligible_resume.pdf", "python_only_resume.pdf", "unrelated_resume.pdf", "blank_resume.pdf"
    }


def test_run_batch_ranks_eligible_candidates_first_by_score(tmp_path):
    for fixture in ["eligible_resume.pdf", "python_only_resume.pdf"]:
        (tmp_path / fixture).write_bytes((FIXTURES / fixture).read_bytes())

    output_path = tmp_path / "results.json"
    batch = run_batch(
        tmp_path,
        output_path,
        concurrency=2,
        cohere_provider=_disabled_cohere(),
        github_client=_disabled_github(),
    )

    assert batch.candidates[0].source_file == "eligible_resume.pdf"
    assert batch.candidates[0].eligible is True
    assert batch.candidates[0].rank == 1
    assert batch.candidates[-1].rank == len(batch.candidates)
    # Ranks are contiguous and ascending.
    ranks = [c.rank for c in batch.candidates]
    assert ranks == sorted(ranks)


def test_run_batch_one_resume_cohere_failure_does_not_abort_batch(tmp_path):
    for fixture in ["eligible_resume.pdf", "python_only_resume.pdf"]:
        (tmp_path / fixture).write_bytes((FIXTURES / fixture).read_bytes())

    flaky_provider = MagicMock(spec=CohereProvider)
    flaky_provider.enabled = True

    def side_effect(text, source_file=""):
        if "eligible_resume" in source_file:
            raise LLMError("simulated failure for this one resume")
        return ResumeAnalysis(candidate_name="John Smith")

    flaky_provider.analyze_resume.side_effect = side_effect

    output_path = tmp_path / "results.json"
    batch = run_batch(
        tmp_path,
        output_path,
        concurrency=2,
        cohere_provider=flaky_provider,
        github_client=_disabled_github(),
    )

    # Both resumes still show up in results -- the LLM failure for one
    # resume did not abort processing of the other.
    assert batch.summary.total_discovered == 2
    statuses = {c.source_file: c.llm_status for c in batch.candidates}
    assert statuses["eligible_resume.pdf"] == Status.FAILED
    assert statuses["python_only_resume.pdf"] == Status.SUCCESS


def test_run_batch_empty_directory_produces_empty_but_valid_result(tmp_path):
    output_path = tmp_path / "results.json"
    batch = run_batch(
        tmp_path, output_path, concurrency=1, cohere_provider=_disabled_cohere(), github_client=_disabled_github()
    )
    assert batch.summary.total_discovered == 0
    assert batch.candidates == []
    assert output_path.exists()


def test_run_batch_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_batch(
            tmp_path / "does_not_exist",
            tmp_path / "results.json",
            cohere_provider=_disabled_cohere(),
            github_client=_disabled_github(),
        )
