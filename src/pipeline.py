"""
Batch pipeline orchestration.

Discovers PDFs, processes each one independently (parse -> deterministic
extraction -> Cohere analysis -> GitHub enrichment -> eligibility ->
scoring), and assembles a ranked BatchResult. A failure in any single stage
for any single resume is recorded on that resume's result and never aborts
the rest of the batch.
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import settings
from src.eligibility import evaluate_eligibility
from src.extractor import extract_deterministic
from src.github import GitHubClient
from src.llm.cohere_provider import CohereProvider, LLMError, LLMNotConfigured
from src.models import (
    BatchResult,
    BatchSummary,
    CandidateResult,
    ResumeAnalysis,
    ScoreBreakdown,
    Status,
)
from src.parser import ParseError, discover_resumes, parse_resume
from src.scorer import score_candidate

logger = logging.getLogger(__name__)


def _merge_evidence(*lists: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for lst in lists:
        for item in lst:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item.strip())
    return merged


def _build_fallback_narrative(
    python_evidence: list[str],
    ai_evidence: list[str],
    github_profile,
    eligible: bool,
) -> tuple[list[str], list[str], str]:
    """
    Deterministic strengths/concerns/summary, built from evidence already
    gathered (regex extraction + GitHub enrichment), used only when Cohere
    did not produce its own analysis for this resume (not configured, or
    failed after retries). This keeps every candidate's output populated
    and useful even on an LLM outage, instead of leaving these fields
    silently empty with no explanation of why.
    """
    strengths: list[str] = []
    concerns: list[str] = []

    if python_evidence:
        strengths.append(f"Python evidence found ({len(python_evidence)} supporting line(s) on resume)")
    else:
        concerns.append("No explicit Python evidence detected on resume")

    if ai_evidence:
        strengths.append(f"AI/LLM/Agentic evidence found ({len(ai_evidence)} supporting line(s) on resume)")
    else:
        concerns.append("No explicit AI/LLM/Agentic evidence detected on resume")

    if github_profile is not None and github_profile.status == Status.SUCCESS:
        repo_count = github_profile.public_repos or 0
        if repo_count > 0:
            strengths.append(f"Active GitHub profile with {repo_count} public repositories")
        if github_profile.top_languages:
            strengths.append(f"GitHub activity in: {', '.join(github_profile.top_languages)}")
    else:
        concerns.append("GitHub profile could not be verified")

    concerns.append(
        "Project depth/quality and skills could not be semantically analyzed "
        "(LLM analysis unavailable for this resume — see errors)"
    )

    summary = (
        "Meets minimum Python + AI/LLM keyword requirements based on deterministic extraction. "
        if eligible
        else "Does not meet minimum Python + AI/LLM keyword requirements based on deterministic extraction. "
    ) + "Deeper semantic project analysis was unavailable for this resume — see errors for details."

    return strengths, concerns, summary


def process_single_resume(
    path: Path,
    cohere_provider: CohereProvider,
    github_client: GitHubClient,
    llm_cache: dict[str, ResumeAnalysis],
) -> CandidateResult:
    """Process one resume end-to-end. Never raises -- all failures are recorded."""
    result = CandidateResult(source_file=path.name)

    # --- 1. Parse PDF -------------------------------------------------
    try:
        parsed = parse_resume(path)
        result.parse_status = Status.SUCCESS
    except ParseError as exc:
        result.parse_status = Status.FAILED
        result.errors.append(f"Parse error: {exc}")
        result.eligible = False
        result.eligibility_reasons = ["Resume could not be parsed"]
        return result
    except Exception as exc:  # noqa: BLE001 - unexpected parser failure, isolate it
        result.parse_status = Status.FAILED
        result.errors.append(f"Unexpected parse error: {exc}")
        result.eligible = False
        result.eligibility_reasons = ["Resume could not be parsed"]
        return result

    # --- 2. Deterministic extraction (always available) ---------------
    deterministic = extract_deterministic(parsed.text)
    result.email = deterministic.email
    result.github_url = deterministic.github_url
    result.candidate_name = deterministic.candidate_name

    # --- 3. Cohere semantic analysis (best-effort) ---------------------
    analysis: ResumeAnalysis | None = None
    if not cohere_provider.enabled:
        result.llm_status = Status.NOT_AVAILABLE
        result.errors.append("Cohere not configured (no COHERE_API_KEY) — using deterministic extraction only")
    else:
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        if content_hash in llm_cache:
            analysis = llm_cache[content_hash]
            result.llm_status = Status.SUCCESS
        else:
            try:
                analysis = cohere_provider.analyze_resume(parsed.text, source_file=path.name)
                result.llm_status = Status.SUCCESS
                llm_cache[content_hash] = analysis
            except LLMNotConfigured as exc:
                result.llm_status = Status.NOT_AVAILABLE
                result.errors.append(f"LLM not available: {exc}")
            except LLMError as exc:
                result.llm_status = Status.FAILED
                result.errors.append(f"LLM processing failure: {exc}")
            except Exception as exc:  # noqa: BLE001 - never let LLM issues kill the batch
                result.llm_status = Status.FAILED
                result.errors.append(f"Unexpected LLM error: {exc}")

    if analysis is not None:
        if analysis.candidate_name:
            result.candidate_name = analysis.candidate_name
        if analysis.email:
            result.email = result.email or analysis.email
        if analysis.github_url:
            result.github_url = result.github_url or analysis.github_url
        result.strengths = analysis.strengths
        result.concerns = analysis.concerns
        result.summary = analysis.summary

    # Fallback candidate name: filename stem, cleaned up a little. Only
    # used if neither Cohere nor deterministic text extraction found a name.
    if not result.candidate_name:
        result.candidate_name = path.stem.replace("_", " ").replace("-", " ").strip().title() or None

    python_evidence = _merge_evidence(deterministic.python_evidence, analysis.python_evidence if analysis else [])
    ai_evidence = _merge_evidence(deterministic.ai_evidence, analysis.ai_evidence if analysis else [])
    projects = analysis.projects if analysis else []

    # --- 4. Hard eligibility gate (deterministic) -----------------------
    eligibility = evaluate_eligibility(python_evidence, ai_evidence)
    result.eligible = eligibility.eligible
    result.eligibility_reasons = eligibility.reasons

    # --- 5. GitHub enrichment (best-effort, only if we have a URL) ------
    github_profile = None
    if result.github_url:
        try:
            github_profile = github_client.fetch_profile(result.github_url)
            result.github_status = github_profile.status
            if github_profile.status == Status.FAILED and github_profile.error:
                result.errors.append(f"GitHub lookup failed: {github_profile.error}")
        except Exception as exc:  # noqa: BLE001 - GitHub issues must never kill the batch
            result.github_status = Status.FAILED
            result.errors.append(f"Unexpected GitHub error: {exc}")
    else:
        result.github_status = Status.NOT_AVAILABLE

    # --- 6. Fallback narrative (only if Cohere didn't supply one) -------
    # Ensures strengths/concerns/summary are always populated with
    # something useful, sourced from resume evidence, rather than left
    # empty whenever the LLM step didn't succeed for this resume.
    if analysis is None:
        result.strengths, result.concerns, result.summary = _build_fallback_narrative(
            python_evidence, ai_evidence, github_profile, result.eligible
        )

    # --- 7. Scoring (only meaningful for eligible candidates, but we
    #        compute it regardless so the JSON is consistent/auditable) --
    result.score_breakdown = score_candidate(python_evidence, ai_evidence, projects, github_profile)

    return result


def run_batch(
    input_dir: Path,
    output_path: Path,
    *,
    concurrency: int | None = None,
    cohere_provider: CohereProvider | None = None,
    github_client: GitHubClient | None = None,
) -> BatchResult:
    concurrency = concurrency or settings.max_workers
    cohere_provider = cohere_provider or CohereProvider()
    github_client = github_client or GitHubClient()
    llm_cache: dict[str, ResumeAnalysis] = {}

    resume_paths = discover_resumes(input_dir)
    results: list[CandidateResult] = []

    if not resume_paths:
        logger.warning("No PDF resumes found in %s", input_dir)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_path = {
            executor.submit(
                process_single_resume, path, cohere_provider, github_client, llm_cache
            ): path
            for path in resume_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - absolute last-resort safety net
                # process_single_resume is designed to never raise, but if a
                # truly unexpected bug slips through, isolate it here too
                # rather than losing the whole batch.
                results.append(
                    CandidateResult(
                        source_file=path.name,
                        parse_status=Status.FAILED,
                        eligible=False,
                        eligibility_reasons=["Unhandled processing error"],
                        errors=[f"Unhandled exception: {exc}"],
                    )
                )

    # Keep output order stable/deterministic regardless of thread completion order.
    results.sort(key=lambda r: r.source_file)

    # Rank: eligible candidates first, sorted by total_score descending.
    def sort_key(r: CandidateResult) -> tuple:
        score = r.score_breakdown.total_score if r.score_breakdown else 0.0
        return (0 if r.eligible else 1, -score)

    ranked = sorted(results, key=sort_key)
    rank = 0
    for r in ranked:
        rank += 1
        r.rank = rank

    summary = BatchSummary(
        total_discovered=len(resume_paths),
        successfully_parsed=sum(1 for r in results if r.parse_status == Status.SUCCESS),
        eligible_count=sum(1 for r in results if r.eligible),
        rejected_count=sum(1 for r in results if r.parse_status == Status.SUCCESS and not r.eligible),
        failed_count=sum(1 for r in results if r.parse_status == Status.FAILED),
    )

    batch_result = BatchResult(summary=summary, candidates=ranked)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(batch_result.model_dump_json(indent=2), encoding="utf-8")

    return batch_result
