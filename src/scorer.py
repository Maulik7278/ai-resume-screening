"""
Candidate scoring.

Only applied to candidates that already pass the deterministic eligibility
gate (see eligibility.py). Combines evidence volume, project quality
(from Cohere analysis, when available), and GitHub activity into a single
comparable score for ranking.
"""
from __future__ import annotations

from src.models import GitHubProfile, ProjectEvidence, ScoreBreakdown, Status

# Weights are intentionally simple and documented here so they're easy to
# tune without hunting through the pipeline.
MAX_PYTHON_SCORE = 20.0
MAX_AI_SCORE = 20.0
MAX_PROJECT_SCORE = 40.0
MAX_GITHUB_SCORE = 20.0

DEPTH_POINTS = {"deep": 1.0, "moderate": 0.6, "shallow": 0.2}


def _evidence_score(evidence_count: int, max_score: float, saturation: int = 5) -> float:
    """Diminishing-returns score: more evidence helps, but caps out."""
    if evidence_count <= 0:
        return 0.0
    return round(min(evidence_count / saturation, 1.0) * max_score, 2)


def _project_quality_score(projects: list[ProjectEvidence]) -> float:
    if not projects:
        return 0.0
    total = 0.0
    for project in projects:
        depth_key = (project.depth or "").strip().lower()
        points = DEPTH_POINTS.get(depth_key, 0.4)  # unknown depth -> middling credit
        if project.shallow_wrapper:
            points *= 0.4  # penalize thin API wrappers even if labeled otherwise
        total += points
    # Average per-project quality, scaled to MAX_PROJECT_SCORE, with a small
    # bonus (capped) for having multiple substantive projects.
    avg = total / len(projects)
    project_count_bonus = min(len(projects), 4) / 4  # 0.25 .. 1.0
    score = avg * MAX_PROJECT_SCORE * (0.7 + 0.3 * project_count_bonus)
    return round(min(score, MAX_PROJECT_SCORE), 2)


def _github_score(profile: GitHubProfile | None) -> float:
    if profile is None or profile.status != Status.SUCCESS:
        return 0.0
    repo_points = min((profile.public_repos or 0) / 10, 1.0) * (MAX_GITHUB_SCORE * 0.6)
    follower_points = min((profile.followers or 0) / 20, 1.0) * (MAX_GITHUB_SCORE * 0.2)
    language_points = min(len(profile.top_languages) / 3, 1.0) * (MAX_GITHUB_SCORE * 0.2)
    return round(repo_points + follower_points + language_points, 2)


def score_candidate(
    python_evidence: list[str],
    ai_evidence: list[str],
    projects: list[ProjectEvidence],
    github_profile: GitHubProfile | None,
) -> ScoreBreakdown:
    python_score = _evidence_score(len(python_evidence), MAX_PYTHON_SCORE)
    ai_score = _evidence_score(len(ai_evidence), MAX_AI_SCORE)
    project_score = _project_quality_score(projects)
    github_score = _github_score(github_profile)

    total = round(python_score + ai_score + project_score + github_score, 2)

    return ScoreBreakdown(
        python_score=python_score,
        ai_score=ai_score,
        project_quality_score=project_score,
        github_score=github_score,
        total_score=total,
    )
