"""
Shared data models for the resume screening pipeline.

These models are used both for validating structured LLM output and for
representing the final per-candidate result written to results.json.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_AVAILABLE = "not_available"


# ---------------------------------------------------------------------------
# Structured Cohere output (validated before entering the scoring pipeline)
# ---------------------------------------------------------------------------

class ProjectEvidence(BaseModel):
    name: str = ""
    technologies: list[str] = Field(default_factory=list)
    description: str = ""
    depth: str = ""  # e.g. "shallow", "moderate", "deep"
    evidence: list[str] = Field(default_factory=list)
    shallow_wrapper: bool = False


class ResumeAnalysis(BaseModel):
    """The schema Cohere is asked to fill in, then validated on the way out."""

    candidate_name: Optional[str] = None
    email: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    github_url: Optional[str] = None
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    projects: list[ProjectEvidence] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("skills", "python_evidence", "ai_evidence", "strengths", "concerns", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        return v or []


# ---------------------------------------------------------------------------
# Deterministic extraction (regex-based, always available even without LLM)
# ---------------------------------------------------------------------------

class DeterministicExtraction(BaseModel):
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    github_url: Optional[str] = None
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GitHub enrichment
# ---------------------------------------------------------------------------

class GitHubProfile(BaseModel):
    status: Status
    username: Optional[str] = None
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    top_languages: list[str] = Field(default_factory=list)
    notable_repo_names: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    python_score: float = 0.0
    ai_score: float = 0.0
    project_quality_score: float = 0.0
    github_score: float = 0.0
    total_score: float = 0.0


# ---------------------------------------------------------------------------
# Final per-candidate result
# ---------------------------------------------------------------------------

class CandidateResult(BaseModel):
    rank: Optional[int] = None
    source_file: str
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    github_url: Optional[str] = None

    parse_status: Status = Status.NOT_AVAILABLE
    llm_status: Status = Status.NOT_AVAILABLE
    github_status: Status = Status.NOT_AVAILABLE

    eligible: bool = False
    eligibility_reasons: list[str] = Field(default_factory=list)

    score_breakdown: Optional[ScoreBreakdown] = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    summary: str = ""

    errors: list[str] = Field(default_factory=list)


class BatchSummary(BaseModel):
    total_discovered: int = 0
    successfully_parsed: int = 0
    eligible_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0


class BatchResult(BaseModel):
    summary: BatchSummary
    candidates: list[CandidateResult]
