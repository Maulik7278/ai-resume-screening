"""
Deterministic (regex/keyword based) extraction from resume text.

This runs independently of any LLM call, so the hard eligibility decision
can be computed even when Cohere is unavailable, times out, or fails for a
particular resume. Cohere output (see llm/cohere_provider.py) supplements
this with richer semantic evidence and project analysis, but never replaces
it for the core eligibility gate.
"""
from __future__ import annotations

import re

from src.models import DeterministicExtraction


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-]+)", re.IGNORECASE)
PHONE_RE = re.compile(r"[+\d][\d\s().-]{7,}\d")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
# A resume "name" line: 2-4 words, letters/periods/hyphens/apostrophes only,
# no digits, reasonably short. Deliberately conservative so we don't grab
# a section heading like "Professional Summary" by accident.
NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){1,3}$")
NAME_BLOCKLIST = {
    "resume", "curriculum vitae", "cv", "profile", "summary", "objective",
    "professional summary", "contact", "contact information", "personal details",
    "education", "experience", "skills", "projects", "about me",
}


def _looks_like_name(line: str) -> bool:
    stripped = line.strip()
    if not (4 <= len(stripped) <= 40):
        return False
    if stripped.lower() in NAME_BLOCKLIST:
        return False
    if EMAIL_RE.search(stripped) or URL_RE.search(stripped) or PHONE_RE.search(stripped):
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    if not NAME_LINE_RE.match(stripped):
        return False
    return True


def _extract_candidate_name(text: str) -> Optional[str]:
    """
    Heuristic: resumes conventionally lead with the candidate's name as the
    very first meaningful line. Scan the first ~10 non-empty lines and take
    the first one that looks like a plausible name (not a heading, contact
    detail, or URL). This is a fallback used when Cohere is unavailable or
    fails for a given resume -- Cohere's own name extraction is preferred
    when it succeeds.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:10]:
        if _looks_like_name(line):
            return line
    return None


# Keyword sets used to gather deterministic "evidence" lines. Kept simple and
# transparent on purpose -- the hard eligibility gate should be easy to audit.
PYTHON_KEYWORDS = [
    "python", "django", "flask", "fastapi", "pandas", "numpy", "pytest",
    "pyspark", "scikit-learn", "sklearn",
]
AI_KEYWORDS = [
    "machine learning", "deep learning", "llm", "large language model",
    "gpt", "openai", "cohere", "anthropic", "claude", "langchain",
    "transformer", "nlp", "natural language processing", "agentic",
    "agent framework", "rag", "retrieval augmented generation",
    "tensorflow", "pytorch", "artificial intelligence", " ai ", "ai/ml",
    "generative ai", "computer vision", "neural network",
]


def _find_evidence_lines(text: str, keywords: list[str]) -> list[str]:
    """Return distinct lines from `text` that mention any of `keywords`."""
    evidence: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = f" {line.lower()} "
        if any(kw in lowered for kw in keywords):
            key = line.lower()
            if key not in seen:
                seen.add(key)
                evidence.append(line)
    return evidence


def extract_deterministic(text: str) -> DeterministicExtraction:
    email_match = EMAIL_RE.search(text)
    github_match = GITHUB_RE.search(text)

    github_url = None
    if github_match:
        github_url = f"https://github.com/{github_match.group(1)}"

    return DeterministicExtraction(
        candidate_name=_extract_candidate_name(text),
        email=email_match.group(0) if email_match else None,
        github_url=github_url,
        python_evidence=_find_evidence_lines(text, PYTHON_KEYWORDS),
        ai_evidence=_find_evidence_lines(text, AI_KEYWORDS),
    )
