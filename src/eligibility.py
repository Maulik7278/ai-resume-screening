"""
Hard eligibility gate.

Deliberately deterministic and predictable: eligibility is decided from
evidence lists (deterministic regex extraction, unioned with any Cohere
semantic evidence), never from free-form LLM judgement alone. Cohere output
enriches the evidence available; it does not decide eligibility on its own.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str]


def evaluate_eligibility(
    python_evidence: list[str],
    ai_evidence: list[str],
) -> EligibilityResult:
    """
    Minimum requirement: at least one piece of Python evidence AND at least
    one piece of AI/LLM/Agentic evidence. Evidence lists are the union of
    deterministic (regex) extraction and any validated Cohere output.
    """
    has_python = len(python_evidence) > 0
    has_ai = len(ai_evidence) > 0

    reasons: list[str] = []
    if not has_python:
        reasons.append("No Python evidence found on resume")
    if not has_ai:
        reasons.append("No AI/LLM/Agentic evidence found on resume")

    eligible = has_python and has_ai
    if eligible:
        reasons.append("Meets minimum requirement: Python evidence + AI/LLM/Agentic evidence")

    return EligibilityResult(eligible=eligible, reasons=reasons)
