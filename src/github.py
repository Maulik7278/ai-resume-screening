"""
GitHub profile enrichment.

Fetches basic public profile + repo info for a candidate's GitHub URL, when
present. Isolated behind a small class so failures (rate limits, 404s,
network errors) never abort resume processing -- they just result in a
GitHubProfile with status=FAILED or NOT_AVAILABLE.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from src.config import settings
from src.models import GitHubProfile, Status

USERNAME_RE = re.compile(r"github\.com/([A-Za-z0-9\-]+)/?", re.IGNORECASE)


def extract_username(github_url: str | None) -> str | None:
    if not github_url:
        return None
    match = USERNAME_RE.search(github_url)
    return match.group(1) if match else None


@dataclass
class GitHubClient:
    token: str | None = None
    timeout_seconds: float | None = None
    _cache: dict[str, GitHubProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.token = self.token if self.token is not None else settings.github_token
        self.timeout_seconds = self.timeout_seconds or settings.github_timeout_seconds

    def _headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def fetch_profile(self, github_url: str | None) -> GitHubProfile:
        if not github_url:
            return GitHubProfile(status=Status.NOT_AVAILABLE, error="No GitHub URL found on resume")

        username = extract_username(github_url)
        if not username:
            return GitHubProfile(status=Status.NOT_AVAILABLE, error=f"Could not parse username from {github_url}")

        # In-memory cache: avoid re-fetching the same profile within a run
        # (rare, but possible if two resumes share a URL, or on retries).
        if username in self._cache:
            return self._cache[username]

        try:
            profile_resp = requests.get(
                f"https://api.github.com/users/{username}",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            if profile_resp.status_code == 404:
                result = GitHubProfile(status=Status.FAILED, username=username, error="GitHub user not found")
                self._cache[username] = result
                return result
            profile_resp.raise_for_status()
            profile_data = profile_resp.json()

            repos_resp = requests.get(
                f"https://api.github.com/users/{username}/repos",
                headers=self._headers(),
                params={"per_page": 10, "sort": "updated"},
                timeout=self.timeout_seconds,
            )
            repos_data = repos_resp.json() if repos_resp.ok else []

            languages: list[str] = []
            repo_names: list[str] = []
            if isinstance(repos_data, list):
                for repo in repos_data[:10]:
                    lang = repo.get("language")
                    if lang and lang not in languages:
                        languages.append(lang)
                    name = repo.get("name")
                    if name:
                        repo_names.append(name)

            result = GitHubProfile(
                status=Status.SUCCESS,
                username=username,
                public_repos=profile_data.get("public_repos"),
                followers=profile_data.get("followers"),
                top_languages=languages,
                notable_repo_names=repo_names,
            )
        except requests.exceptions.RequestException as exc:
            result = GitHubProfile(status=Status.FAILED, username=username, error=str(exc))

        self._cache[username] = result
        return result
