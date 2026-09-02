from src.models import GitHubProfile, ProjectEvidence, Status
from src.scorer import score_candidate


def test_no_evidence_no_projects_no_github_scores_zero():
    breakdown = score_candidate([], [], [], None)
    assert breakdown.total_score == 0.0
    assert breakdown.python_score == 0.0
    assert breakdown.ai_score == 0.0
    assert breakdown.project_quality_score == 0.0
    assert breakdown.github_score == 0.0


def test_more_evidence_increases_score_up_to_saturation():
    low = score_candidate(["a"], ["b"], [], None)
    high = score_candidate(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"], [], None)
    assert high.python_score > low.python_score
    assert high.ai_score > low.ai_score
    # saturates at 5 pieces of evidence
    saturated = score_candidate(["a"] * 5, ["b"] * 5, [], None)
    over_saturated = score_candidate(["a"] * 20, ["b"] * 20, [], None)
    assert saturated.python_score == over_saturated.python_score


def test_deep_project_scores_higher_than_shallow():
    deep_project = [ProjectEvidence(name="X", depth="deep", shallow_wrapper=False)]
    shallow_project = [ProjectEvidence(name="Y", depth="shallow", shallow_wrapper=False)]
    deep_score = score_candidate([], [], deep_project, None)
    shallow_score = score_candidate([], [], shallow_project, None)
    assert deep_score.project_quality_score > shallow_score.project_quality_score


def test_shallow_wrapper_flag_penalizes_project_score():
    labeled_deep_but_wrapper = [ProjectEvidence(name="X", depth="deep", shallow_wrapper=True)]
    labeled_deep_no_wrapper = [ProjectEvidence(name="X", depth="deep", shallow_wrapper=False)]
    penalized = score_candidate([], [], labeled_deep_but_wrapper, None)
    unpenalized = score_candidate([], [], labeled_deep_no_wrapper, None)
    assert penalized.project_quality_score < unpenalized.project_quality_score


def test_github_success_profile_increases_score():
    profile = GitHubProfile(status=Status.SUCCESS, username="octocat", public_repos=15, followers=50,
                             top_languages=["Python", "Go", "Rust"])
    with_github = score_candidate([], [], [], profile)
    without_github = score_candidate([], [], [], None)
    assert with_github.github_score > without_github.github_score


def test_github_failed_status_contributes_zero():
    failed_profile = GitHubProfile(status=Status.FAILED, username="ghost", error="not found")
    breakdown = score_candidate([], [], [], failed_profile)
    assert breakdown.github_score == 0.0


def test_total_score_is_sum_of_components():
    breakdown = score_candidate(["a"], ["b"], [ProjectEvidence(name="x", depth="moderate")], None)
    expected_total = round(
        breakdown.python_score + breakdown.ai_score + breakdown.project_quality_score + breakdown.github_score, 2
    )
    assert breakdown.total_score == expected_total
