from weakness_driven_problem_synthesis.dedup import duplicate_key, ngram_jaccard


def test_ngram_jaccard_detects_high_similarity():
    a = "alpha beta gamma delta epsilon zeta eta theta"
    b = "alpha beta gamma delta epsilon zeta eta lambda"
    assert ngram_jaccard(a, b, n=4) >= 0.6


def test_duplicate_key_uses_scenario_and_signature():
    problem = {
        "scenario": "payments reconciliation",
        "function_signature": "def f(x: list[int]) -> int:",
    }
    assert duplicate_key(problem) == (
        "payments reconciliation",
        "def f(x: list[int]) -> int:",
    )
