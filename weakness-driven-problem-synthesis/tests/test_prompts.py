from weakness_driven_problem_synthesis.prompts import load_prompt


def test_synthesize_prompt_includes_content_safety_constraints():
    prompt = load_prompt("synthesize.txt")

    assert "Content safety:" in prompt
    assert (
        "- Do not mention any weakness, pitfall, internal objective, or that the problem is designed to test a specific mistake or reasoning pattern."
        in prompt
    )
    assert (
        "- Keep all meta rationale only in `anti_homogeneity_notes`, `primary_pitfall`, and `novelty_reason`, never in solver-facing fields."
        in prompt
    )
