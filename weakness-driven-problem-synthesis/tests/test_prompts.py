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
    assert "Global diversity preference:" in prompt
    assert (
        "- Avoid reusing an already-common `(input_scale_class, data_shape_class)` combination when a different combination would fit."
        in prompt
    )
    assert (
        "- If a scale/shape combination must be reused, make the `scenario` and `primary_pitfall` clearly different from prior accepted problems."
        in prompt
    )
