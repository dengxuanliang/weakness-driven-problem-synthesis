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
    assert (
        '- Do not include coaching or cautionary phrasing in solver-facing fields, such as "be careful", "watch out", "note that", "remember that", or similar guidance language.'
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


def test_attribute_prompt_includes_root_cause_and_tagging_guidance():
    prompt = load_prompt("attribute.txt")

    assert "Rules:" in prompt
    assert "- Diagnose the underlying cause, not just the surface mismatch." in prompt
    assert "- `root_cause` must explain why the solution failed." in prompt
    assert (
        "- `root_cause` should describe the immediate failure mechanism; `ability_dimensions` should describe the broader capability gap."
        in prompt
    )
    assert "- Use a small set of tags that best explain the failure; do not enumerate every symptom." in prompt
    assert "Tagging guidance:" in prompt
    assert "- Prefer stable, reusable tags over problem-specific ones." in prompt


def test_cluster_prompt_includes_grouping_and_granularity_guidance():
    prompt = load_prompt("cluster.txt")

    assert "Rules:" in prompt
    assert "- Group by shared underlying weakness, not surface topic or wording." in prompt
    assert "- `name` should describe the reusable weakness, not just restate one tag." in prompt
    assert "- Prefer a small number of coherent groups over many fragmented ones." in prompt
    assert (
        "- Use a singleton group only when the failure mode is clearly distinct and not well covered by any broader group."
        in prompt
    )
    assert "- Do not collapse clearly different weaknesses into one bucket." in prompt


def test_cluster_refine_prompt_constrains_local_candidate_review():
    prompt = load_prompt("cluster_refine.txt")

    assert "Return JSON only as an array of objects with:" in prompt
    assert "- Review only the provided candidate cluster; do not invent outside evidence." in prompt
    assert "- If the candidate cluster is coherent, return exactly one weakness." in prompt
    assert "- Prefer conservative splitting over incorrect merging." in prompt
    assert "- `covered_tags` must be chosen only from the provided candidate tags." in prompt


def test_cluster_merge_prompt_constrains_pairwise_merge_decisions():
    prompt = load_prompt("cluster_merge.txt")

    assert "Return JSON only as an object with:" in prompt
    assert "- `should_merge`" in prompt
    assert "- `merged_weakness`" in prompt
    assert "- Merge only when both clusters reflect the same underlying weakness." in prompt
    assert "- Be conservative: when uncertain, do not merge." in prompt
