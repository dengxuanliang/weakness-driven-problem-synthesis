from weakness_driven_problem_synthesis.schemas import EvalRecord, SynthProblem


def test_eval_record_normalizes_failed_pass_values():
    record = EvalRecord.model_validate(
        {
            "question_id": 1,
            "content": "x",
            "canonical_solution": "y",
            "completion": "z",
            "test": "assert True",
            "labels": {
                "category": "algorithms",
                "programming_language": "python",
                "difficulty": "hard",
            },
            "pass_at_1": None,
        }
    )
    assert record.is_failed is True


def test_synth_problem_requires_all_fields():
    SynthProblem.model_validate(
        {
            "id": "S00001",
            "weakness_id": "W001",
            "language": "python",
            "difficulty": "hard",
            "scenario": "stream compaction",
            "problem_statement": "x" * 240,
            "function_signature": "def f(x: list[int]) -> int:",
            "input_format": "list of ints",
            "output_format": "int",
            "constraints": ["1 <= n <= 1e5"],
            "edge_cases_hinted": ["empty input"],
            "anti_homogeneity_notes": "unique angle",
            "input_scale_class": "1e5-sequence",
            "data_shape_class": "flat-array",
            "primary_pitfall": "off-by-one",
            "novelty_reason": "Uses boundary-sensitive traversal under large input.",
        }
    )
