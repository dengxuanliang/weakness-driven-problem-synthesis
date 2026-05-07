from weakness_driven_problem_synthesis.load_filter import load_failed_records
from weakness_driven_problem_synthesis.schemas import EvalRecord


def test_load_failed_records_filters_non_failures(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":1}\n'
        '{"content":"b","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    records = list(load_failed_records(path))
    assert len(records) == 1
    assert records[0].content == "b"


def test_eval_record_maps_real_log_id_to_question_id():
    record = EvalRecord.model_validate(
        {
            "id": 229,
            "content": "problem",
            "canonical_solution": "x",
            "completion": "y",
            "test": {"code": "assert True"},
            "labels": {
                "category": "Desktop and Web Development",
                "programming_language": "python",
                "difficulty": "easy",
            },
            "pass_at_1": 0,
        }
    )

    assert record.question_id == 229


def test_load_failed_records_accepts_real_log_shape_with_id_field(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id":229,"content":"a","canonical_solution":"x","completion":"y","test":{"code":"assert True"},"labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    records = list(load_failed_records(path))

    assert len(records) == 1
    assert records[0].question_id == 229
