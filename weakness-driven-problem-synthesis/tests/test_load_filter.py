from weakness_driven_problem_synthesis.load_filter import load_failed_records


def test_load_failed_records_filters_non_failures(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":1}\n'
        '{"content":"b","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    records = list(load_failed_records(path))
    assert len(records) == 1
    assert records[0].content == "b"
