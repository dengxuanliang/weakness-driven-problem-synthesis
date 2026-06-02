import json

from weakness_driven_problem_synthesis.solver_view import write_solver_view


def test_write_solver_view_exports_only_solver_fields_and_prompt(tmp_path):
    synth_path = tmp_path / "synthesized_problems.jsonl"
    solver_path = tmp_path / "solver_view.jsonl"
    synth_path.write_text(
        json.dumps(
            {
                "id": "S00001",
                "weakness_id": "WG01",
                "batch_index": 0,
                "language": "python",
                "difficulty": "hard",
                "scenario": "demo scenario",
                "problem_statement": "Solve a batch processing problem.",
                "function_signature": "def solve(items: list[int]) -> int:",
                "input_format": "A list of integers.",
                "output_format": "An integer result.",
                "constraints": ["1 <= n <= 1e5", "values fit in 64-bit signed integers"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "internal note",
                "input_scale_class": "scale-a",
                "data_shape_class": "shape-a",
                "primary_pitfall": "pitfall-a",
                "novelty_reason": "novelty-a",
            }
        )
        + "\n"
    )

    write_solver_view(synthesized_path=synth_path, solver_view_path=solver_path)

    lines = solver_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {
        "id",
        "language",
        "difficulty",
        "problem_statement",
        "function_signature",
        "input_format",
        "output_format",
        "constraints",
        "solver_prompt",
    }
    assert record["solver_prompt"] == (
        "Solve a batch processing problem.\n\n"
        "Function signature:\n"
        "def solve(items: list[int]) -> int:\n\n"
        "Input:\n"
        "A list of integers.\n\n"
        "Output:\n"
        "An integer result.\n\n"
        "Constraints:\n"
        "- 1 <= n <= 1e5\n"
        "- values fit in 64-bit signed integers\n\n"
        "Return only code."
    )
    assert "Edge cases" not in record["solver_prompt"]
    assert "Return only code." in record["solver_prompt"]


def test_write_solver_view_uses_document_label_for_html_problems(tmp_path):
    synth_path = tmp_path / "synthesized_problems.jsonl"
    solver_path = tmp_path / "solver_view.jsonl"
    synth_path.write_text(
        json.dumps(
            {
                "id": "S00002",
                "weakness_id": "WG01",
                "batch_index": 0,
                "language": "html",
                "difficulty": "hard",
                "scenario": "demo html",
                "problem_statement": "Build a standalone HTML document.",
                "function_signature": "<!DOCTYPE html><html><body>...</body></html>",
                "input_format": "No external input.",
                "output_format": "A complete HTML document.",
                "constraints": ["Must be standalone"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "internal note",
                "input_scale_class": "scale-a",
                "data_shape_class": "shape-a",
                "primary_pitfall": "pitfall-a",
                "novelty_reason": "novelty-a",
            }
        )
        + "\n"
    )

    write_solver_view(synthesized_path=synth_path, solver_view_path=solver_path)

    record = json.loads(solver_path.read_text().strip())
    assert "Function signature:" not in record["solver_prompt"]
    assert "Required output form:" in record["solver_prompt"]
