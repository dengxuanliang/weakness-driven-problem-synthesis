from pathlib import Path

import pytest

from weakness_driven_problem_synthesis.run import build_parser, main_with_args, prepare_output_dir
from weakness_driven_problem_synthesis.schemas import Attribution, SynthesisSummary, Weakness, WeaknessSet


def test_skill_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "SKILL.md").exists()
    assert (root / "scripts").is_dir()
    assert (root / "scripts" / "run.py").exists()
    assert (root / "references" / "prompts" / "attribute.txt").exists()


def test_cli_parses_expected_arguments():
    args = build_parser().parse_args(["--eval-log", "eval.jsonl", "--total-questions", "500"])
    assert args.eval_log == "eval.jsonl"
    assert args.total_questions == 500
    assert args.resume is True


def test_restart_deletes_stage_artifacts(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact = output_dir / "weaknesses.json"
    artifact.write_text("{}")

    prepare_output_dir(output_dir, restart=True)

    assert not artifact.exists()


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end_with_stubbed_llm(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    async def fake_attribute_failures(*args, **kwargs):
        return [
            Attribution(
                question_id=1,
                is_truly_failed=True,
                error_tags=["recursion:base-case-missing"],
                root_cause="missed base case",
                ability_dimensions=["reasoning"],
                evidence_snippet="if n == 0",
            )
        ]

    async def fake_cluster_weaknesses(*args, **kwargs):
        return WeaknessSet(
            weaknesses=[
                Weakness(
                    id="W001",
                    name="Recursion termination",
                    description="recursion bugs",
                    covered_tags=["recursion:base-case-missing"],
                    dominant_language="python",
                    dominant_category="algorithms",
                )
            ],
            evidence_question_ids={"W001": [1]},
        )

    async def fake_synthesize_for_weaknesses(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.write_text(
            '{"id":"S00001","weakness_id":"W001","batch_index":0,"language":"python","difficulty":"hard","scenario":"demo","problem_statement":"'
            + ("x" * 240)
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fake_attribute_failures)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fake_cluster_weaknesses)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fake_synthesize_for_weaknesses)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "openai",
            "--model",
            "test-model",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out" / "report.md").exists()
