from pathlib import Path
import subprocess
import sys

import pytest

from weakness_driven_problem_synthesis.run import (
    _validate_allocations,
    build_parser,
    estimate_call_counts,
    main_with_args,
    prepare_output_dir,
    should_continue_after_estimate,
)
from weakness_driven_problem_synthesis.schemas import Attribution, SynthesisSummary, Weakness, WeaknessSet


def test_skill_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "SKILL.md").exists()
    assert (root / "scripts").is_dir()
    assert (root / "scripts" / "bootstrap.sh").exists()
    assert (root / "scripts" / "run.py").exists()
    assert (root / "references" / "prompts" / "attribute.txt").exists()


def test_bootstrap_script_mentions_env_template_keys():
    root = Path(__file__).resolve().parents[1]
    script_text = (root / "scripts" / "bootstrap.sh").read_text()
    assert "OPENAI_API_KEY=" in script_text
    assert "OPENAI_BASE_URL=" in script_text
    assert "OPENAI_MODEL=" in script_text


def test_cli_parses_expected_arguments():
    args = build_parser().parse_args(["--eval-log", "eval.jsonl", "--total-questions", "500"])
    assert args.eval_log == "eval.jsonl"
    assert args.total_questions == 500
    assert args.provider == "openai"
    assert args.resume is True
    assert args.start_stage == "attribute"


def test_cli_parses_cluster_start_stage_arguments():
    args = build_parser().parse_args(
        [
            "--eval-log",
            "eval.jsonl",
            "--total-questions",
            "5",
            "--start-stage",
            "cluster",
            "--attributions-file",
            "error_attributions.jsonl",
        ]
    )
    assert args.start_stage == "cluster"
    assert args.attributions_file == "error_attributions.jsonl"


def test_cli_parses_synthesize_start_stage_arguments():
    args = build_parser().parse_args(
        [
            "--total-questions",
            "5",
            "--start-stage",
            "synthesize",
            "--attributions-file",
            "error_attributions.jsonl",
            "--weaknesses-file",
            "weaknesses.json",
        ]
    )
    assert args.start_stage == "synthesize"
    assert args.attributions_file == "error_attributions.jsonl"
    assert args.weaknesses_file == "weaknesses.json"


def test_restart_deletes_stage_artifacts(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    artifact = output_dir / "weaknesses.json"
    artifact.write_text("{}")

    prepare_output_dir(output_dir, restart=True)

    assert not artifact.exists()


def test_prepare_output_dir_clears_stage_artifacts_when_resume_disabled(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for name in [
        "error_attributions.jsonl",
        "weaknesses.json",
        "synthesized_problems.jsonl",
        "solver_view.jsonl",
        "report.md",
        "keep.txt",
    ]:
        (output_dir / name).write_text("x")

    prepare_output_dir(output_dir, restart=False, resume=False)

    assert not (output_dir / "error_attributions.jsonl").exists()
    assert not (output_dir / "weaknesses.json").exists()
    assert not (output_dir / "synthesized_problems.jsonl").exists()
    assert not (output_dir / "solver_view.jsonl").exists()
    assert not (output_dir / "report.md").exists()
    assert (output_dir / "keep.txt").exists()


def test_estimate_call_counts_uses_failed_count_and_batch_size():
    estimates = estimate_call_counts(failed_count=23, total_questions=27, batch_size=10)
    assert estimates["attribution_calls"] == 23
    assert estimates["synthesis_batches"] == 3


def test_estimate_call_counts_reports_zero_synthesis_batches_when_no_failures():
    estimates = estimate_call_counts(failed_count=0, total_questions=50, batch_size=10)
    assert estimates["attribution_calls"] == 0
    assert estimates["synthesis_batches"] == 0


def test_estimate_call_counts_reports_zero_attribution_calls_when_starting_from_cluster():
    estimates = estimate_call_counts(failed_count=23, total_questions=27, batch_size=10, start_stage="cluster")
    assert estimates["attribution_calls"] == 0
    assert estimates["synthesis_batches"] == 3


def test_estimate_call_counts_reports_zero_attribution_calls_when_starting_from_synthesize():
    estimates = estimate_call_counts(failed_count=23, total_questions=27, batch_size=10, start_stage="synthesize")
    assert estimates["attribution_calls"] == 0
    assert estimates["synthesis_batches"] == 3


def test_validate_allocations_rejects_negative_or_mismatched_totals():
    with pytest.raises(ValueError, match="must not contain negative quotas"):
        _validate_allocations({"W001": -1, "W002": 2}, total_questions=1)

    with pytest.raises(ValueError, match="must sum to total_questions"):
        _validate_allocations({"W001": 1, "W002": 1}, total_questions=3)


def test_should_continue_after_estimate_accepts_non_interactive_mode():
    assert should_continue_after_estimate(non_interactive=True) is True


def test_cli_parses_yes_flag():
    args = build_parser().parse_args(["--eval-log", "eval.jsonl", "--total-questions", "5", "--yes"])
    assert args.yes is True


def test_should_continue_after_estimate_reads_user_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert should_continue_after_estimate(non_interactive=False) is True


def test_should_continue_after_estimate_rejects_user_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert should_continue_after_estimate(non_interactive=False) is False


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
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
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
            "--yes",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out" / "report.md").exists()
    assert (tmp_path / "out" / "solver_view.jsonl").exists()
    solver_view = (tmp_path / "out" / "solver_view.jsonl").read_text()
    assert '"edge_cases_hinted"' not in solver_view
    assert '"solver_prompt"' in solver_view


@pytest.mark.asyncio
async def test_pipeline_stops_when_confirmation_rejected(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("synthesis stages should not run after confirmation rejection")

    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fail_if_called)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1


@pytest.mark.asyncio
async def test_synthesize_start_stage_passes_eval_records_when_eval_log_is_provided(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        '{"question_id":1,"content":"problem one","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    attributions_path = tmp_path / "error_attributions.jsonl"
    attributions_path.write_text(
        '{"question_id": 1, "is_truly_failed": true, "error_tags": ["recursion:base-case-missing"], "root_cause": "missed base case", "ability_dimensions": ["reasoning"], "evidence_snippet": "if n == 0"}\n'
    )
    weaknesses_path = tmp_path / "weaknesses.json"
    weaknesses_path.write_text(
        WeaknessSet(
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
        ).model_dump_json(indent=2)
    )

    captured = {}

    async def fake_synthesize_for_weaknesses(*args, **kwargs):
        captured["eval_records_by_id"] = kwargs.get("eval_records_by_id")
        output_path = kwargs["output_path"]
        output_path.write_text(
            '{"id":"S00001","weakness_id":"W001","batch_index":0,"language":"python","difficulty":"hard","scenario":"demo","problem_statement":"'
            + ("x" * 240)
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

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
            "--start-stage",
            "synthesize",
            "--attributions-file",
            str(attributions_path),
            "--weaknesses-file",
            str(weaknesses_path),
            "--yes",
        ]
    )

    assert exit_code == 0
    assert captured["eval_records_by_id"] is not None
    assert 1 in captured["eval_records_by_id"]


@pytest.mark.asyncio
async def test_pipeline_excludes_non_truly_failed_attributions_from_clustering(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
        '{"question_id":2,"content":"b","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
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
            ),
            Attribution(
                question_id=2,
                is_truly_failed=False,
                error_tags=["judge:false-positive"],
                root_cause="equivalent answer",
                ability_dimensions=["reasoning"],
                evidence_snippet="return ans",
            ),
        ]

    async def fake_cluster_weaknesses(attributions, **kwargs):
        assert [item.question_id for item in attributions] == [1]
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
            "--yes",
        ]
    )

    assert exit_code == 0


@pytest.mark.asyncio
async def test_pipeline_writes_empty_report_when_no_failed_records(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"x","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":1}\n'
    )
    output_dir.mkdir()
    (output_dir / "error_attributions.jsonl").write_text("stale attribution\n")
    (output_dir / "synthesized_problems.jsonl").write_text("stale synth\n")
    (output_dir / "solver_view.jsonl").write_text("stale solver\n")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("downstream llm stages should not run when there are no failed records")

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fail_if_called)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "5",
            "--output-dir",
            str(output_dir),
            "--yes",
        ]
    )

    assert exit_code == 0
    report_text = (output_dir / "report.md").read_text()
    assert (output_dir / "weaknesses.json").read_text() == '{\n  "weaknesses": [],\n  "evidence_question_ids": {}\n}'
    assert "Failed questions: 0" in report_text
    assert "Weaknesses: 0" in report_text
    assert "Synthesized problems: 0" in report_text
    assert not (output_dir / "error_attributions.jsonl").exists()
    assert not (output_dir / "synthesized_problems.jsonl").exists()
    assert not (output_dir / "solver_view.jsonl").exists()


@pytest.mark.asyncio
async def test_pipeline_writes_empty_report_when_no_truly_failed_attributions(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    output_dir.mkdir()
    (output_dir / "error_attributions.jsonl").write_text("stale attribution\n")
    (output_dir / "synthesized_problems.jsonl").write_text("stale synth\n")
    (output_dir / "solver_view.jsonl").write_text("stale solver\n")

    async def fake_attribute_failures(*args, **kwargs):
        return [
            Attribution(
                question_id=1,
                is_truly_failed=False,
                error_tags=["judge:false-positive"],
                root_cause="equivalent answer",
                ability_dimensions=["reasoning"],
                evidence_snippet="return ans",
            )
        ]

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("downstream synthesis stages should not run when no true failures remain")

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fake_attribute_failures)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fail_if_called)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "5",
            "--output-dir",
            str(output_dir),
            "--yes",
        ]
    )

    assert exit_code == 0
    report_text = (output_dir / "report.md").read_text()
    assert (output_dir / "weaknesses.json").read_text() == '{\n  "weaknesses": [],\n  "evidence_question_ids": {}\n}'
    assert "Failed questions: 1" in report_text
    assert "Weaknesses: 0" in report_text
    assert "Synthesized problems: 0" in report_text
    assert not (output_dir / "error_attributions.jsonl").exists()
    assert not (output_dir / "synthesized_problems.jsonl").exists()
    assert not (output_dir / "solver_view.jsonl").exists()


@pytest.mark.asyncio
async def test_pipeline_skips_oversized_failed_records_before_attribution(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    oversized_content = "x" * 1_100_000
    eval_log.write_text(
        '{"question_id":1,"content":"small","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
        + '{"question_id":2,"content":"'
        + oversized_content
        + '","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    async def fake_attribute_failures(records, *args, **kwargs):
        assert [record.question_id for record in records] == [1]
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
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
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
            str(output_dir),
            "--provider",
            "openai",
            "--model",
            "test-model",
            "--yes",
        ]
    )

    assert exit_code == 0
    skipped_lines = (output_dir / "skipped_failed_records.jsonl").read_text().strip().splitlines()
    assert len(skipped_lines) == 1
    assert '"question_id": 2' in skipped_lines[0]
    assert '"reason": "record_too_large"' in skipped_lines[0]
    report_text = (output_dir / "report.md").read_text()
    assert "Failed questions: 2" in report_text
    assert "Failed records skipped before attribution: 1" in report_text


@pytest.mark.asyncio
async def test_pipeline_writes_empty_report_when_all_failed_records_are_oversized(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "out"
    oversized_content = "x" * 1_100_000
    eval_log.write_text(
        '{"question_id":2,"content":"'
        + oversized_content
        + '","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("downstream llm stages should not run when all failed records are oversized")

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fail_if_called)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "5",
            "--output-dir",
            str(output_dir),
            "--yes",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "weaknesses.json").read_text() == '{\n  "weaknesses": [],\n  "evidence_question_ids": {}\n}'
    skipped_lines = (output_dir / "skipped_failed_records.jsonl").read_text().strip().splitlines()
    assert len(skipped_lines) == 1
    report_text = (output_dir / "report.md").read_text()
    assert "Failed questions: 1" in report_text
    assert "Failed records skipped before attribution: 1" in report_text
    assert "Weaknesses: 0" in report_text
    assert "Synthesized problems: 0" in report_text


@pytest.mark.asyncio
async def test_cluster_start_stage_requires_attributions_file(tmp_path):
    eval_log = tmp_path / "eval.jsonl"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )

    with pytest.raises(ValueError, match="--attributions-file is required when --start-stage=cluster"):
        await main_with_args(
            [
                "--eval-log",
                str(eval_log),
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "cluster",
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_cluster_start_stage_requires_eval_log(tmp_path):
    attributions_path = tmp_path / "error_attributions.jsonl"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
    )

    with pytest.raises(ValueError, match="--eval-log is required when --start-stage=cluster"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "cluster",
                "--attributions-file",
                str(attributions_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_requires_attributions_file(tmp_path):
    weaknesses_path = tmp_path / "weaknesses.json"
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[1]}}'
    )

    with pytest.raises(ValueError, match="--attributions-file is required when --start-stage=synthesize"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "synthesize",
                "--weaknesses-file",
                str(weaknesses_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_requires_weaknesses_file(tmp_path):
    attributions_path = tmp_path / "error_attributions.jsonl"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
    )

    with pytest.raises(ValueError, match="--weaknesses-file is required when --start-stage=synthesize"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "synthesize",
                "--attributions-file",
                str(attributions_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_skips_attribute_and_cluster_and_uses_existing_artifacts(tmp_path, monkeypatch):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    output_dir = tmp_path / "out"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
        '{"question_id":2,"is_truly_failed":false,"error_tags":["judge:false-positive"],"root_cause":"equivalent answer","ability_dimensions":["reasoning"],"evidence_snippet":"return ans"}\n'
    )
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[1]}}'
    )

    async def fail_attribute(*args, **kwargs):
        raise AssertionError("attribute_failures should not run when --start-stage=synthesize")

    async def fail_cluster(*args, **kwargs):
        raise AssertionError("cluster_weaknesses should not run when --start-stage=synthesize")

    async def fake_synthesize_for_weaknesses(weakness_set, *, allocations, output_path, provider, model, provider_client=None, **kwargs):
        assert [item.id for item in weakness_set.weaknesses] == ["W001"]
        assert allocations == {"W001": 1}
        output_path.write_text(
            '{"id":"S00001","weakness_id":"W001","batch_index":0,"language":"python","difficulty":"hard","scenario":"demo","problem_statement":"'
            + ("x" * 240)
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fail_attribute)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fail_cluster)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fake_synthesize_for_weaknesses)

    exit_code = await main_with_args(
        [
            "--total-questions",
            "1",
            "--output-dir",
            str(output_dir),
            "--start-stage",
            "synthesize",
            "--attributions-file",
            str(attributions_path),
            "--weaknesses-file",
            str(weaknesses_path),
            "--yes",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "report.md").exists()
    assert (output_dir / "solver_view.jsonl").exists()


@pytest.mark.asyncio
async def test_synthesize_start_stage_rejects_invalid_weaknesses_file(tmp_path):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
    )
    weaknesses_path.write_text('{"weaknesses":[],"evidence_question_ids":{}}')

    with pytest.raises(ValueError, match="weaknesses file must contain at least one weakness"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "synthesize",
                "--attributions-file",
                str(attributions_path),
                "--weaknesses-file",
                str(weaknesses_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_rejects_mismatched_evidence_question_ids(tmp_path):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
    )
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[999]}}'
    )

    with pytest.raises(ValueError, match="weakness evidence question ids do not match truly failed attributions"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "synthesize",
                "--attributions-file",
                str(attributions_path),
                "--weaknesses-file",
                str(weaknesses_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_rejects_evidence_ids_with_non_overlapping_tags(tmp_path):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["dp:state-transition"],"root_cause":"bad transition","ability_dimensions":["reasoning"],"evidence_snippet":"dp[i]"}\n'
    )
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[1]}}'
    )

    with pytest.raises(ValueError, match="weakness evidence tags do not align with attribution tags"):
        await main_with_args(
            [
                "--total-questions",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
                "--start-stage",
                "synthesize",
                "--attributions-file",
                str(attributions_path),
                "--weaknesses-file",
                str(weaknesses_path),
                "--yes",
            ]
        )


@pytest.mark.asyncio
async def test_synthesize_start_stage_accepts_matching_evidence_ids_and_tags(tmp_path, monkeypatch):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    output_dir = tmp_path / "out"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing","recursion:stack-overflow"],"root_cause":"bad recursion","ability_dimensions":["reasoning"],"evidence_snippet":"dfs"}\n'
    )
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[1]}}'
    )

    async def fake_synthesize_for_weaknesses(weakness_set, *, allocations, output_path, provider, model, provider_client=None, **kwargs):
        assert allocations == {"W001": 1}
        output_path.write_text(
            '{"id":"S00001","weakness_id":"W001","batch_index":0,"language":"python","difficulty":"hard","scenario":"demo","problem_statement":"'
            + ("x" * 240)
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fake_synthesize_for_weaknesses)

    exit_code = await main_with_args(
        [
            "--total-questions",
            "1",
            "--output-dir",
            str(output_dir),
            "--start-stage",
            "synthesize",
            "--attributions-file",
            str(attributions_path),
            "--weaknesses-file",
            str(weaknesses_path),
            "--yes",
        ]
    )

    assert exit_code == 0


@pytest.mark.asyncio
async def test_synthesize_start_stage_loads_attributions_only_once(tmp_path, monkeypatch):
    attributions_path = tmp_path / "error_attributions.jsonl"
    weaknesses_path = tmp_path / "weaknesses.json"
    output_dir = tmp_path / "out"
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"bad recursion","ability_dimensions":["reasoning"],"evidence_snippet":"dfs"}\n'
    )
    weaknesses_path.write_text(
        '{"weaknesses":[{"id":"W001","name":"Recursion termination","description":"recursion bugs","covered_tags":["recursion:base-case-missing"],"dominant_language":"python","dominant_category":"algorithms"}],"evidence_question_ids":{"W001":[1]}}'
    )

    load_calls = {"count": 0}

    def fake_load_attributions_jsonl(*, path):
        load_calls["count"] += 1
        return [
            Attribution(
                question_id=1,
                is_truly_failed=True,
                error_tags=["recursion:base-case-missing"],
                root_cause="bad recursion",
                ability_dimensions=["reasoning"],
                evidence_snippet="dfs",
            )
        ]

    def fake_load_weakness_set_json(*, path):
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

    def fake_validate_weakness_evidence_against_attributions(*, weakness_set, truly_failed_attributions):
        assert [item.question_id for item in truly_failed_attributions] == [1]

    async def fake_synthesize_for_weaknesses(weakness_set, *, allocations, output_path, provider, model, provider_client=None, **kwargs):
        output_path.write_text(
            '{"id":"S00001","weakness_id":"W001","batch_index":0,"language":"python","difficulty":"hard","scenario":"demo","problem_statement":"'
            + ("x" * 240)
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

    monkeypatch.setattr("weakness_driven_problem_synthesis.run._load_attributions_jsonl", fake_load_attributions_jsonl)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run._load_weakness_set_json", fake_load_weakness_set_json)
    monkeypatch.setattr(
        "weakness_driven_problem_synthesis.run._validate_weakness_evidence_against_attributions",
        fake_validate_weakness_evidence_against_attributions,
    )
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fake_synthesize_for_weaknesses)

    exit_code = await main_with_args(
        [
            "--total-questions",
            "1",
            "--output-dir",
            str(output_dir),
            "--start-stage",
            "synthesize",
            "--attributions-file",
            str(attributions_path),
            "--weaknesses-file",
            str(weaknesses_path),
            "--yes",
        ]
    )

    assert exit_code == 0
    assert load_calls["count"] == 1


@pytest.mark.asyncio
async def test_cluster_start_stage_skips_attribution_and_uses_existing_attributions(tmp_path, monkeypatch):
    eval_log = tmp_path / "eval.jsonl"
    attributions_path = tmp_path / "error_attributions.jsonl"
    output_dir = tmp_path / "out"
    eval_log.write_text(
        '{"question_id":1,"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"algorithms","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
        '{"question_id":2,"content":"b","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"algorithms","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    attributions_path.write_text(
        '{"question_id":1,"is_truly_failed":true,"error_tags":["recursion:base-case-missing"],"root_cause":"missed base case","ability_dimensions":["reasoning"],"evidence_snippet":"if n == 0"}\n'
        '{"question_id":2,"is_truly_failed":false,"error_tags":["judge:false-positive"],"root_cause":"equivalent answer","ability_dimensions":["reasoning"],"evidence_snippet":"return ans"}\n'
    )

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("attribute_failures should not run when --start-stage=cluster")

    async def fake_cluster_weaknesses(attributions, **kwargs):
        assert [item.question_id for item in attributions] == [1]
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
            + '","function_signature":"def solve(x: list[int]) -> int:","input_format":"list[int]","output_format":"int","constraints":["1 <= n <= 1e5"],"edge_cases_hinted":["empty input"],"anti_homogeneity_notes":"demo","input_scale_class":"scale-a","data_shape_class":"shape-a","primary_pitfall":"pitfall-a","novelty_reason":"novelty-a"}\n'
        )
        return SynthesisSummary(completed=1, retry_count=0, dropped=0)

    monkeypatch.setattr("weakness_driven_problem_synthesis.run.attribute_failures", fail_if_called)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.cluster_weaknesses", fake_cluster_weaknesses)
    monkeypatch.setattr("weakness_driven_problem_synthesis.run.synthesize_for_weaknesses", fake_synthesize_for_weaknesses)

    exit_code = await main_with_args(
        [
            "--eval-log",
            str(eval_log),
            "--total-questions",
            "1",
            "--output-dir",
            str(output_dir),
            "--start-stage",
            "cluster",
            "--attributions-file",
            str(attributions_path),
            "--yes",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "report.md").exists()
    assert (output_dir / "solver_view.jsonl").exists()


def test_module_cli_entrypoint_executes_and_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "weakness_driven_problem_synthesis.run", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert "--eval-log" in result.stdout
