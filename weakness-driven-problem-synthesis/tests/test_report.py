from weakness_driven_problem_synthesis.report import write_report
from weakness_driven_problem_synthesis.schemas import SynthesisSummary, Weakness, WeaknessSet


def test_write_report_includes_counts_and_sampled_problem(tmp_path):
    report_path = tmp_path / "report.md"
    weakness_set = WeaknessSet(
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
        evidence_question_ids={"W001": [1, 2]},
    )
    synthesis_summary = SynthesisSummary(
        completed=3,
        retry_count=1,
        dropped=0,
        skipped=2,
        extra_batches=1,
        completed_by_weakness={"W001": 3},
    )

    write_report(
        report_path=report_path,
        failed_count=2,
        weakness_set=weakness_set,
        synthesis_summary=synthesis_summary,
        sampled_problems={"W001": "Solve a deeply nested recursion problem with multiple constraints."},
    )

    text = report_path.read_text()
    assert "Overall counts" in text
    assert "W001" in text
    assert "Solve a deeply nested recursion problem" in text
    assert "Retries: 1" in text
    assert "Skipped: 2" in text
    assert "Extra batches: 1" in text
    assert "Evidence count" in text
    assert "Completed: 3" in text
