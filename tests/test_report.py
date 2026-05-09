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
        shortfall_by_weakness={"W001": 2},
    )

    write_report(
        report_path=report_path,
        failed_count=2,
        weakness_set=weakness_set,
        allocations={"W001": 5},
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
    assert "Top Weaknesses" in text
    assert "Allocated quota" in text
    assert "Shortfall" in text
    assert "| W001 | Recursion termination | 2 | 5 | 3 | 2 |" in text
    assert "Shortfall: 2" in text


def test_write_report_sorts_weaknesses_by_importance(tmp_path):
    report_path = tmp_path / "report.md"
    weakness_set = WeaknessSet(
        weaknesses=[
            Weakness(
                id="W003",
                name="Third",
                description="d3",
                covered_tags=["t3"],
                dominant_language="python",
                dominant_category="algorithms",
            ),
            Weakness(
                id="W001",
                name="First",
                description="d1",
                covered_tags=["t1"],
                dominant_language="python",
                dominant_category="algorithms",
            ),
            Weakness(
                id="W002",
                name="Second",
                description="d2",
                covered_tags=["t2"],
                dominant_language="python",
                dominant_category="algorithms",
            ),
        ],
        evidence_question_ids={
            "W001": [1, 2, 3],
            "W002": [4, 5, 6],
            "W003": [7],
        },
    )
    synthesis_summary = SynthesisSummary(
        completed=8,
        retry_count=0,
        dropped=0,
        skipped=0,
        extra_batches=0,
        completed_by_weakness={"W001": 3, "W002": 2, "W003": 3},
        shortfall_by_weakness={"W001": 0, "W002": 1, "W003": 0},
    )

    write_report(
        report_path=report_path,
        failed_count=7,
        weakness_set=weakness_set,
        allocations={"W001": 4, "W002": 5, "W003": 2},
        synthesis_summary=synthesis_summary,
        sampled_problems={"W001": "p1", "W002": "p2", "W003": "p3"},
    )

    text = report_path.read_text()
    top_lines = [
        line
        for line in text.splitlines()
        if line.startswith("| W")
    ]
    assert top_lines[:3] == [
        "| W002 | Second | 3 | 5 | 2 | 1 |",
        "| W001 | First | 3 | 4 | 3 | 0 |",
        "| W003 | Third | 1 | 2 | 3 | 0 |",
    ]
