"""Report generation for synthesis runs."""

from __future__ import annotations

from pathlib import Path

from weakness_driven_problem_synthesis.schemas import SynthesisSummary, WeaknessSet


def write_report(
    *,
    report_path: Path,
    failed_count: int,
    weakness_set: WeaknessSet,
    synthesis_summary: SynthesisSummary,
    sampled_problems: dict[str, str],
) -> None:
    lines = [
        "# Synthesis Report",
        "",
        "## Overall counts",
        f"- Failed questions: {failed_count}",
        f"- Weaknesses: {len(weakness_set.weaknesses)}",
        f"- Synthesized problems: {synthesis_summary.completed}",
        f"- Dropped: {synthesis_summary.dropped}",
        f"- Retries: {synthesis_summary.retry_count}",
        "",
        "## Weaknesses",
    ]

    for weakness in weakness_set.weaknesses:
        evidence_count = len(weakness_set.evidence_question_ids.get(weakness.id, []))
        lines.extend(
            [
                f"### {weakness.id} {weakness.name}",
                f"- Evidence count: {evidence_count}",
                f"- Sample: {sampled_problems.get(weakness.id, '')}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines))
