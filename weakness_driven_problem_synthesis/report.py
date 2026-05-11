"""Report generation for synthesis runs."""

from __future__ import annotations

from pathlib import Path

from weakness_driven_problem_synthesis.schemas import SynthesisSummary, WeaknessSet


def _sorted_weaknesses(
    weakness_set: WeaknessSet,
    allocations: dict[str, int],
    synthesis_summary: SynthesisSummary,
):
    return sorted(
        weakness_set.weaknesses,
        key=lambda weakness: (
            -len(weakness_set.evidence_question_ids.get(weakness.id, [])),
            -allocations.get(weakness.id, 0),
            -synthesis_summary.completed_by_weakness.get(weakness.id, 0),
            weakness.id,
        ),
    )


def write_report(
    *,
    report_path: Path,
    failed_count: int,
    failed_records_skipped_before_attribution: int = 0,
    weakness_set: WeaknessSet,
    allocations: dict[str, int],
    synthesis_summary: SynthesisSummary,
    sampled_problems: dict[str, str],
) -> None:
    ordered_weaknesses = _sorted_weaknesses(weakness_set, allocations, synthesis_summary)
    lines = [
        "# Synthesis Report",
        "",
        "## Overall counts",
        f"- Failed questions: {failed_count}",
        f"- Failed records skipped before attribution: {failed_records_skipped_before_attribution}",
        f"- Weaknesses: {len(weakness_set.weaknesses)}",
        f"- Synthesized problems: {synthesis_summary.completed}",
        f"- Dropped: {synthesis_summary.dropped}",
        f"- Retries: {synthesis_summary.retry_count}",
        f"- Skipped: {synthesis_summary.skipped}",
        f"- Extra batches: {synthesis_summary.extra_batches}",
        "",
        "## Top Weaknesses",
        "| ID | Name | Evidence count | Allocated quota | Completed | Shortfall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for weakness in ordered_weaknesses:
        evidence_count = len(weakness_set.evidence_question_ids.get(weakness.id, []))
        lines.append(
            f"| {weakness.id} | {weakness.name} | {evidence_count} | {allocations.get(weakness.id, 0)} | {synthesis_summary.completed_by_weakness.get(weakness.id, 0)} | {synthesis_summary.shortfall_by_weakness.get(weakness.id, 0)} |"
        )

    lines.extend(
        [
            "",
        "## Weaknesses",
        ]
    )

    for weakness in ordered_weaknesses:
        evidence_count = len(weakness_set.evidence_question_ids.get(weakness.id, []))
        lines.extend(
            [
                f"### {weakness.id} {weakness.name}",
                f"- Evidence count: {evidence_count}",
                f"- Allocated quota: {allocations.get(weakness.id, 0)}",
                f"- Completed: {synthesis_summary.completed_by_weakness.get(weakness.id, 0)}",
                f"- Shortfall: {synthesis_summary.shortfall_by_weakness.get(weakness.id, 0)}",
                f"- Sample: {sampled_problems.get(weakness.id, '')}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines))
