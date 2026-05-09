"""CLI entrypoint for the synthesis pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

from weakness_driven_problem_synthesis.allocate import allocate_quotas
from weakness_driven_problem_synthesis.attribute import attribute_failures
from weakness_driven_problem_synthesis.cluster import cluster_weaknesses
from weakness_driven_problem_synthesis.load_filter import load_failed_records
from weakness_driven_problem_synthesis.report import write_report
from weakness_driven_problem_synthesis.schemas import SynthesisSummary, WeaknessSet
from weakness_driven_problem_synthesis.solver_view import write_solver_view
from weakness_driven_problem_synthesis.synthesize import synthesize_for_weaknesses

STAGE_ARTIFACTS = (
    "error_attributions.jsonl",
    "weaknesses.json",
    "synthesized_problems.jsonl",
    "solver_view.jsonl",
    "report.md",
)


def _validate_allocations(allocations: dict[str, int], *, total_questions: int) -> None:
    if any(value < 0 for value in allocations.values()):
        raise ValueError("allocations must not contain negative quotas")
    if sum(allocations.values()) != total_questions:
        raise ValueError("allocations must sum to total_questions")


def _empty_synthesis_summary() -> SynthesisSummary:
    return SynthesisSummary(completed=0, retry_count=0)


def _empty_weakness_set() -> WeaknessSet:
    return WeaknessSet(weaknesses=[], evidence_question_ids={})


def _write_empty_weaknesses_artifact(*, output_path: Path) -> None:
    output_path.write_text(_empty_weakness_set().model_dump_json(indent=2))


def _clear_stale_outputs_for_empty_run(*, output_dir: Path) -> None:
    for artifact_name in (
        "error_attributions.jsonl",
        "synthesized_problems.jsonl",
        "solver_view.jsonl",
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()


def _write_empty_report(*, report_path: Path, failed_count: int) -> None:
    write_report(
        report_path=report_path,
        failed_count=failed_count,
        weakness_set=_empty_weakness_set(),
        allocations={},
        synthesis_summary=_empty_synthesis_summary(),
        sampled_problems={},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--total-questions", type=int, required=True)
    parser.add_argument("--output-dir", default="./synthesis_output")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def prepare_output_dir(output_dir: Path, *, restart: bool, resume: bool = True) -> Path:
    if restart and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for artifact_name in STAGE_ARTIFACTS:
            artifact_path = output_dir / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()
    return output_dir


def estimate_call_counts(*, failed_count: int, total_questions: int, batch_size: int = 10) -> dict[str, int]:
    synthesis_batches = 0 if failed_count <= 0 else (total_questions + batch_size - 1) // batch_size
    return {
        "attribution_calls": failed_count,
        "synthesis_batches": synthesis_batches,
    }


def should_continue_after_estimate(*, non_interactive: bool = False) -> bool:
    if non_interactive:
        return True
    answer = input("Proceed with synthesis? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


async def main_with_args(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    output_dir = prepare_output_dir(Path(args.output_dir), restart=args.restart, resume=args.resume)
    failed_records = list(load_failed_records(Path(args.eval_log)))
    estimates = estimate_call_counts(failed_count=len(failed_records), total_questions=args.total_questions)
    print(
        "Pre-flight estimate: "
        f"{estimates['attribution_calls']} attribution calls, "
        f"{estimates['synthesis_batches']} synthesis batches"
    )
    if not should_continue_after_estimate(non_interactive=args.yes):
        return 1

    report_path = output_dir / "report.md"
    if not failed_records:
        _clear_stale_outputs_for_empty_run(output_dir=output_dir)
        _write_empty_weaknesses_artifact(output_path=output_dir / "weaknesses.json")
        _write_empty_report(report_path=report_path, failed_count=0)
        return 0

    error_attributions = await attribute_failures(
        failed_records,
        output_path=output_dir / "error_attributions.jsonl",
        provider=args.provider,
        model=args.model,
        concurrency=args.concurrency,
    )
    truly_failed_attributions = [item for item in error_attributions if item.is_truly_failed]
    if not truly_failed_attributions:
        _clear_stale_outputs_for_empty_run(output_dir=output_dir)
        _write_empty_weaknesses_artifact(output_path=output_dir / "weaknesses.json")
        _write_empty_report(report_path=report_path, failed_count=len(failed_records))
        return 0

    weakness_set = await cluster_weaknesses(
        truly_failed_attributions,
        eval_records=failed_records,
        output_path=output_dir / "weaknesses.json",
        provider=args.provider,
        model=args.model,
    )
    allocations = allocate_quotas(
        {weakness.id: len(weakness_set.evidence_question_ids.get(weakness.id, [])) for weakness in weakness_set.weaknesses},
        args.total_questions,
    )
    _validate_allocations(allocations, total_questions=args.total_questions)
    synthesis_summary = await synthesize_for_weaknesses(
        weakness_set,
        allocations=allocations,
        output_path=output_dir / "synthesized_problems.jsonl",
        provider=args.provider,
        model=args.model,
    )
    sampled = {}
    synth_path = output_dir / "synthesized_problems.jsonl"
    if synth_path.exists():
        write_solver_view(
            synthesized_path=synth_path,
            solver_view_path=output_dir / "solver_view.jsonl",
        )
        with synth_path.open() as handle:
            for raw_line in handle:
                if raw_line.strip():
                    item = json.loads(raw_line)
                    sampled.setdefault(item["weakness_id"], item["problem_statement"][:200])
    write_report(
        report_path=report_path,
        failed_count=len(failed_records),
        weakness_set=weakness_set,
        allocations=allocations,
        synthesis_summary=synthesis_summary,
        sampled_problems=sampled,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_with_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
