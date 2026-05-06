"""CLI entrypoint for the synthesis pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from weakness_driven_problem_synthesis.allocate import allocate_quotas
from weakness_driven_problem_synthesis.attribute import attribute_failures
from weakness_driven_problem_synthesis.cluster import cluster_weaknesses
from weakness_driven_problem_synthesis.load_filter import load_failed_records
from weakness_driven_problem_synthesis.report import write_report
from weakness_driven_problem_synthesis.schemas import SynthesisSummary
from weakness_driven_problem_synthesis.synthesize import synthesize_for_weaknesses

STAGE_ARTIFACTS = (
    "error_attributions.jsonl",
    "weaknesses.json",
    "synthesized_problems.jsonl",
    "report.md",
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
    return {
        "attribution_calls": failed_count,
        "synthesis_batches": (total_questions + batch_size - 1) // batch_size,
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

    error_attributions = await attribute_failures(
        failed_records,
        output_path=output_dir / "error_attributions.jsonl",
        provider=args.provider,
        model=args.model,
        concurrency=args.concurrency,
    )
    truly_failed_attributions = [item for item in error_attributions if item.is_truly_failed]
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
    synthesis_summary = await synthesize_for_weaknesses(
        weakness_set,
        allocations=allocations,
        output_path=output_dir / "synthesized_problems.jsonl",
        provider=args.provider,
        model=args.model,
    )
    report_path = output_dir / "report.md"
    sampled = {}
    synth_path = output_dir / "synthesized_problems.jsonl"
    if synth_path.exists():
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
