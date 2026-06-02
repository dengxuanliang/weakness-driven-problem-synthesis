"""Load and filter evaluation records."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from weakness_driven_problem_synthesis.schemas import EvalRecord


@dataclass
class FailedRecordCandidate:
    record: EvalRecord
    raw_size_bytes: int


def load_failed_records(path: Path) -> Iterator[EvalRecord]:
    with path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = EvalRecord.model_validate_json(raw_line)
            if record.is_failed:
                yield record


def load_failed_record_candidates(path: Path) -> Iterator[FailedRecordCandidate]:
    with path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = EvalRecord.model_validate_json(raw_line)
            if record.is_failed:
                yield FailedRecordCandidate(
                    record=record,
                    raw_size_bytes=len(raw_line.encode("utf-8")),
                )
