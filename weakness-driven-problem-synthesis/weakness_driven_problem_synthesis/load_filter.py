"""Load and filter evaluation records."""

from collections.abc import Iterator
from pathlib import Path

from weakness_driven_problem_synthesis.schemas import EvalRecord


def load_failed_records(path: Path) -> Iterator[EvalRecord]:
    with path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = EvalRecord.model_validate_json(raw_line)
            if record.is_failed:
                yield record
