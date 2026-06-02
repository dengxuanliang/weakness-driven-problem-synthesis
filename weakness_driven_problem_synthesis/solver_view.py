"""Export solver-facing views of synthesized problems."""

from __future__ import annotations

import json
from pathlib import Path


def _build_solver_prompt(problem: dict) -> str:
    constraint_lines = "\n".join(f"- {item}" for item in problem["constraints"])
    if problem["language"] == "html":
        interface_label = "Required output form"
    else:
        interface_label = "Function signature"
    return (
        f"{problem['problem_statement']}\n\n"
        f"{interface_label}:\n{problem['function_signature']}\n\n"
        f"Input:\n{problem['input_format']}\n\n"
        f"Output:\n{problem['output_format']}\n\n"
        f"Constraints:\n{constraint_lines}\n\n"
        "Return only code."
    )


def write_solver_view(*, synthesized_path: Path, solver_view_path: Path) -> None:
    lines: list[str] = []
    with synthesized_path.open() as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            problem = json.loads(raw_line)
            solver_record = {
                "id": problem["id"],
                "language": problem["language"],
                "difficulty": problem["difficulty"],
                "problem_statement": problem["problem_statement"],
                "function_signature": problem["function_signature"],
                "input_format": problem["input_format"],
                "output_format": problem["output_format"],
                "constraints": problem["constraints"],
                "solver_prompt": _build_solver_prompt(problem),
            }
            lines.append(json.dumps(solver_record, ensure_ascii=False))
    solver_view_path.write_text("\n".join(lines) + ("\n" if lines else ""))
