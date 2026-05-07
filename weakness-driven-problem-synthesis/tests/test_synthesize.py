import json

import pytest

from weakness_driven_problem_synthesis.schemas import SynthesisSummary, Weakness, WeaknessSet
from weakness_driven_problem_synthesis.synthesize import (
    RECENT_SUMMARY_LIMIT,
    has_high_similarity,
    synthesize_for_weaknesses,
)


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete_json(self, *, prompt, schema, system, max_tokens, model):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        return self.outputs.pop(0)


def make_weakness_set() -> WeaknessSet:
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
        evidence_question_ids={"W001": [1, 2, 3]},
    )


@pytest.mark.asyncio
async def test_synthesize_problems_respects_existing_batches_on_resume(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing = {
        "id": "S00001",
        "weakness_id": "W001",
        "batch_index": 0,
        "language": "python",
        "difficulty": "hard",
        "scenario": "existing scenario",
        "problem_statement": "x" * 240,
        "function_signature": "def solve(items: list[int]) -> int:",
        "input_format": "list[int]",
        "output_format": "int",
        "constraints": ["1 <= n <= 1e5"],
        "edge_cases_hinted": ["empty input"],
        "anti_homogeneity_notes": "baseline",
    }
    output_path.write_text(json.dumps(existing) + "\n")
    client = FakeProvider(outputs=[])

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 1},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result == SynthesisSummary(
        completed=1,
        retry_count=0,
        dropped=0,
        skipped=1,
        extra_batches=0,
        completed_by_weakness={"W001": 1},
        shortfall_by_weakness={"W001": 0},
    )
    assert len(client.calls) == 0


@pytest.mark.asyncio
async def test_synthesize_problems_skips_completed_batch_by_batch_identity(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing_records = []
    for idx in range(10):
        existing_records.append(
            {
                "id": f"S{idx:05d}",
                "weakness_id": "W001",
                "batch_index": 0,
                "language": "python",
                "difficulty": "hard",
                "scenario": f"scenario-{idx}",
                "problem_statement": "x" * 240,
                "function_signature": f"def solve_{idx}(items: list[int]) -> int:",
                "input_format": "list[int]",
                "output_format": "int",
                "constraints": ["1 <= n <= 1e5"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "baseline",
                "input_scale_class": f"scale-{idx}",
                "data_shape_class": f"shape-{idx}",
                "primary_pitfall": f"pitfall-{idx}",
                "novelty_reason": "existing batch item",
            }
        )
    output_path.write_text("\n".join(json.dumps(item) for item in existing_records) + "\n")
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S99999",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "next-batch-scenario",
                    "problem_statement": "y" * 240,
                    "function_signature": "def solve_next(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "new batch",
                    "input_scale_class": "new-scale",
                    "data_shape_class": "new-shape",
                    "primary_pitfall": "new-pitfall",
                    "novelty_reason": "continues after completed batch 0",
                }
            ]
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 11},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.skipped == 10
    lines = [json.loads(line) for line in output_path.read_text().strip().splitlines()]
    assert lines[-1]["batch_index"] == 1


@pytest.mark.asyncio
async def test_synthesize_problems_does_not_treat_partial_batch_as_complete(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing_records = []
    for idx in range(9):
        existing_records.append(
            {
                "id": f"S{idx:05d}",
                "weakness_id": "W001",
                "batch_index": 0,
                "language": "python",
                "difficulty": "hard",
                "scenario": f"scenario-{idx}",
                "problem_statement": "x" * 240,
                "function_signature": f"def solve_{idx}(items: list[int]) -> int:",
                "input_format": "list[int]",
                "output_format": "int",
                "constraints": ["1 <= n <= 1e5"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "baseline",
                "input_scale_class": f"scale-{idx}",
                "data_shape_class": f"shape-{idx}",
                "primary_pitfall": f"pitfall-{idx}",
                "novelty_reason": "partial batch item",
            }
        )
    output_path.write_text("\n".join(json.dumps(item) for item in existing_records) + "\n")
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S99998",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "fills-partial-batch",
                    "problem_statement": "z" * 240,
                    "function_signature": "def solve_fill(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "fills batch 0",
                    "input_scale_class": "fill-scale",
                    "data_shape_class": "fill-shape",
                    "primary_pitfall": "fill-pitfall",
                    "novelty_reason": "completes the partial batch",
                }
            ]
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 10},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.skipped == 9
    lines = [json.loads(line) for line in output_path.read_text().strip().splitlines()]
    assert lines[-1]["batch_index"] == 0


@pytest.mark.asyncio
async def test_synthesize_problems_caps_skipped_to_current_target(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing_records = []
    for idx in range(10):
        existing_records.append(
            {
                "id": f"S{idx:05d}",
                "weakness_id": "W001",
                "batch_index": 0,
                "language": "python",
                "difficulty": "hard",
                "scenario": f"scenario-{idx}",
                "problem_statement": "x" * 240,
                "function_signature": f"def solve_{idx}(items: list[int]) -> int:",
                "input_format": "list[int]",
                "output_format": "int",
                "constraints": ["1 <= n <= 1e5"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "baseline",
                "input_scale_class": f"scale-{idx}",
                "data_shape_class": f"shape-{idx}",
                "primary_pitfall": f"pitfall-{idx}",
                "novelty_reason": "existing batch item",
            }
        )
    output_path.write_text("\n".join(json.dumps(item) for item in existing_records) + "\n")
    client = FakeProvider(outputs=[])

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 5},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.skipped == 5
    assert len(client.calls) == 0


@pytest.mark.asyncio
async def test_synthesize_prompt_limits_recent_history_and_uses_latest_entries(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing_records = []
    for idx in range(RECENT_SUMMARY_LIMIT + 5):
        existing_records.append(
            {
                "id": f"S{idx:05d}",
                "weakness_id": "W001",
                "batch_index": idx // 10,
                "language": "python",
                "difficulty": "hard",
                "scenario": f"scenario-{idx}",
                "problem_statement": "x" * 240,
                "function_signature": f"def solve_{idx}(items: list[int]) -> int:",
                "input_format": "list[int]",
                "output_format": "int",
                "constraints": ["1 <= n <= 1e5"],
                "edge_cases_hinted": ["empty input"],
                "anti_homogeneity_notes": "baseline",
                "input_scale_class": f"scale-{idx}",
                "data_shape_class": f"shape-{idx}",
                "primary_pitfall": f"pitfall-{idx}",
                "novelty_reason": f"novelty-{idx}",
            }
        )
    output_path.write_text("\n".join(json.dumps(item) for item in existing_records) + "\n")
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S99997",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "fresh-scenario",
                    "problem_statement": "y" * 240,
                    "function_signature": "def solve_fresh(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bounded prompt",
                    "input_scale_class": "fresh-scale",
                    "data_shape_class": "fresh-shape",
                    "primary_pitfall": "fresh-pitfall",
                    "novelty_reason": "fresh-novelty",
                }
            ]
        ]
    )

    await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": len(existing_records) + 1},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    prompt = client.calls[0]["prompt"]
    assert "scenario-0" not in prompt
    assert f"scenario-{len(existing_records) - 1}" in prompt
    assert prompt.count("novelty=") == RECENT_SUMMARY_LIMIT


@pytest.mark.asyncio
async def test_synthesize_prompt_includes_coverage_summary(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing_records = [
        {
            "id": "S00001",
            "weakness_id": "W001",
            "batch_index": 0,
            "language": "python",
            "difficulty": "hard",
            "scenario": "scenario-a",
            "problem_statement": "x" * 240,
            "function_signature": "def solve_a(items: list[int]) -> int:",
            "input_format": "list[int]",
            "output_format": "int",
            "constraints": ["1 <= n <= 1e5"],
            "edge_cases_hinted": ["empty input"],
            "anti_homogeneity_notes": "baseline",
            "input_scale_class": "1e5-stream",
            "data_shape_class": "flat-log",
            "primary_pitfall": "boundary-reset",
            "novelty_reason": "a",
        },
        {
            "id": "S00002",
            "weakness_id": "W001",
            "batch_index": 0,
            "language": "python",
            "difficulty": "hard",
            "scenario": "scenario-b",
            "problem_statement": "y" * 240,
            "function_signature": "def solve_b(items: list[int]) -> int:",
            "input_format": "list[int]",
            "output_format": "int",
            "constraints": ["1 <= n <= 1e5"],
            "edge_cases_hinted": ["empty input"],
            "anti_homogeneity_notes": "baseline",
            "input_scale_class": "1e5-stream",
            "data_shape_class": "flat-log",
            "primary_pitfall": "boundary-reset",
            "novelty_reason": "b",
        },
    ]
    output_path.write_text("\n".join(json.dumps(item) for item in existing_records) + "\n")
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S99996",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "fresh-scenario",
                    "problem_statement": "z" * 240,
                    "function_signature": "def solve_fresh(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "coverage prompt",
                    "input_scale_class": "fresh-scale",
                    "data_shape_class": "fresh-shape",
                    "primary_pitfall": "fresh-pitfall",
                    "novelty_reason": "fresh-novelty",
                }
            ]
        ]
    )

    await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 3},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    prompt = client.calls[0]["prompt"]
    assert "Coverage memory:" in prompt
    assert "input_scale_class counts:" in prompt
    assert "1e5-stream: 2" in prompt
    assert "data_shape_class counts:" in prompt
    assert "flat-log: 2" in prompt
    assert "primary_pitfall counts:" in prompt
    assert "boundary-reset: 2" in prompt


@pytest.mark.asyncio
async def test_synthesize_refill_prompt_uses_latest_accepted_problem_context(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S10001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "accepted-first",
                    "problem_statement": "a" * 240,
                    "function_signature": "def solve_first(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "first accepted",
                    "input_scale_class": "accepted-scale",
                    "data_shape_class": "accepted-shape",
                    "primary_pitfall": "accepted-pitfall",
                    "novelty_reason": "accepted-novelty",
                },
                {
                    "id": "S10002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "accepted-first",
                    "problem_statement": "b" * 240,
                    "function_signature": "def solve_first(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["duplicate timestamps"],
                    "anti_homogeneity_notes": "forces refill",
                    "input_scale_class": "accepted-scale",
                    "data_shape_class": "accepted-shape",
                    "primary_pitfall": "accepted-pitfall",
                    "novelty_reason": "forces-refill",
                },
            ],
            [
                {
                    "id": "S10003",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "accepted-third",
                    "problem_statement": "c" * 240,
                    "function_signature": "def solve_third(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["duplicate timestamps"],
                    "anti_homogeneity_notes": "refill success",
                    "input_scale_class": "refill-scale",
                    "data_shape_class": "refill-shape",
                    "primary_pitfall": "refill-pitfall",
                    "novelty_reason": "refill-novelty",
                }
            ],
        ]
    )

    await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 2},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    refill_prompt = client.calls[1]["prompt"]
    assert "accepted-first" in refill_prompt
    assert "accepted-novelty" in refill_prompt
    assert "accepted-scale: 1" in refill_prompt


@pytest.mark.asyncio
async def test_synthesize_problems_rejects_empty_refill_payload_with_clear_error(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S10001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "accepted-first",
                    "problem_statement": "a" * 240,
                    "function_signature": "def solve_first(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "first accepted",
                    "input_scale_class": "accepted-scale",
                    "data_shape_class": "accepted-shape",
                    "primary_pitfall": "accepted-pitfall",
                    "novelty_reason": "accepted-novelty",
                },
                {
                    "id": "S10002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "accepted-first",
                    "problem_statement": "b" * 240,
                    "function_signature": "def solve_first(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["duplicate timestamps"],
                    "anti_homogeneity_notes": "forces refill",
                    "input_scale_class": "accepted-scale",
                    "data_shape_class": "accepted-shape",
                    "primary_pitfall": "accepted-pitfall",
                    "novelty_reason": "forces-refill",
                },
            ],
            [],
        ]
    )

    with pytest.raises(ValueError, match=r"synthesize_for_weaknesses expected non-empty JSON array payload"):
        await synthesize_for_weaknesses(
            make_weakness_set(),
            allocations={"W001": 2},
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=client,
        )


@pytest.mark.asyncio
async def test_synthesize_problems_regenerates_duplicates_and_short_statements(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S00001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "duplicate scenario",
                    "problem_statement": "short text",
                    "function_signature": "def solve(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "first try",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Boundary-heavy sequence scan.",
                }
            ],
            [
                {
                    "id": "S00002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "duplicate scenario",
                    "problem_statement": "y" * 240,
                    "function_signature": "def solve(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "second try",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Still too close in setup.",
                }
            ],
            [
                {
                    "id": "S00003",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "fresh scenario",
                    "problem_statement": "z" * 240,
                    "function_signature": "def solve(data: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "third try",
                    "input_scale_class": "sparse-graph",
                    "data_shape_class": "graph",
                    "primary_pitfall": "termination-condition",
                    "novelty_reason": "Switches to graph recursion instead of linear scan.",
                }
            ],
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 1},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.completed == 1
    assert result.retry_count == 2
    assert result.completed_by_weakness == {"W001": 1}
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["id"] == "S00003"


@pytest.mark.asyncio
async def test_synthesize_problems_writes_batch_index_for_batch_generation(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S00001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "scenario one",
                    "problem_statement": "a" * 240,
                    "function_signature": "def solve_a(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "one",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Large sequence with boundary-sensitive updates.",
                },
                {
                    "id": "S00002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "scenario two",
                    "problem_statement": "b" * 240,
                    "function_signature": "def solve_b(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["duplicate timestamps"],
                    "anti_homogeneity_notes": "two",
                    "input_scale_class": "event-stream",
                    "data_shape_class": "nested-records",
                    "primary_pitfall": "ordering-stability",
                    "novelty_reason": "Streaming record ordering under duplicate timestamps.",
                },
            ]
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 2},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.completed == 2
    lines = [json.loads(line) for line in output_path.read_text().strip().splitlines()]
    assert [line["batch_index"] for line in lines] == [0, 0]


@pytest.mark.asyncio
async def test_synthesize_problems_drops_after_retry_budget_and_counts_extra_batch(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S00001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad one",
                    "problem_statement": "short",
                    "function_signature": "def solve_bad(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Too short and under-specified.",
                }
            ],
            [
                {
                    "id": "S00002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad two",
                    "problem_statement": "still short",
                    "function_signature": "def solve_bad2(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Still under-specified.",
                }
            ],
            [
                {
                    "id": "S00003",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad three",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad3(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Again too short.",
                }
            ],
            [
                {
                    "id": "S00004",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "refill success",
                    "problem_statement": "c" * 240,
                    "function_signature": "def solve_good(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "good",
                    "input_scale_class": "tree-dp",
                    "data_shape_class": "tree",
                    "primary_pitfall": "state-carryover",
                    "novelty_reason": "Moves to tree state propagation instead of arrays.",
                }
            ],
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 1},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.retry_count == 3
    assert result.dropped == 1
    assert result.extra_batches == 1
    assert result.completed == 1
    assert result.shortfall_by_weakness == {"W001": 0}


@pytest.mark.asyncio
async def test_synthesize_problems_records_shortfall_when_refill_is_exhausted(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S10001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad one",
                    "problem_statement": "short",
                    "function_signature": "def solve_bad_a(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad two",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_b(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10003",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad three",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_c(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10004",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad four",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_d(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10005",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad five",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_e(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10006",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad six",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_f(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10007",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad seven",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_g(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10008",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad eight",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_h(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
            [
                {
                    "id": "S10009",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "bad nine",
                    "problem_statement": "tiny",
                    "function_signature": "def solve_bad_i(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "bad",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "bad",
                }
            ],
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 1},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.completed == 0
    assert result.shortfall_by_weakness == {"W001": 1}


@pytest.mark.asyncio
async def test_synthesize_problems_includes_prior_summary_in_following_batch_prompt(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S00001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "scenario one",
                    "problem_statement": "a" * 240,
                    "function_signature": "def solve_a(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "one",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Sequence-focused recursion setup.",
                }
            ],
            [
                {
                    "id": "S00002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "scenario two",
                    "problem_statement": "b" * 240,
                    "function_signature": "def solve_b(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["duplicate timestamps"],
                    "anti_homogeneity_notes": "two",
                    "input_scale_class": "event-stream",
                    "data_shape_class": "nested-records",
                    "primary_pitfall": "ordering-stability",
                    "novelty_reason": "Switches to stream ordering and nested records.",
                }
            ],
        ]
    )

    await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 2},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert "Recent generated problems" in client.calls[1]["prompt"]
    assert "scenario one" in client.calls[1]["prompt"]
    assert "Coverage memory:" in client.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_synthesize_problems_retries_high_jaccard_similarity(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    existing = {
        "id": "S00000",
        "weakness_id": "W001",
        "batch_index": 0,
        "language": "python",
        "difficulty": "hard",
        "scenario": "existing scenario",
        "problem_statement": "alpha beta gamma delta epsilon zeta eta theta",
        "function_signature": "def solve_existing(items: list[int]) -> int:",
        "input_format": "list[int]",
        "output_format": "int",
        "constraints": ["1 <= n <= 1e5"],
        "edge_cases_hinted": ["empty input"],
        "anti_homogeneity_notes": "baseline",
        "input_scale_class": "1e5-sequence",
        "data_shape_class": "flat-array",
        "primary_pitfall": "off-by-one",
        "novelty_reason": "Existing baseline sequence problem.",
    }
    output_path.write_text(json.dumps(existing) + "\n")
    client = FakeProvider(
        outputs=[
            [
                {
                    "id": "S00001",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "new scenario",
                    "problem_statement": "alpha beta gamma delta epsilon zeta eta lambda",
                    "function_signature": "def solve_new(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "similar",
                    "input_scale_class": "1e5-sequence",
                    "data_shape_class": "flat-array",
                    "primary_pitfall": "off-by-one",
                    "novelty_reason": "Still too close to the prior sequence pattern.",
                }
            ],
            [
                {
                    "id": "S00002",
                    "weakness_id": "W001",
                    "language": "python",
                    "difficulty": "hard",
                    "scenario": "fresh scenario",
                    "problem_statement": "c" * 240,
                    "function_signature": "def solve_fresh(items: list[int]) -> int:",
                    "input_format": "list[int]",
                    "output_format": "int",
                    "constraints": ["1 <= n <= 1e5"],
                    "edge_cases_hinted": ["empty input"],
                    "anti_homogeneity_notes": "fresh",
                    "input_scale_class": "event-stream",
                    "data_shape_class": "nested-records",
                    "primary_pitfall": "ordering-stability",
                    "novelty_reason": "Moves to record ordering rather than sequence overlap.",
                }
            ],
        ]
    )

    result = await synthesize_for_weaknesses(
        make_weakness_set(),
        allocations={"W001": 2},
        output_path=output_path,
        provider="openai",
        model="test-model",
        provider_client=client,
    )

    assert result.retry_count >= 1
    lines = [json.loads(line) for line in output_path.read_text().strip().splitlines()]
    assert lines[-1]["id"] == "S00002"


@pytest.mark.asyncio
async def test_synthesize_problems_rejects_non_array_payload_with_clear_error(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(
        outputs=[
            {"problems": []}
        ]
    )

    with pytest.raises(ValueError, match=r"synthesize_for_weaknesses expected JSON array payload"):
        await synthesize_for_weaknesses(
            make_weakness_set(),
            allocations={"W001": 1},
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=client,
        )


@pytest.mark.asyncio
async def test_synthesize_problems_rejects_empty_top_level_batch_payload(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    client = FakeProvider(outputs=[[]])

    with pytest.raises(ValueError, match=r"synthesize_for_weaknesses expected non-empty JSON array payload"):
        await synthesize_for_weaknesses(
            make_weakness_set(),
            allocations={"W001": 1},
            output_path=output_path,
            provider="openai",
            model="test-model",
            provider_client=client,
        )


def test_has_high_similarity_detects_ngram_overlap():
    existing = [
        {
            "problem_statement": "alpha beta gamma delta epsilon zeta eta theta",
        }
    ]
    candidate = "alpha beta gamma delta epsilon zeta eta lambda"
    assert has_high_similarity(candidate, existing) is True
