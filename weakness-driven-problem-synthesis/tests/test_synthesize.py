import json

import pytest

from weakness_driven_problem_synthesis.schemas import SynthesisSummary, Weakness, WeaknessSet
from weakness_driven_problem_synthesis.synthesize import has_high_similarity, synthesize_for_weaknesses


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
    )
    assert len(client.calls) == 0


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

    assert "Prior problems summary" in client.calls[1]["prompt"]
    assert "scenario one" in client.calls[1]["prompt"]


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


def test_has_high_similarity_detects_ngram_overlap():
    existing = [
        {
            "problem_statement": "alpha beta gamma delta epsilon zeta eta theta",
        }
    ]
    candidate = "alpha beta gamma delta epsilon zeta eta lambda"
    assert has_high_similarity(candidate, existing) is True
