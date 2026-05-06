# Weakness-Driven Problem Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable skill that reads an evaluation jsonl log and produces high-difficulty, deduplicated coding problem statements targeting observed model weaknesses.

**Architecture:** Implement a repo-local skill package at `weakness-driven-problem-synthesis/` that mirrors the final installed skill layout from the spec. The pipeline stays file-oriented and stage-based: load/filter failed eval records, attribute root causes with an async LLM client, cluster weaknesses, allocate quotas, synthesize problems with lightweight QA/dedup, then emit a markdown report. Every stage is resumable through stage artifacts under a user-provided output directory.

**Tech Stack:** Python 3.11+, pydantic, pytest, pytest-asyncio, anthropic SDK, openai SDK

---

## File Structure

- Create: `weakness-driven-problem-synthesis/SKILL.md`
- Create: `weakness-driven-problem-synthesis/pyproject.toml`
- Create: `weakness-driven-problem-synthesis/scripts/run.py`
- Create: `weakness-driven-problem-synthesis/scripts/load_filter.py`
- Create: `weakness-driven-problem-synthesis/scripts/attribute.py`
- Create: `weakness-driven-problem-synthesis/scripts/cluster.py`
- Create: `weakness-driven-problem-synthesis/scripts/allocate.py`
- Create: `weakness-driven-problem-synthesis/scripts/synthesize.py`
- Create: `weakness-driven-problem-synthesis/scripts/report.py`
- Create: `weakness-driven-problem-synthesis/scripts/llm_client.py`
- Create: `weakness-driven-problem-synthesis/scripts/dedup.py`
- Create: `weakness-driven-problem-synthesis/scripts/schemas.py`
- Create: `weakness-driven-problem-synthesis/references/error_tag_vocabulary.md`
- Create: `weakness-driven-problem-synthesis/references/prompts/attribute.txt`
- Create: `weakness-driven-problem-synthesis/references/prompts/cluster.txt`
- Create: `weakness-driven-problem-synthesis/references/prompts/synthesize.txt`
- Create: `weakness-driven-problem-synthesis/tests/test_schemas.py`
- Create: `weakness-driven-problem-synthesis/tests/test_load_filter.py`
- Create: `weakness-driven-problem-synthesis/tests/test_allocate.py`
- Create: `weakness-driven-problem-synthesis/tests/test_dedup.py`
- Create: `weakness-driven-problem-synthesis/tests/test_attribute.py`
- Create: `weakness-driven-problem-synthesis/tests/test_cluster.py`
- Create: `weakness-driven-problem-synthesis/tests/test_synthesize.py`
- Create: `weakness-driven-problem-synthesis/tests/test_report.py`
- Create: `weakness-driven-problem-synthesis/tests/test_run.py`

### Task 1: Bootstrap The Skill Package

**Files:**
- Create: `weakness-driven-problem-synthesis/pyproject.toml`
- Create: `weakness-driven-problem-synthesis/SKILL.md`
- Create: `weakness-driven-problem-synthesis/scripts/__init__.py`
- Create: `weakness-driven-problem-synthesis/tests/conftest.py`

- [ ] **Step 1: Write the failing package smoke test**

```python
from pathlib import Path


def test_skill_package_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "SKILL.md").exists()
    assert (root / "scripts").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_run.py -k package -v`
Expected: FAIL with missing package files

- [ ] **Step 3: Create the minimal package skeleton**

```toml
[project]
name = "weakness-driven-problem-synthesis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "anthropic>=0.57.0",
  "openai>=1.99.0",
  "pydantic>=2.11.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.4.0",
  "pytest-asyncio>=1.1.0",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_run.py -k package -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/pyproject.toml weakness-driven-problem-synthesis/SKILL.md weakness-driven-problem-synthesis/scripts/__init__.py weakness-driven-problem-synthesis/tests/conftest.py weakness-driven-problem-synthesis/tests/test_run.py
git commit -m "chore: bootstrap weakness synthesis skill package"
```

### Task 2: Define Typed Schemas For All Disk I/O

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/schemas.py`
- Test: `weakness-driven-problem-synthesis/tests/test_schemas.py`

- [ ] **Step 1: Write the failing schema tests**

```python
from weakness_driven_problem_synthesis.scripts.schemas import EvalRecord, SynthProblem


def test_eval_record_normalizes_failed_pass_values():
    record = EvalRecord.model_validate({
        "question_id": 1,
        "content": "x",
        "canonical_solution": "y",
        "completion": "z",
        "test": "assert True",
        "labels": {"category": "algorithms", "programming_language": "python", "difficulty": "hard"},
        "pass_at_1": None,
    })
    assert record.is_failed is True


def test_synth_problem_requires_all_fields():
    SynthProblem.model_validate({
        "id": "S00001",
        "weakness_id": "W001",
        "language": "python",
        "difficulty": "hard",
        "scenario": "stream compaction",
        "problem_statement": "x" * 240,
        "function_signature": "def f(x: list[int]) -> int:",
        "input_format": "list of ints",
        "output_format": "int",
        "constraints": ["1 <= n <= 1e5"],
        "edge_cases_hinted": ["empty input"],
        "anti_homogeneity_notes": "unique angle",
    })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_schemas.py -v`
Expected: FAIL with import or validation errors

- [ ] **Step 3: Implement pydantic models and helpers**

```python
class EvalRecord(BaseModel):
    question_id: int | None = None
    content: str
    canonical_solution: str
    completion: str
    test: str
    labels: EvalLabels
    pass_at_1: bool | int | float | None

    @property
    def is_failed(self) -> bool:
        return self.pass_at_1 in (False, 0, 0.0, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/schemas.py weakness-driven-problem-synthesis/tests/test_schemas.py
git commit -m "feat: add typed schemas for synthesis pipeline"
```

### Task 3: Implement Load And Filter

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/load_filter.py`
- Test: `weakness-driven-problem-synthesis/tests/test_load_filter.py`

- [ ] **Step 1: Write the failing load/filter tests**

```python
def test_load_failed_records_filters_non_failures(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"content":"a","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":1}\n'
        '{"content":"b","canonical_solution":"x","completion":"y","test":"t","labels":{"category":"c","programming_language":"python","difficulty":"hard"},"pass_at_1":0}\n'
    )
    records = list(load_failed_records(path))
    assert len(records) == 1
    assert records[0].content == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_load_filter.py -v`
Expected: FAIL with missing loader

- [ ] **Step 3: Implement streaming jsonl loading**

```python
def load_failed_records(path: Path) -> Iterator[EvalRecord]:
    with path.open() as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = EvalRecord.model_validate_json(raw_line)
            if record.is_failed:
                yield record
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_load_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/load_filter.py weakness-driven-problem-synthesis/tests/test_load_filter.py
git commit -m "feat: add failed-record loader"
```

### Task 4: Implement Quota Allocation

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/allocate.py`
- Test: `weakness-driven-problem-synthesis/tests/test_allocate.py`

- [ ] **Step 1: Write the failing allocation tests**

```python
def test_allocate_quotas_preserves_total():
    alloc = allocate_quotas({"W001": 80, "W002": 20}, total_questions=200)
    assert sum(alloc.values()) == 200
    assert alloc["W001"] > alloc["W002"]


def test_allocate_quotas_applies_floor_and_ceil():
    alloc = allocate_quotas({"W001": 999, "W002": 1}, total_questions=1000)
    assert alloc["W002"] >= 20
    assert alloc["W001"] <= 80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_allocate.py -v`
Expected: FAIL with missing allocator

- [ ] **Step 3: Implement the quota math exactly from the spec**

```python
def allocate_quotas(raw_counts: dict[str, int], total_questions: int) -> dict[str, int]:
    floor = max(20, round(total_questions * 0.005))
    ceil = round(total_questions * 0.08)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_allocate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/allocate.py weakness-driven-problem-synthesis/tests/test_allocate.py
git commit -m "feat: add weakness quota allocation"
```

### Task 5: Build N-Gram Dedup Utilities

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/dedup.py`
- Test: `weakness-driven-problem-synthesis/tests/test_dedup.py`

- [ ] **Step 1: Write the failing dedup tests**

```python
def test_ngram_jaccard_detects_high_similarity():
    a = "alpha beta gamma delta epsilon zeta eta theta"
    b = "alpha beta gamma delta epsilon zeta eta lambda"
    assert ngram_jaccard(a, b, n=4) >= 0.6


def test_duplicate_key_uses_scenario_and_signature():
    problem = {"scenario": "payments reconciliation", "function_signature": "def f(x: list[int]) -> int:"}
    assert duplicate_key(problem) == ("payments reconciliation", "def f(x: list[int]) -> int:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_dedup.py -v`
Expected: FAIL with missing dedup helpers

- [ ] **Step 3: Implement similarity and exact-key helpers**

```python
def ngram_jaccard(left: str, right: str, n: int = 4) -> float:
    ...


def duplicate_key(problem: Mapping[str, str]) -> tuple[str, str]:
    return (problem["scenario"], problem["function_signature"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_dedup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/dedup.py weakness-driven-problem-synthesis/tests/test_dedup.py
git commit -m "feat: add synthesized problem dedup helpers"
```

### Task 6: Build The Provider-Agnostic Async LLM Client

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/llm_client.py`
- Test: `weakness-driven-problem-synthesis/tests/test_attribute.py`

- [ ] **Step 1: Write the failing client tests**

```python
@pytest.mark.asyncio
async def test_complete_json_retries_invalid_json(monkeypatch):
    client = FakeProvider(outputs=["not json", '{"ok": true}'])
    result = await complete_json("prompt", {"type": "object"}, provider_client=client, model="test-model")
    assert result == {"ok": True}


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_provider_client(provider="openai", model=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_attribute.py -k client -v`
Expected: FAIL with missing client implementation

- [ ] **Step 3: Implement client selection, JSON repair, and retry policy**

```python
async def complete_json(prompt: str, schema: dict, *, system: str | None = None,
                        max_tokens: int = 4096, provider: str = "anthropic",
                        model: str | None = None, provider_client: Any | None = None) -> dict | list[dict]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_attribute.py -k client -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/llm_client.py weakness-driven-problem-synthesis/tests/test_attribute.py
git commit -m "feat: add async llm client abstraction"
```

### Task 7: Implement Attribution With Resume-Aware Appends

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/attribute.py`
- Create: `weakness-driven-problem-synthesis/references/error_tag_vocabulary.md`
- Create: `weakness-driven-problem-synthesis/references/prompts/attribute.txt`
- Test: `weakness-driven-problem-synthesis/tests/test_attribute.py`

- [ ] **Step 1: Write the failing attribution tests**

```python
@pytest.mark.asyncio
async def test_attribute_failures_appends_one_json_line_per_record(tmp_path):
    output_path = tmp_path / "error_attributions.jsonl"
    ...
    await attribute_failures(records, output_path=output_path, concurrency=2, ...)
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_attribute_failures_skips_already_processed_question_ids(tmp_path):
    output_path.write_text('{"question_id": 7, "is_truly_failed": true, "error_tags": ["x:y"], "root_cause": "r", "ability_dimensions": ["a"], "evidence_snippet": "e"}\n')
    ...
    assert only_unfinished_records_are_sent is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_attribute.py -k attribution -v`
Expected: FAIL with missing attribution pipeline

- [ ] **Step 3: Implement prompt assembly, semaphore control, and append-on-success writes**

```python
async def attribute_failures(records: Sequence[EvalRecord], *, output_path: Path,
                             provider: str, model: str | None, concurrency: int) -> list[Attribution]:
    processed_ids = load_processed_question_ids(output_path)
    semaphore = asyncio.Semaphore(concurrency)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_attribute.py -k attribution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/attribute.py weakness-driven-problem-synthesis/references/error_tag_vocabulary.md weakness-driven-problem-synthesis/references/prompts/attribute.txt weakness-driven-problem-synthesis/tests/test_attribute.py
git commit -m "feat: add failure attribution stage"
```

### Task 8: Implement Weakness Clustering And Evidence Mapping

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/cluster.py`
- Create: `weakness-driven-problem-synthesis/references/prompts/cluster.txt`
- Test: `weakness-driven-problem-synthesis/tests/test_cluster.py`

- [ ] **Step 1: Write the failing clustering tests**

```python
@pytest.mark.asyncio
async def test_cluster_weaknesses_writes_resume_artifact(tmp_path):
    output_path = tmp_path / "weaknesses.json"
    result = await cluster_weaknesses(attributions, output_path=output_path, provider="openai", model="test-model")
    assert output_path.exists()
    assert result.weaknesses[0].id == "W001"


def test_map_questions_to_clusters_counts_multi_cluster_membership():
    mapping = map_questions_to_clusters(attributions, weaknesses)
    assert mapping["W001"] == [1, 3]
    assert mapping["W002"] == [3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_cluster.py -v`
Expected: FAIL with missing cluster implementation

- [ ] **Step 3: Implement tag aggregation, representative summaries, cluster call, and evidence mapping**

```python
async def cluster_weaknesses(attributions: Sequence[Attribution], *, output_path: Path,
                             provider: str, model: str | None) -> WeaknessSet:
    if output_path.exists():
        return WeaknessSet.model_validate_json(output_path.read_text())
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_cluster.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/cluster.py weakness-driven-problem-synthesis/references/prompts/cluster.txt weakness-driven-problem-synthesis/tests/test_cluster.py
git commit -m "feat: add weakness clustering stage"
```

### Task 9: Implement Synthesis QA, Refill Logic, And Output Streaming

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/synthesize.py`
- Create: `weakness-driven-problem-synthesis/references/prompts/synthesize.txt`
- Test: `weakness-driven-problem-synthesis/tests/test_synthesize.py`

- [ ] **Step 1: Write the failing synthesis tests**

```python
@pytest.mark.asyncio
async def test_synthesize_problems_respects_existing_batches_on_resume(tmp_path):
    output_path = tmp_path / "synthesized_problems.jsonl"
    output_path.write_text(existing_batch_jsonl)
    result = await synthesize_for_weaknesses(...)
    assert skipped_completed_batches is True


@pytest.mark.asyncio
async def test_synthesize_problems_regenerates_duplicates_and_short_statements(tmp_path):
    result = await synthesize_for_weaknesses(...)
    assert result.completed == target_quota
    assert result.retry_count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_synthesize.py -v`
Expected: FAIL with missing synthesis implementation

- [ ] **Step 3: Implement sequential weakness processing, per-batch generation, validation, retry budget, and extra-batch refill**

```python
async def synthesize_for_weaknesses(weakness_set: WeaknessSet, *, total_questions: int,
                                    output_path: Path, provider: str, model: str | None) -> SynthesisSummary:
    for weakness in weakness_set.weaknesses:
        await synthesize_weakness_batches(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_synthesize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/synthesize.py weakness-driven-problem-synthesis/references/prompts/synthesize.txt weakness-driven-problem-synthesis/tests/test_synthesize.py
git commit -m "feat: add weakness-targeted synthesis pipeline"
```

### Task 10: Implement Markdown Reporting

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/report.py`
- Test: `weakness-driven-problem-synthesis/tests/test_report.py`

- [ ] **Step 1: Write the failing report tests**

```python
def test_write_report_includes_counts_and_sampled_problem(tmp_path):
    report_path = tmp_path / "report.md"
    write_report(report_path=report_path, ...)
    text = report_path.read_text()
    assert "Overall counts" in text
    assert "W001" in text
    assert "first ~200 chars" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_report.py -v`
Expected: FAIL with missing report writer

- [ ] **Step 3: Implement markdown summary rendering**

```python
def write_report(*, report_path: Path, failed_count: int, weakness_set: WeaknessSet,
                 synthesis_summary: SynthesisSummary) -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/report.py weakness-driven-problem-synthesis/tests/test_report.py
git commit -m "feat: add synthesis report generation"
```

### Task 11: Wire The CLI And Stage Orchestration

**Files:**
- Create: `weakness-driven-problem-synthesis/scripts/run.py`
- Test: `weakness-driven-problem-synthesis/tests/test_run.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
def test_cli_parses_expected_arguments():
    args = build_parser().parse_args(["--eval-log", "eval.jsonl", "--total-questions", "500"])
    assert args.eval_log == "eval.jsonl"
    assert args.total_questions == 500
    assert args.resume is True


def test_restart_deletes_stage_artifacts(tmp_path):
    ...
    assert not old_artifact.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_run.py -k "cli or restart" -v`
Expected: FAIL with missing CLI orchestration

- [ ] **Step 3: Implement argument parsing, pre-flight estimates, restart behavior, and stage calls**

```python
async def main() -> int:
    args = build_parser().parse_args()
    output_dir = prepare_output_dir(args.output_dir, restart=args.restart)
    failed_records = list(load_failed_records(Path(args.eval_log)))
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest weakness-driven-problem-synthesis/tests/test_run.py -k "cli or restart" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add weakness-driven-problem-synthesis/scripts/run.py weakness-driven-problem-synthesis/tests/test_run.py
git commit -m "feat: add resumable synthesis cli"
```

### Task 12: Run End-To-End Verification And Final Documentation Pass

**Files:**
- Modify: `weakness-driven-problem-synthesis/SKILL.md`
- Modify: `weakness-driven-problem-synthesis/references/prompts/attribute.txt`
- Modify: `weakness-driven-problem-synthesis/references/prompts/cluster.txt`
- Modify: `weakness-driven-problem-synthesis/references/prompts/synthesize.txt`

- [ ] **Step 1: Write the final integration test**

```python
@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end_with_stubbed_llm(tmp_path):
    ...
    exit_code = await main_with_args([...])
    assert exit_code == 0
    assert (tmp_path / "out" / "report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest weakness-driven-problem-synthesis/tests/test_run.py -k end_to_end -v`
Expected: FAIL until stage wiring and prompts are coherent

- [ ] **Step 3: Finish prompt text, trigger description, and integration gaps**

```markdown
description: Use when the user provides a model evaluation jsonl log ... and asks to synthesize high-difficulty, deduplicated coding problem statements that target the model's weaknesses.
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest weakness-driven-problem-synthesis/tests -v`
Expected: PASS

- [ ] **Step 5: Run a local smoke command against the provided eval log**

Run: `python weakness-driven-problem-synthesis/scripts/run.py --eval-log glm51_eval_inference_merged.jsonl --total-questions 50 --output-dir ./tmp/weakness-synthesis-smoke --provider openai --model gpt-4o`
Expected: Pre-flight estimate is printed, artifacts are created under `./tmp/weakness-synthesis-smoke/`, and the command exits `0` when valid API credentials are present

- [ ] **Step 6: Commit**

```bash
git add weakness-driven-problem-synthesis/SKILL.md weakness-driven-problem-synthesis/references/prompts/attribute.txt weakness-driven-problem-synthesis/references/prompts/cluster.txt weakness-driven-problem-synthesis/references/prompts/synthesize.txt weakness-driven-problem-synthesis/tests/test_run.py
git commit -m "feat: finalize weakness synthesis skill"
```

## Notes

- Keep module imports package-relative from `weakness-driven-problem-synthesis/scripts/` so tests can stub providers without network access.
- Use append-only writes for `error_attributions.jsonl` and `synthesized_problems.jsonl`; never rewrite the entire file during normal resume flow.
- Keep constants for synthesis QA in one place inside `synthesize.py`: `BATCH_SIZE = 10`, `MIN_STATEMENT_CHARS = 200`, `NGRAM_N = 4`, `SIMILARITY_THRESHOLD = 0.6`, `PER_SLOT_RETRY_LIMIT = 3`, `MAX_EXTRA_BATCHES = 2`.
- Fail fast on missing API credentials, malformed stage files, or unsupported providers with explicit error messages that include the offending path or provider name.
- If `weakness-driven-problem-synthesis/scripts/` needs shared helpers during implementation, add `scripts/io_utils.py` in a separate small commit instead of growing unrelated modules.

## Manual Review

- I did not run the `writing-plans` subagent review loop because this session does not have explicit delegation requested by the user.
- Before execution starts, do a quick human pass on package naming. If Python import ergonomics become awkward because of the hyphenated repo-local directory, switch the importable package to `weakness_driven_problem_synthesis/` nested inside the skill root while keeping the external skill folder name unchanged.
