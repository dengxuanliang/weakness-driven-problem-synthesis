# Weakness-Driven Problem Synthesis — Design Spec

**Date**: 2026-05-06
**Status**: Draft for review
**Skill name**: `weakness-driven-problem-synthesis`

## 1. Goal

Given an evaluation log of an LLM on coding problems (jsonl, schema same as
`glm51_eval_inference_merged.jsonl`), produce a large batch (typically several
thousand) of **high-difficulty, deduplicated coding problem statements** that
specifically target the model's weaknesses — not the original failed problems.

The skill outputs only **problem statements** (no reference solutions, no test
cases). Generating solutions/tests is explicitly out of scope and should be
handled by a separate downstream skill.

## 2. Inputs and Outputs

### 2.1 Inputs

| Param             | Required | Default                  | Notes                                                              |
| ----------------- | -------- | ------------------------ | ------------------------------------------------------------------ |
| `eval_log_path`   | yes      | —                        | Path to jsonl. Must contain `content`, `canonical_solution`, `completion`, `test`, `labels`, `pass_at_1`. |
| `total_questions` | yes      | —                        | Target total count of synthesized problem statements.              |
| `output_dir`      | no       | `./synthesis_output/`    | All artifacts written here.                                        |
| `provider`        | no       | `anthropic`              | `anthropic` or `openai`.                                           |
| `model`           | no       | provider default         | Overrides default model (`claude-opus-4-6` / `gpt-4o`).            |
| `concurrency`     | no       | `8`                      | Max parallel LLM calls in attribution stage.                       |
| `resume`          | no       | `true`                   | Reuse existing stage artifacts in `output_dir`.                    |

API keys come from environment variables: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional, for OpenAI-compatible endpoints).

### 2.2 Outputs (under `output_dir/`)

- `error_attributions.jsonl` — one record per failed question (per-question root-cause analysis).
- `weaknesses.json` — clustered weakness list (15–40 items).
- `synthesized_problems.jsonl` — final problem statements.
- `report.md` — human-readable summary.

## 3. Pipeline

```
load_filter → attribute → cluster → allocate → synthesize → report
```

### 3.1 Load & Filter

Read jsonl line by line. Keep records where `pass_at_1 < 1` (treat `False`,
`0`, `0.0`, `None` as failed). Output is the in-memory failed set.

### 3.2 Per-question Attribution (static analysis, LLM-driven)

For each failed record, send a prompt containing `content`,
`canonical_solution`, `completion`, `labels.{category,programming_language,difficulty}`,
and (for context only) `test`. The LLM returns:

```json
{
  "question_id": 229,
  "is_truly_failed": true,
  "error_tags": ["api-misuse:beautifulsoup-class_", "edge-case:empty-input"],
  "root_cause": "...",
  "ability_dimensions": ["library API details", "Python keyword handling"],
  "evidence_snippet": "spans = soup.find_all('span', class='title')"
}
```

Rules:

- `error_tags` use a controlled vocabulary `<dimension>:<detail>` (kebab-case).
  The prompt provides `references/error_tag_vocabulary.md` few-shot examples
  and the running set of tags already seen, asking the LLM to reuse existing
  tags when possible.
- `is_truly_failed=false` is used to drop suspected scoring false-positives
  (e.g. semantically equivalent correct answers misjudged). These are excluded
  from clustering.
- Concurrency: `asyncio` semaphore, default 8.
- Each record is appended to `error_attributions.jsonl` immediately on success
  to support resume.

### 3.3 Weakness Clustering (LLM-driven)

No external embeddings. Procedure:

1. Aggregate all `error_tags` (deduplicated) and, for each tag, attach up to
   3 short representative question summaries (`{id, category, language, one-line content}`).
2. Send the tag-with-evidence list to the LLM with instructions to produce
   15–40 weakness clusters:

   ```json
   {
     "id": "W007",
     "name": "Recursion and backtracking termination conditions",
     "description": "...",
     "covered_tags": ["recursion:base-case-missing", "..."],
     "dominant_language": "python",
     "dominant_category": "Advanced Programming"
   }
   ```

3. Map each failed question to its weakness via the tag → cluster relation,
   building `evidence_question_ids[w]`. A question can map to multiple
   weaknesses if it carries tags from several clusters; for quota purposes it
   is counted in each.

Output: `weaknesses.json` (the clusters plus the evidence map).

### 3.4 Quota Allocation

```
raw[w]   = len(evidence_question_ids[w])
share[w] = raw[w] / sum(raw)
floor    = max(20, round(total_questions * 0.005))
ceil     = round(total_questions * 0.08)
alloc[w] = clamp(round(share[w] * total_questions), floor, ceil)
```

After clamping, the largest cluster absorbs the rounding/clamping delta so
`sum(alloc) == total_questions`. Constants `floor` and `ceil` are kept in a
single config block for easy tuning.

Rationale: floor (≥20 or 0.5%) prevents under-represented but real weaknesses
from disappearing; ceil (8%) prevents one dominant weakness from monopolizing
the synthesis batch.

### 3.5 Synthesis

For each weakness `w`, generate `alloc[w]` problems in **batches of N=10**.

Per-batch prompt contains:

- Weakness `name`, `description`, `dominant_language` (used as the language).
- A summary list of all problems already produced for this weakness in prior
  batches (id + one-sentence scenario), with explicit instruction to avoid
  overlapping.
- In-batch deduplication instruction: the 10 problems within this batch must
  differ along four axes — application scenario, input scale, data shape,
  primary pitfall.
- Difficulty contract: hard. Multi-step reasoning, multiple constraints, easy-
  to-miss edge cases.
- Strict output JSON schema (per problem):

  ```json
  {
    "id": "S00001",
    "weakness_id": "W007",
    "language": "python",
    "difficulty": "hard",
    "scenario": "real-time log compaction",
    "problem_statement": "<full natural-language problem>",
    "function_signature": "def compress_log(events: list[dict]) -> list[dict]:",
    "input_format": "...",
    "output_format": "...",
    "constraints": ["1 <= len(events) <= 1e5", "..."],
    "edge_cases_hinted": ["empty input", "duplicate timestamps"],
    "anti_homogeneity_notes": "differs from S00003: sliding window vs full bucketing",
    "input_scale_class": "1e5-event-stream",
    "data_shape_class": "nested-records",
    "primary_pitfall": "ordering-stability",
    "novelty_reason": "Focuses on stable online compaction rather than batch aggregation."
  }
  ```

Additional structured diversity fields are part of the persisted problem schema:

- `input_scale_class` — short label for scale profile, e.g. `1e5-sequence`,
  `sparse-graph`, `event-stream`
- `data_shape_class` — short label for dominant structure, e.g. `flat-array`,
  `nested-records`, `tree`, `graph`
- `primary_pitfall` — short label for the main failure mode the problem is
  designed to trigger
- `novelty_reason` — one-sentence explanation of why the problem is not
  redundant with prior or same-batch items

Ordering: weaknesses run **sequentially** (one weakness at a time). Inside a
weakness, batches are also sequential so each new batch sees the running
summary of all prior problems for that weakness. (Per-weakness parallelism
was explicitly rejected during design discussion in favor of cleaner
deduplication context; revisit only if wall-clock cost becomes blocking.)

### 3.6 QA on synthesized problems

Two layers, lightweight only:

1. **Format check** per problem:
   - Valid JSON; all required fields present.
   - `language` matches the weakness's `dominant_language`.
   - `problem_statement` length ≥ a minimum threshold (e.g. 200 chars).
   Failure → drop the offending problem and ask the model to regenerate just
   that slot in the next attempt for this weakness.

2. **Cross-batch dedup** within a weakness:
   - Key 1: `(scenario, function_signature)` — exact duplicate → regenerate.
   - Key 2: token n-gram (n=4) Jaccard similarity over `problem_statement`.
     Threshold ≥ 0.6 against any prior problem in the same weakness →
     regenerate.
   - Per-slot retry budget: 3. After 3 failures the slot is dropped and an
     extra batch is queued at the end of the weakness to refill the quota.
   - Cross-weakness dedup: only `(scenario, function_signature)` exact-key
     check (avoids excessive cross-weakness similarity work).

No execution of code. No reference solutions. No test cases.

### 3.7 Report

`report.md` contains:

- Overall counts: failed questions, weaknesses, synthesized problems, dropped.
- Top-N weakness table: name, evidence count, allocated quota, completed.
- One sampled problem statement (first ~200 chars) per weakness.
- Skip / retry / drop counters.

## 4. Engineering Layout

Skill installed at `~/.claude/skills/weakness-driven-problem-synthesis/`:

```
SKILL.md
scripts/
  run.py                # CLI entry: python run.py --eval-log ... --total ...
  load_filter.py
  attribute.py
  cluster.py
  allocate.py
  synthesize.py
  llm_client.py         # provider abstraction
  dedup.py              # n-gram Jaccard
  schemas.py            # pydantic models for all I/O
references/
  error_tag_vocabulary.md
  prompts/
    attribute.txt
    cluster.txt
    synthesize.txt
```

### 4.1 LLM client abstraction

`llm_client.py` exposes:

```python
async def complete_json(prompt: str, schema: dict, *, system: str | None = None,
                        max_tokens: int = 4096) -> dict | list[dict]: ...
```

- `provider=anthropic` → `anthropic` SDK, default model `claude-opus-4-6`.
- `provider=openai`    → `openai` SDK, `base_url=os.getenv("OPENAI_BASE_URL")`,
                         default model `gpt-4o`.
- JSON parse retry: up to 3 attempts; on failure the previous raw output is
  appended to the next prompt asking the model to repair to valid JSON.
- Rate-limit / 5xx: exponential backoff (1s, 2s, 4s, 8s; max 5 attempts).
- Keys read from env: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Missing key →
  fail fast with a clear message.

### 4.2 Resumability

- Each stage writes to its own file. On startup, if the file exists and
  `--resume` (default true), the stage skips already-completed items:
  - `error_attributions.jsonl`: dedup by `question_id`.
  - `weaknesses.json`: present → skip clustering entirely.
  - `synthesized_problems.jsonl`: dedup by `(weakness_id, batch_index)`;
    weaknesses with `alloc` already met are skipped.
- `--restart` deletes prior artifacts and re-runs from scratch.
- Any stage that fails non-recoverably preserves prior artifacts and exits
  with a non-zero code and an actionable message.

### 4.3 Concurrency

- Attribution: `asyncio.Semaphore(concurrency)`.
- Synthesis: serial across weaknesses; serial across batches within a
  weakness. `concurrency` does not apply here.

### 4.4 SKILL.md trigger

Frontmatter `description`:

> Use when the user provides a model evaluation jsonl log (with
> content/completion/canonical_solution/pass_at_1/labels) and asks to
> synthesize high-difficulty, deduplicated coding problem statements that
> target the model's weaknesses.

### 4.5 Schemas

`schemas.py` defines pydantic models for: `EvalRecord`, `Attribution`,
`Weakness`, `WeaknessSet`, `SynthProblem`. All disk I/O passes through these.

## 5. Out of Scope

- Generating reference solutions for synthesized problems.
- Generating test cases for synthesized problems.
- Executing any code (model completions or canonical solutions).
- Embedding-based similarity (kept intentionally simple with n-gram Jaccard).
- Re-evaluating the original model on the synthesized problems.

## 6. Open Risks

- **Tag vocabulary drift**: many singleton tags would weaken clustering. Mitigation:
  vocabulary file as anchor + clustering prompt explicitly merges near-duplicates.
- **LLM clustering instability**: same input may produce different cluster
  counts/names across runs. Mitigation: clustering is deterministic per
  `output_dir` once `weaknesses.json` exists (resume semantics).
- **Quota refill loops**: pathological weaknesses where the model keeps
  producing duplicates could loop. Mitigation: per-weakness max additional
  batches = 2; if still short, the report flags the shortfall instead of
  looping forever.
- **Cost**: thousands of synthesis calls at batch=10 means hundreds of LLM
  calls. The skill prints a pre-flight estimate (#attribution calls +
  #synthesis batches) and asks for confirmation before starting synthesis.
