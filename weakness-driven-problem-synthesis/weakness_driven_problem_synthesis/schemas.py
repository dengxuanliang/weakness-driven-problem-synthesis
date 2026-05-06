"""Schema definitions for the synthesis pipeline."""

from pydantic import BaseModel


class EvalLabels(BaseModel):
    category: str
    programming_language: str
    difficulty: str


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


class SynthProblem(BaseModel):
    id: str
    weakness_id: str
    language: str
    difficulty: str
    scenario: str
    problem_statement: str
    function_signature: str
    input_format: str
    output_format: str
    constraints: list[str]
    edge_cases_hinted: list[str]
    anti_homogeneity_notes: str
    input_scale_class: str
    data_shape_class: str
    primary_pitfall: str
    novelty_reason: str


class Attribution(BaseModel):
    question_id: int
    is_truly_failed: bool
    error_tags: list[str]
    root_cause: str
    ability_dimensions: list[str]
    evidence_snippet: str


class Weakness(BaseModel):
    id: str
    name: str
    description: str
    covered_tags: list[str]
    dominant_language: str
    dominant_category: str


class WeaknessSet(BaseModel):
    weaknesses: list[Weakness]
    evidence_question_ids: dict[str, list[int]]


class SynthesisSummary(BaseModel):
    completed: int
    retry_count: int
    dropped: int = 0
    skipped: int = 0
    extra_batches: int = 0
    completed_by_weakness: dict[str, int] = {}
    shortfall_by_weakness: dict[str, int] = {}
