"""Internal data structures for weakness clustering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusterUnit:
    question_id: int
    error_tags: list[str]
    root_cause: str
    ability_dimensions: list[str]
    language: str
    category: str
    one_line_content: str


@dataclass
class CandidateCluster:
    candidate_id: str
    member_question_ids: list[int]
    member_tags: list[str]
    representative_units: list[ClusterUnit]
    dominant_language: str
    dominant_category: str


@dataclass
class RefinedCluster:
    refined_id: str
    name: str
    description: str
    covered_tags: list[str]
    member_question_ids: list[int]
    representative_units: list[ClusterUnit]
    dominant_language: str
    dominant_category: str
