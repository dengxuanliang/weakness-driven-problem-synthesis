"""Deduplication helpers for synthesized problems."""

from collections.abc import Mapping


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def ngram_jaccard(left: str, right: str, n: int = 4) -> float:
    left_ngrams = _ngrams(left, n)
    right_ngrams = _ngrams(right, n)
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    if not union:
        return 0.0
    intersection = left_ngrams & right_ngrams
    return len(intersection) / len(union)


def duplicate_key(problem: Mapping[str, str]) -> tuple[str, str]:
    return (problem["scenario"], problem["function_signature"])
