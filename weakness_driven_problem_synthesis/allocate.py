"""Quota allocation for weakness-targeted synthesis."""

import math


def allocate_quotas(raw_counts: dict[str, int], total_questions: int) -> dict[str, int]:
    if total_questions <= 0:
        raise ValueError("total_questions must be positive")
    if not raw_counts:
        raise ValueError("raw_counts must not be empty")

    total_raw = sum(raw_counts.values())
    if total_raw <= 0:
        raise ValueError("raw_counts must sum to a positive value")

    exact_targets: dict[str, float] = {
        weakness_id: (raw / total_raw) * total_questions for weakness_id, raw in raw_counts.items()
    }
    allocations = {weakness_id: math.floor(target) for weakness_id, target in exact_targets.items()}

    remaining = total_questions - sum(allocations.values())
    remainders = sorted(
        ((exact_targets[weakness_id] - allocations[weakness_id], weakness_id) for weakness_id in raw_counts),
        key=lambda item: (-item[0], item[1]),
    )
    for _, weakness_id in remainders[:remaining]:
        allocations[weakness_id] += 1

    return allocations
