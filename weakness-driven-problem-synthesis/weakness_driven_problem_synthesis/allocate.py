"""Quota allocation for weakness-targeted synthesis."""


def allocate_quotas(raw_counts: dict[str, int], total_questions: int) -> dict[str, int]:
    if total_questions <= 0:
        raise ValueError("total_questions must be positive")
    if not raw_counts:
        raise ValueError("raw_counts must not be empty")

    total_raw = sum(raw_counts.values())
    if total_raw <= 0:
        raise ValueError("raw_counts must sum to a positive value")

    floor = max(20, round(total_questions * 0.005))
    ceil = round(total_questions * 0.08)

    allocations: dict[str, int] = {}
    for weakness_id, raw in raw_counts.items():
        share = raw / total_raw
        target = round(share * total_questions)
        allocations[weakness_id] = min(max(target, floor), ceil)

    largest_id = max(raw_counts, key=raw_counts.get)
    delta = total_questions - sum(allocations.values())
    allocations[largest_id] += delta
    return allocations
