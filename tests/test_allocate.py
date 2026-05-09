from weakness_driven_problem_synthesis.allocate import allocate_quotas


def test_allocate_quotas_preserves_total():
    alloc = allocate_quotas({"W001": 80, "W002": 20}, total_questions=200)
    assert sum(alloc.values()) == 200
    assert alloc["W001"] > alloc["W002"]


def test_allocate_quotas_applies_floor_and_ceil():
    alloc = allocate_quotas({"W001": 999, "W002": 1}, total_questions=1000)
    assert sum(alloc.values()) == 1000
    assert alloc["W001"] > alloc["W002"]


def test_allocate_quotas_small_total_many_weaknesses_stays_non_negative_and_exact():
    raw_counts = {f"W{i:03d}": 1 for i in range(12)}
    alloc = allocate_quotas(raw_counts, total_questions=20)

    assert sum(alloc.values()) == 20
    assert all(value >= 0 for value in alloc.values())
    assert all(value in {1, 2} for value in alloc.values())
