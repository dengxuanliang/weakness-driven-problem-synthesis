from weakness_driven_problem_synthesis.allocate import allocate_quotas


def test_allocate_quotas_preserves_total():
    alloc = allocate_quotas({"W001": 80, "W002": 20}, total_questions=200)
    assert sum(alloc.values()) == 200
    assert alloc["W001"] > alloc["W002"]


def test_allocate_quotas_applies_floor_and_ceil():
    alloc = allocate_quotas({"W001": 999, "W002": 1}, total_questions=1000)
    assert alloc["W002"] >= 20
    assert sum(alloc.values()) == 1000
    assert alloc["W001"] == 980
