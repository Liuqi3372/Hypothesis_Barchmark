from mrrs.solver import ReferenceCoverage, exact_minimum_cover, removal_test


def coverage(ref_id, units, confidence=1.0):
    return ReferenceCoverage(ref_id, frozenset(units), confidence)


def test_exact_solver_finds_global_minimum_not_greedy_local_choice():
    references = [
        coverage("R1", {"U1", "U2"}),
        coverage("R2", {"U1", "U3"}),
        coverage("R3", {"U2", "U4"}),
        coverage("R4", {"U3", "U4"}),
    ]
    selected, uncovered = exact_minimum_cover(["U1", "U2", "U3", "U4"], references)
    assert len(selected) == 2
    assert uncovered == set()


def test_uncoverable_units_are_reported():
    selected, uncovered = exact_minimum_cover(
        ["U1", "U2"], [coverage("R1", {"U1"})]
    )
    assert selected == ["R1"]
    assert uncovered == {"U2"}


def test_removal_test_reports_only_newly_uncovered_units():
    selected = [coverage("R1", {"U1"}), coverage("R2", {"U2"})]
    result = removal_test(["U1", "U2", "U3"], selected)
    assert result[0]["newly_uncovered_unit_ids"] == ["U1"]
    assert result[1]["newly_uncovered_unit_ids"] == ["U2"]
    assert "U3" not in result[0]["newly_uncovered_unit_ids"]
