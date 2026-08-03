from collections import Counter

from pmc_m.pmc import _topic_coverage_sample


def test_topic_coverage_is_unique_and_assigns_one_primary_category():
    selected, assignments, report = _topic_coverage_sample(
        {
            "A": ["1", "2", "3", "8", "9", "10"],
            "B": ["1", "4", "5"],
            "C": ["6", "7"],
        },
        limit=7,
        min_per_category=1,
    )

    assert len(selected) == len(set(selected)) == 7
    assert set(assignments) == set(selected)
    assert all(assignments[pmc_id] in {"A", "B", "C"} for pmc_id in selected)
    assert all(value >= 1 for value in report["sampled_by_primary_category"].values())


def test_remaining_budget_approximately_follows_candidate_distribution():
    selected, assignments, report = _topic_coverage_sample(
        {
            "large": [str(number) for number in range(1, 81)],
            "medium": [str(number) for number in range(81, 101)],
            "small": [str(number) for number in range(101, 111)],
        },
        limit=50,
        min_per_category=2,
    )
    counts = Counter(assignments.values())

    assert len(selected) == 50
    assert counts["large"] > counts["medium"] > counts["small"] >= 2
    assert sum(report["sampled_by_primary_category"].values()) == 50


def test_shared_paper_is_never_selected_or_assigned_twice():
    selected, assignments, _ = _topic_coverage_sample(
        {
            "A": ["shared", "a1", "a2"],
            "B": ["shared", "b1", "b2"],
        },
        limit=5,
        min_per_category=1,
    )

    assert selected.count("shared") == 1
    assert list(assignments).count("shared") == 1


def test_full_population_counts_control_proportional_allocation():
    selected, assignments, report = _topic_coverage_sample(
        {
            "popular": [f"p{number}" for number in range(10)],
            "scarce": [f"s{number}" for number in range(10)],
        },
        limit=10,
        min_per_category=1,
        population_by_category={"popular": 900, "scarce": 100},
    )
    counts = Counter(assignments.values())

    assert len(selected) == 10
    assert counts["popular"] > counts["scarce"]
    assert report["annual_pmc_hit_count_by_category"]["popular"] == 900
