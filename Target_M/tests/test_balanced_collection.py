from types import SimpleNamespace

from step1_collect_pmc import allocate_year_quotas, collect_eligible_year


def make_paper(index: int, year: int, excluded: bool = False):
    return SimpleNamespace(
        pmcid=f"PMC{year}{index}", pmid=str(index), doi=f"10.test/{year}.{index}",
        year=year, title="Review of cells" if excluded else f"Original cell study {index}",
        abstract="Experimental methods and original cellular results. " * 4,
        journal="Test Journal", article_types=["research-article"],
        source_categories={"Signal transduction and cell communication"},
        open_access=True, license="CC BY", license_url="", introduction="",
        methods="methods " * 80, results="results " * 80, discussion="",
        conclusions="", full_text="full experimental text " * 100,
        primary_category="Signal transduction and cell communication",
    )


def test_year_quotas_are_even_and_sum_to_target():
    quotas = allocate_year_quotas(2024, 2026, 3001)
    assert quotas == {2024: 1001, 2025: 1000, 2026: 1000}
    assert max(quotas.values()) - min(quotas.values()) <= 1
    assert sum(quotas.values()) == 3001


def test_collection_expands_candidates_until_final_eligible_quota_is_full():
    calls = []

    def fake_collect(client, start_year, end_year, count, **kwargs):
        calls.append(count)
        if len(calls) == 1:
            return [make_paper(i, start_year, excluded=(i < count - 2)) for i in range(count)]
        return [make_paper(i, start_year, excluded=(i < count - 5)) for i in range(count)]

    eligible, excluded, report = collect_eligible_year(
        client=None, year=2025, quota=5, collect_fn=fake_collect
    )
    assert len(calls) == 2
    assert len(eligible) == 5
    assert all(row["year"] == 2025 for row in eligible)
    assert len(excluded) == calls[-1] - 5
    assert report["eligible_quota"] == 5
