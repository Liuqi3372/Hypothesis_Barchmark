import argparse

import pytest

from step2_llm_screen import parse_index_range, parse_limit_interval, read_csv_rows, select_index_range


def test_range_is_one_based_and_inclusive():
    rows = [{"id": number} for number in range(1, 12)]
    assert parse_index_range("2-10") == (2, 10)
    assert [row["id"] for row in select_index_range(rows, (2, 10))] == list(range(2, 11))


@pytest.mark.parametrize("value", ["0-2", "10-2", "2", "x-y"])
def test_invalid_range_is_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_index_range(value)


def test_range_cannot_exceed_candidate_count():
    with pytest.raises(ValueError, match="exceeds"):
        select_index_range([{}, {}], (2, 3))


def test_limit_accepts_prefix_or_explicit_interval():
    assert parse_limit_interval("17") == (1, 17)
    assert parse_limit_interval("1-17") == (1, 17)
    assert parse_limit_interval("5-12") == (5, 12)


def test_source_csv_restores_pipe_separated_list_fields(tmp_path):
    path = tmp_path / "eligible.csv"
    path.write_text(
        "pmcid,article_types,source_categories,full_text\n"
        'PMC1,"research-article|article","A|B","long text"\n',
        encoding="utf-8-sig",
    )
    row = read_csv_rows(path)[0]
    assert row["article_types"] == ["research-article", "article"]
    assert row["source_categories"] == ["A", "B"]
