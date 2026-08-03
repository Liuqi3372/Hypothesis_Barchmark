import argparse

import pytest

from pmc_m.cli import parse_year_range


def test_year_range_is_inclusive_pair():
    assert parse_year_range("2020-2026") == (2020, 2026)
    assert parse_year_range("2025") == (2025, 2025)


def test_reversed_year_range_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_year_range("2026-2020")
