from __future__ import annotations

import argparse


def parse_year_range(value: str) -> tuple[int, int]:
    """Parse one inclusive publication-year range, for example 2020-2026."""
    try:
        parts = value.strip().split("-", 1)
        start = int(parts[0])
        end = int(parts[1]) if len(parts) == 2 else start
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use YEAR-YEAR, for example 2020-2026") from exc
    if start < 1900 or end > 2100 or start > end:
        raise argparse.ArgumentTypeError(
            "The year range must be ascending and between 1900 and 2100"
        )
    return start, end
