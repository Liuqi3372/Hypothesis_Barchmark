from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_limit_interval(value: str) -> tuple[int, int]:
    text = str(value).strip()
    try:
        if "-" in text:
            left, right = text.split("-", 1)
            start, end = int(left), int(right)
        else:
            start, end = 1, int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use N or START-END, for example 2 or 1-2") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("Interval must satisfy 1 <= START <= END")
    return start, end


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_interval(rows: list[dict], interval: tuple[int, int] | None) -> list[dict]:
    if interval is None:
        return rows
    start, end = interval
    if end > len(rows):
        raise ValueError(f"Interval {start}-{end} exceeds CSV row count ({len(rows)})")
    return rows[start - 1:end]


def write_csv_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
