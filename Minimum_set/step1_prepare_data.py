from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from mrrs.ncbi import NCBIClient
from mrrs.package import build_package, write_json


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_limit_interval(value: str) -> tuple[int, int]:
    text = str(value).strip()
    try:
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start, end = 1, int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use N or START-END, for example 5 or 1-5") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("The interval must satisfy 1 <= START <= END")
    return start, end


def select_interval(rows: list[dict], interval: tuple[int, int] | None) -> list[dict]:
    if interval is None:
        return rows
    start, end = interval
    if end > len(rows):
        raise ValueError(f"Interval {start}-{end} exceeds CSV row count ({len(rows)})")
    return rows[start - 1:end]


def load_key(name: str, paths: list[Path]) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8-sig").strip()
    return ""


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Step 1: build local Minimum-set data packages")
    parser.add_argument("--input", type=Path, help="Legacy Target_M Step3 final.jsonl")
    parser.add_argument("--source-csv", type=Path, help="Target_M Step3 final.csv")
    parser.add_argument("--output", type=Path, default=project / "data" / "step1")
    parser.add_argument("--email", default=os.getenv("NCBI_EMAIL", "lllmeqi77@gmail.com"))
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help="Optional 1-based inclusive CSV interval; omitted means all rows",
    )
    args = parser.parse_args()
    if args.source_csv and args.input:
        parser.error("Use either --source-csv or legacy --input, not both")
    source = args.source_csv or args.input
    if source is None:
        parser.error("--source-csv is required (or use legacy --input JSONL)")
    rows = read_csv_rows(source) if args.source_csv else read_jsonl(source)
    try:
        rows = select_interval(rows, args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    dataset_root = source.parent.parent
    args.output.mkdir(parents=True, exist_ok=True)
    ncbi_key = load_key("NCBI_API_KEY", [project.parent / "key" / "NCBI_API_KEY.txt"])
    client = NCBIClient(args.email, ncbi_key)
    manifests = []
    try:
        progress = tqdm(rows, desc="Minimum Step 1 packages", unit="paper")
        for index, row in enumerate(progress, start=1):
            pmcid = str(row["pmcid"]).upper()
            progress.set_postfix_str(pmcid)
            audit_path = dataset_root / row["feasibility_audit_path"]
            audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
            source_position = (args.limit[0] - 1 + index) if args.limit else index
            package_dir = args.output / f"m{source_position}_{pmcid}"
            print(f"[Step 1 {index}/{len(rows)}] {pmcid}", flush=True)
            manifests.append(build_package(row, audit, client, package_dir))
    finally:
        client.close()
    write_json(args.output / "dataset_manifest.json", {
        "schema_version": "minimum-set-dataset-1.0",
        "source": str(source.resolve()), "paper_count": len(manifests),
        "packages": [{"package_id": item["package_id"], "manifest": f"{item['package_id']}/manifest.json"} for item in manifests],
    })
    with (args.output / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["package_id", "manifest", "target_pmcid"])
        writer.writeheader()
        for item in manifests:
            writer.writerow({
                "package_id": item["package_id"],
                "manifest": f"{item['package_id']}/manifest.json",
                "target_pmcid": item["target_pmcid"],
            })
    print(f"Saved {len(manifests)} packages to {args.output}", flush=True)


if __name__ == "__main__":
    main()
