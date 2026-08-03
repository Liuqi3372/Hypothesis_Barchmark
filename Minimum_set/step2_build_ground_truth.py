from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from tqdm import tqdm

from mrrs.ground_truth import GroundTruthReviewer, build_ground_truth
from mrrs.package import write_json
from step1_prepare_data import parse_limit_interval, select_interval


def load_key(project: Path) -> str:
    value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    for path in (project.parent / "key" / "DEEPSEEK_API_KEY.txt",):
        if path.is_file():
            return path.read_text(encoding="utf-8-sig").strip()
    return ""


def read_package_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Step 2: build minimum-Fact ground truth")
    parser.add_argument("--input", type=Path, default=project / "data" / "step1")
    parser.add_argument(
        "--source-csv", type=Path,
        help="Package CSV; default: <input>/dataset_manifest.csv",
    )
    parser.add_argument("--output", type=Path, default=project / "data" / "step2")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help="Optional 1-based inclusive package interval; omitted means all CSV rows",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    key = load_key(project)
    if not key:
        parser.error("Missing DEEPSEEK_API_KEY or key file")
    source_csv = args.source_csv or args.input / "dataset_manifest.csv"
    if source_csv.exists():
        packages = read_package_csv(source_csv)
    else:
        dataset = json.loads((args.input / "dataset_manifest.json").read_text(encoding="utf-8"))
        packages = dataset["packages"]
    try:
        packages = select_interval(packages, args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    reviewer = GroundTruthReviewer(key, args.model)
    summaries = []
    processed_bar = tqdm(total=len(packages), desc="Minimum Step 2 processed", unit="paper", position=0)
    completed_bar = tqdm(desc="Minimum Step 2 complete", unit="paper", position=1)
    for index, item in enumerate(packages, start=1):
        package_dir = args.input / item["package_id"]
        processed_bar.set_postfix_str(item["package_id"])
        try:
            result = build_ground_truth(package_dir, reviewer, args.output, args.force)
            output_path = args.output / f"{result['target_paper']['pmcid']}_ground_truth.json"
            write_json(output_path, result)
            summaries.append({"package_id": item["package_id"], "status": "COMPLETE", "output": str(output_path)})
            completed_bar.update(1)
            tqdm.write(f"COMPLETE {item['package_id']}  {output_path.name}")
        except Exception as exc:
            error_path = args.output / f"{item['package_id']}_error.json"
            write_json(error_path, {"package_id": item["package_id"], "error_type": type(exc).__name__, "error": str(exc)})
            summaries.append({"package_id": item["package_id"], "status": "ERROR", "output": str(error_path)})
            tqdm.write(f"ERROR    {item['package_id']}  {type(exc).__name__}: {exc}")
        finally:
            processed_bar.update(1)
    processed_bar.close()
    completed_bar.close()
    write_json(args.output / "run_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
