from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from pmc_m.config import get_ncbi_email, load_ncbi_api_key
from pmc_m.pmc import PMCClient


def load_oa_module(project: Path):
    path = project / "step1_collect_pmc.py"
    spec = importlib.util.spec_from_file_location("step1_collect_pmc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualifies(audit: dict, args: argparse.Namespace) -> bool:
    return bool(
        audit.get("intro_reference_count", 0) >= args.min_intro_references
        and audit.get("intro_reference_pmc_count", 0) >= args.min_intro_pmc_references
        and audit.get("intro_reference_resolution_rate", 0) >= args.min_intro_resolution_rate
        and audit.get("intro_reference_oa_coverage_all", 0) >= args.min_intro_oa_coverage
        and audit.get("intro_reference_pmc_coverage_all", 0) >= args.min_intro_pmc_coverage
    )


def main() -> None:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export currently qualified OA-cache papers")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--email", default=get_ncbi_email())
    parser.add_argument("--min-intro-oa-coverage", type=float, default=0.90)
    parser.add_argument("--min-intro-pmc-coverage", type=float, default=0.70)
    parser.add_argument("--min-intro-resolution-rate", type=float, default=0.95)
    parser.add_argument("--min-intro-references", type=int, default=5)
    parser.add_argument("--min-intro-pmc-references", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Export only the first N qualified cached papers")
    args = parser.parse_args()

    step1 = args.data_root / "step1"
    cache_path = step1 / "oa_reference_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    qualified = []
    for key, audit in cache.items():
        if key.startswith("paper:") and qualifies(audit, args):
            qualified.append((key.split(":", 2)[1], audit))
    if not qualified:
        raise RuntimeError("No cached paper currently passes the requested OA thresholds")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        qualified = qualified[:args.limit]

    oa = load_oa_module(project)
    client = PMCClient(args.email, load_ncbi_api_key())
    try:
        ids = [pmcid.removeprefix("PMC") for pmcid, _ in qualified]
        paper_by_id = {paper.pmcid: paper for paper in client.fetch(ids)}
        rows = []
        for pmcid, audit in qualified:
            paper = paper_by_id.get(pmcid)
            if paper is None:
                continue
            paper.open_access = True
            row, passed = oa.screen_paper(paper)
            if not passed:
                continue
            row.update(audit)
            row.update({"rule_decision": "ELIGIBLE_FOR_LLM", "exclusion_code": ""})
            rows.append(row)
    finally:
        client.close()

    oa.write_jsonl(step1 / "eligible.jsonl", rows)
    oa.write_csv(step1 / "eligible.csv", rows)
    summary = {
        "schema_version": "target-m-oa-step1-snapshot-v1",
        "source_cache": str(cache_path.resolve()),
        "eligible_for_llm": len(rows),
        "pmcids": [row["pmcid"] for row in rows],
        "oa_thresholds": {
            "min_intro_oa_coverage": args.min_intro_oa_coverage,
            "min_intro_pmc_coverage": args.min_intro_pmc_coverage,
            "min_intro_resolution_rate": args.min_intro_resolution_rate,
            "min_intro_references": args.min_intro_references,
            "min_intro_pmc_references": args.min_intro_pmc_references,
        },
    }
    (step1 / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(rows)} qualified papers")
    print(f"CSV: {step1 / 'eligible.csv'}")
    print(f"JSONL: {step1 / 'eligible.jsonl'}")


if __name__ == "__main__":
    main()
