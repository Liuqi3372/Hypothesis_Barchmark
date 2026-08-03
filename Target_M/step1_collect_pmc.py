from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from pmc_m.cli import parse_year_range
from pmc_m.config import DEFAULT_NCBI_KEY_FILE, get_ncbi_email, load_ncbi_api_key
from pmc_m.pmc import PMCClient, collect_range
from pmc_m.rules import CATEGORIES, choose_category, hard_exclusion

FIELDS = [
    "pmcid", "pmid", "doi", "year", "title", "abstract", "journal", "category",
    "article_types", "source_categories", "open_access", "license", "license_url",
    "introduction", "methods", "results", "discussion", "conclusions", "full_text",
    "rule_decision", "exclusion_code",
]


def record(paper, category: str, decision: str, code: str = "") -> dict:
    return {
        "pmcid": paper.pmcid, "pmid": paper.pmid, "doi": paper.doi, "year": paper.year,
        "title": paper.title, "abstract": paper.abstract, "journal": paper.journal,
        "category": category, "article_types": paper.article_types,
        "source_categories": sorted(paper.source_categories), "open_access": paper.open_access,
        "license": paper.license, "license_url": paper.license_url,
        "introduction": paper.introduction, "methods": paper.methods,
        "results": paper.results, "discussion": paper.discussion,
        "conclusions": paper.conclusions, "full_text": paper.full_text,
        "rule_decision": decision, "exclusion_code": code,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = row.copy()
            item["article_types"] = "|".join(item["article_types"])
            item["source_categories"] = "|".join(item["source_categories"])
            writer.writerow(item)


def allocate_year_quotas(start_year: int, end_year: int, count: int) -> dict[int, int]:
    """Split the final eligible-paper target evenly; quotas differ by at most one."""
    years = list(range(start_year, end_year + 1))
    base, remainder = divmod(count, len(years))
    return {year: base + int(index < remainder) for index, year in enumerate(years)}


def screen_paper(paper) -> tuple[dict, bool]:
    category = paper.primary_category or choose_category(
        paper.title, paper.abstract, paper.source_categories
    )
    code = hard_exclusion(
        paper.title, paper.abstract, paper.article_types,
        paper.open_access, paper.journal,
    )
    if not code and len(paper.full_text.strip()) < 500:
        code = "E_MISSING_OR_INSUFFICIENT_FULL_TEXT"
    return (
        record(paper, category, "EXCLUDE" if code else "ELIGIBLE_FOR_LLM", code or ""),
        not bool(code),
    )


def collect_eligible_year(
    client: PMCClient,
    year: int,
    quota: int,
    collect_fn=collect_range,
) -> tuple[list[dict], list[dict], dict]:
    """Increase the candidate pool until one year's final eligible quota is full."""
    if quota == 0:
        return [], [], {"year": year, "eligible_quota": 0, "candidate_rounds": []}

    candidate_count = max(quota + 25, math.ceil(quota * 1.4))
    previous_fetched = -1
    rounds: list[dict] = []
    while True:
        per_round_report: dict = {}
        min_per_category = min(
            200, max(1, candidate_count // (2 * len(CATEGORIES)))
        )
        papers = collect_fn(
            client, year, year, candidate_count,
            min_per_category=min_per_category,
            sampling_report=per_round_report,
        )
        screened = [screen_paper(paper) for paper in papers]
        eligible = [row for row, passed in screened if passed]
        excluded = [row for row, passed in screened if not passed]
        rounds.append({
            "requested_candidates": candidate_count,
            "fetched_candidates": len(papers),
            "eligible_candidates": len(eligible),
            "excluded_candidates": len(excluded),
        })
        print(
            f"  {year}: {len(eligible)}/{quota} eligible from "
            f"{len(papers)} candidates",
            flush=True,
        )
        if len(eligible) >= quota:
            return eligible[:quota], excluded, {
                "year": year,
                "eligible_quota": quota,
                "candidate_rounds": rounds,
                "final_topic_sampling": per_round_report,
            }
        if len(papers) <= previous_fetched or len(papers) < candidate_count:
            raise RuntimeError(
                f"Cannot fill the {year} quota: requested {quota} eligible papers, "
                f"but only {len(eligible)} passed all Step 1 rules from the available candidates."
            )
        previous_fetched = len(papers)
        candidate_count = math.ceil(candidate_count * 1.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: collect and rule-screen one PMC OA sample"
    )
    parser.add_argument(
        "--year-range", type=parse_year_range, required=True, metavar="START-END",
        help="Inclusive publication-year range, for example 2020-2026",
    )
    parser.add_argument(
        "--count", type=int, required=True,
        help="Exact final eligible-paper count, divided evenly across the year range",
    )
    parser.add_argument("--output", type=Path, default=Path("../data"))
    parser.add_argument("--email", default=get_ncbi_email())
    parser.add_argument(
        "--ncbi-api-key", default=None,
        help=f"Optional key override; default file: {DEFAULT_NCBI_KEY_FILE}",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be a positive integer")

    start_year, end_year = args.year_range
    quotas = allocate_year_quotas(start_year, end_year, args.count)
    output_dir = args.output / "step1"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.ncbi_api_key if args.ncbi_api_key is not None else load_ncbi_api_key()
    client = PMCClient(args.email, api_key)
    try:
        print(
            f"[Step 1 {start_year}-{end_year}] Final eligible target: {args.count}; "
            f"year quotas: {quotas}",
            flush=True,
        )
        eligible, excluded = [], []
        sampling_report: dict[str, dict] = {}
        for year, quota in quotas.items():
            year_eligible, year_excluded, year_report = collect_eligible_year(
                client, year, quota
            )
            eligible.extend(year_eligible)
            excluded.extend(year_excluded)
            sampling_report[str(year)] = year_report

        if len(eligible) != args.count:
            raise RuntimeError(
                f"Internal count error: expected {args.count} eligible papers, got {len(eligible)}"
            )

        write_jsonl(output_dir / "eligible.jsonl", eligible)
        write_csv(output_dir / "eligible.csv", eligible)
        write_jsonl(output_dir / "excluded.jsonl", excluded)
        write_csv(output_dir / "excluded.csv", excluded)
        summary = {
            "year_range": {"start": start_year, "end": end_year},
            "requested_count": args.count,
            "year_quotas": quotas,
            "candidate_papers_screened": sum(
                report["candidate_rounds"][-1]["fetched_candidates"]
                for report in sampling_report.values()
                if report["candidate_rounds"]
            ),
            "eligible_for_llm": len(eligible),
            "rule_excluded": len(excluded),
            "eligible_by_year": dict(Counter(row["year"] for row in eligible)),
            "eligible_by_category": dict(Counter(row["category"] for row in eligible)),
            "open_access_filter": True,
            "excluded_statuses": ["preprint", "retracted_or_withdrawn", "on_hold"],
            "sampling_by_year": sampling_report,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
