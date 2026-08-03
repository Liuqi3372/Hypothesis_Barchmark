from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from pmc_m.config import DEFAULT_DEEPSEEK_KEY_FILE, load_deepseek_api_key
from pmc_m.llm import BinaryReviewer
from pmc_m.rules import CATEGORIES

FIELDS = [
    "pmcid", "pmid", "doi", "year", "title", "abstract", "journal", "category",
    "article_types", "source_categories", "open_access", "license", "license_url",
    "introduction", "methods", "results", "discussion", "conclusions", "full_text",
    "research_question_result", "research_question_reason",
    "original_experimental_research_result", "original_experimental_research_reason",
    "novel_biological_finding_result", "novel_biological_finding_reason",
    "experimental_evidence_result", "experimental_evidence_reason",
    "decision", "exclusion_code", "model", "review_json",
]
REVIEW_MODE = "full_text_four_question_en_v4"


def parse_index_range(value: str) -> tuple[int, int]:
    """Parse a one-based inclusive paper interval such as 2-10."""
    try:
        start_text, end_text = value.strip().split("-", 1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Use START-END, for example 2-10") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(
            "The paper range is one-based and must satisfy 1 <= START <= END"
        )
    return start, end


def parse_limit_interval(value: str) -> tuple[int, int]:
    """Accept N as positions 1..N, or START-END as an inclusive interval."""
    text = str(value).strip()
    if "-" in text:
        return parse_index_range(text)
    try:
        end = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use a positive integer or START-END, for example 17 or 1-17"
        ) from exc
    if end < 1:
        raise argparse.ArgumentTypeError("--limit must be positive")
    return 1, end


def select_index_range(rows: list[dict], interval: tuple[int, int]) -> list[dict]:
    """Select a validated one-based inclusive interval from Step 1 order."""
    start, end = interval
    if end > len(rows):
        raise ValueError(
            f"Paper range {start}-{end} exceeds the Step 1 candidate count ({len(rows)})"
        )
    return rows[start - 1:end]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 1 output: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict]:
    """Read full-text CSV rows without Python's small default field limit."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("article_types", "source_categories"):
            value = row.get(field, "")
            if isinstance(value, str):
                row[field] = [item for item in value.split("|") if item]
    return rows


def read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paper_key(row: dict) -> str:
    """Return a stable key, including compatibility fallbacks."""
    for field in ("pmcid", "doi", "pmid", "title"):
        value = str(row.get(field, "")).strip()
        if value:
            return f"{field}:{value}"
    raise ValueError("The paper has no PMCID, DOI, PMID, or title for a stable cache key")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = row.copy()
            item["article_types"] = "|".join(item.get("article_types", []))
            item["source_categories"] = "|".join(item.get("source_categories", []))
            writer.writerow(item)


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        x.get("paper_key") or x.get("pmcid"): x
        for x in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: full-text four-question expert review in JSON")
    parser.add_argument("--input", type=Path, default=Path("../data"))
    parser.add_argument(
        "--source-csv", type=Path,
        help="Optional Step 1 CSV to read directly; default: <input>/step1/eligible.jsonl",
    )
    parser.add_argument(
        "--provider", choices=("deepseek", "openai"), default="deepseek",
        help="Model API provider; default: deepseek",
    )
    parser.add_argument("--model", default=None, help="Default DeepSeek model: deepseek-v4-pro")
    parser.add_argument("--force", action="store_true", help="Clear the cache and review every paper again")
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help=(
            "One-based inclusive scan interval: 17 scans papers 1-17; "
            "5-12 scans papers 5-12"
        ),
    )
    parser.add_argument(
        "--range", dest="index_range", type=parse_index_range, metavar="START-END",
        help="One-based inclusive Step 1 paper interval; for example 2-10 reviews 9 papers",
    )
    args = parser.parse_args()
    if args.limit is not None and args.index_range is not None:
        parser.error("--limit and --range cannot be used together")

    if args.model is None:
        args.model = (
            os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
            if args.provider == "deepseek"
            else os.getenv("OPENAI_MODEL", "gpt-5.6")
        )
    api_key = load_deepseek_api_key() if args.provider == "deepseek" else None
    if args.provider == "deepseek" and not api_key:
        parser.error(
            "Missing DeepSeek API key. Set DEEPSEEK_API_KEY or save the key in "
            f"{DEFAULT_DEEPSEEK_KEY_FILE}"
        )
    reviewer = BinaryReviewer(
        model=args.model, api_key=api_key, provider=args.provider
    )
    summaries = []
    for folder in [args.input]:
        input_dir = folder / "step1"
        output_dir = folder / "step2"
        output_dir.mkdir(parents=True, exist_ok=True)
        source = args.source_csv or input_dir / "eligible.jsonl"
        candidates = read_csv_rows(source) if args.source_csv else read_jsonl(source)
        legacy = [
            row.get("pmcid", "")
            for row in candidates
            if not row.get("full_text") and not any(
                row.get(name) for name in ("methods", "results", "discussion", "conclusions")
            )
        ]
        if legacy:
            raise RuntimeError(
                f"Step 1 data has no full-text evidence (for example, {legacy[0]}). "
                "Run the current step1_collect_pmc.py first."
            )
        total_candidates = len(candidates)
        selected_interval = None
        requested_interval = args.index_range or args.limit
        if requested_interval is not None:
            try:
                candidates = select_index_range(candidates, requested_interval)
            except ValueError as exc:
                parser.error(str(exc))
            selected_interval = {
                "start": requested_interval[0], "end": requested_interval[1],
                "inclusive": True,
            }
        state_path = output_dir / "state.jsonl"
        if args.force and state_path.exists():
            state_path.unlink()
        state = load_state(state_path)
        retained, excluded = [], []
        range_text = (
            f" (Step 1 positions {requested_interval[0]}-{requested_interval[1]}, inclusive)"
            if requested_interval else ""
        )
        print(f"[Step 2] Reviewing {len(candidates)} papers{range_text}", flush=True)
        reviewed_bar = tqdm(total=len(candidates), desc="Step 2 reviewed", unit="paper", position=0)
        included_bar = tqdm(desc="Step 2 included", unit="paper", position=1)
        for index, row in enumerate(candidates, 1):
            reviewed_bar.set_postfix_str(str(row.get("pmcid", "")))
            stable_key = paper_key(row)
            cached = state.get(stable_key)
            if (
                cached
                and cached.get("model") == args.model
                and cached.get("provider") == args.provider
                and cached.get("review_mode") == REVIEW_MODE
                and isinstance(cached.get("review"), dict)
            ):
                review = reviewer._validate_review(cached["review"])
            else:
                for attempt in range(3):
                    try:
                        review = reviewer.classify(
                            title=row["title"], abstract=row["abstract"], category=row["category"],
                            journal=row.get("journal", ""),
                            article_types="|".join(row.get("article_types", [])),
                            introduction=row.get("introduction", ""),
                            methods=row.get("methods", ""), results=row.get("results", ""),
                            discussion=row.get("discussion", ""),
                            conclusions=row.get("conclusions", ""),
                            full_text=row.get("full_text", ""),
                        )
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
                state_item = {
                    "paper_key": stable_key, "pmcid": row.get("pmcid", ""),
                    "decision": review["final_decision"], "review": review,
                    "model": args.model, "provider": args.provider,
                    "review_mode": REVIEW_MODE,
                }
                with state_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(state_item, ensure_ascii=False) + "\n")
            decision = review["final_decision"]
            final = {
                **row, "decision": decision,
                "model": f"{args.provider}:{args.model}",
                "research_question_result": review["research_question"]["result"],
                "research_question_reason": review["research_question"]["reason"],
                "original_experimental_research_result": review["original_experimental_research"]["result"],
                "original_experimental_research_reason": review["original_experimental_research"]["reason"],
                "novel_biological_finding_result": review["novel_biological_finding"]["result"],
                "novel_biological_finding_reason": review["novel_biological_finding"]["reason"],
                "experimental_evidence_result": review["experimental_evidence"]["result"],
                "experimental_evidence_reason": review["experimental_evidence"]["reason"],
                "review_json": json.dumps(review, ensure_ascii=False),
                "exclusion_code": "" if decision == "INCLUDE" else "E_LLM_EXPERT_REJECT",
            }
            (retained if decision == "INCLUDE" else excluded).append(final)
            reviewed_bar.update(1)
            if decision == "INCLUDE":
                included_bar.update(1)
                tqdm.write(f"INCLUDED  {row.get('pmcid', '')}  {row.get('title', '')}")
            else:
                tqdm.write(f"EXCLUDED  {row.get('pmcid', '')}")
        reviewed_bar.close()
        included_bar.close()

        # Preserve earlier interval results. A newly reviewed paper replaces its
        # previous result, while untouched papers remain in the cumulative files.
        previous_rows = (
            read_jsonl_if_exists(output_dir / "scientifically_eligible.jsonl")
            + read_jsonl_if_exists(output_dir / "excluded.jsonl")
        )
        cumulative = {paper_key(row): row for row in previous_rows}
        for row in retained + excluded:
            cumulative[paper_key(row)] = row
        retained_all = [row for row in cumulative.values() if row.get("decision") == "INCLUDE"]
        excluded_all = [row for row in cumulative.values() if row.get("decision") != "INCLUDE"]
        write_jsonl(output_dir / "scientifically_eligible.jsonl", retained_all)
        write_csv(output_dir / "scientifically_eligible.csv", retained_all)
        write_jsonl(output_dir / "excluded.jsonl", excluded_all)
        write_csv(output_dir / "excluded.csv", excluded_all)
        category_folder = output_dir / "categories"
        category_folder.mkdir(exist_ok=True)
        for category in CATEGORIES:
            rows = [x for x in retained_all if x["category"] == category.name]
            write_csv(category_folder / f"{category.name}.csv", rows)
        summary = {
            "source": str(source.resolve()),
            "step1_total_candidates": total_candidates,
            "selected_interval": selected_interval,
            "screened_this_run": len(candidates),
            "retained_this_run": len(retained),
            "llm_excluded_this_run": len(excluded),
            "cumulative_screened": len(cumulative),
            "cumulative_retained": len(retained_all),
            "cumulative_llm_excluded": len(excluded_all),
            "cumulative_retained_by_category": dict(
                Counter(x["category"] for x in retained_all)
            ),
            "model": args.model,
            "provider": args.provider,
            "review_mode": REVIEW_MODE,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (args.input / "step2" / "run_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
