from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from pmc_m.config import get_ncbi_email, load_deepseek_api_key, load_ncbi_api_key
from pmc_m.feasibility import (
    FACT_PROMPT, EVIDENCE_PROMPT, TRACE_PROMPT,
    DeepSeekJSON, ReferenceRecord, eligible_assessment, evidence_payload,
    extract_article_data, resolve_reference, stable_id, validate_facts,
)
from pmc_m.pmc import PMCClient


STEP3_VERSION = "atomic_qualified_facts_oa_primary_figure_feasibility_v3"


def parse_scan_range(value: str) -> tuple[int, int]:
    """Parse N as papers 1..N, or START-END as a 1-based inclusive range."""
    text = str(value).strip()
    try:
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start, end = 1, int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--limit must be a positive integer or a range such as 100-200"
        ) from exc
    if start <= 0 or end <= 0 or start > end:
        raise argparse.ArgumentTypeError(
            "--limit range must satisfy 1 <= START <= END"
        )
    return start, end


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Step 2 output: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")
    # full_text can be several megabytes; Python's csv default is only 128 KiB.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_durable(path: Path, row: dict) -> None:
    """Persist one completed decision immediately so interruption cannot lose it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["pmcid"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = row.copy()
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    value[key] = json.dumps(item, ensure_ascii=False)
            writer.writerow(value)


def get_target_xml(client: PMCClient, pmcid: str) -> ET.Element:
    response = client._request("efetch.fcgi", {
        "db": "pmc", "id": pmcid.upper().removeprefix("PMC"), "retmode": "xml",
    })
    return ET.fromstring(response.content)


def assess_batches(
    llm: DeepSeekJSON,
    facts: list[dict],
    references: list[ReferenceRecord],
    cache_dir: Path,
    prefix: str,
    force: bool,
    batch_size: int = 3,
) -> list[dict]:
    output = []
    fact_signature = stable_id(json.dumps(facts, ensure_ascii=False, sort_keys=True))
    for offset in range(0, len(references), batch_size):
        batch = references[offset:offset + batch_size]
        batch_key = stable_id(
            f"{STEP3_VERSION}|{fact_signature}|"
            + "|".join(item.reference_id for item in batch)
        )
        print(f"    Evidence batch {offset // batch_size + 1}: {len(batch)} references", flush=True)
        result = llm.call(
            EVIDENCE_PROMPT,
            evidence_payload(facts, batch),
            cache_dir / f"{prefix}_evidence_{batch_key}.json",
            force,
        )
        output.extend(result.get("assessments", []))
    return output


def trace_review(
    client: PMCClient,
    llm: DeepSeekJSON,
    review: ReferenceRecord,
    fact: dict,
    cache_dir: Path,
    force: bool,
) -> tuple[list[ReferenceRecord], list[dict]]:
    if not review.pmcid:
        return [], []
    review_data = extract_article_data(get_target_xml(client, review.pmcid))
    trace_candidates = [
        {
            "reference_id": item.reference_id,
            "citation_text": item.citation_text,
            "citation_contexts": item.citation_contexts,
        }
        for item in review_data["references"].values()
        if item.citation_contexts
    ]
    trace_result = llm.call(
        TRACE_PROMPT,
        {"fact": fact, "review_title": review.title, "review_citations": trace_candidates},
        cache_dir / f"trace_{stable_id(review.reference_id + fact['fact_id'])}.json",
        force,
    )
    selected_ids = [item.get("reference_id") for item in trace_result.get("selected_reference_ids", [])]
    selected = []
    for ref_id in selected_ids[:8]:
        item = review_data["references"].get(ref_id)
        if item is None:
            continue
        item.source_reference_id = review.reference_id
        item.trace_depth = 1
        item.reference_id = f"{review.reference_id}::{ref_id}"
        selected.append(resolve_reference(client, item))
    assessments = assess_batches(
        llm, [fact], selected, cache_dir,
        f"trace_{stable_id(review.reference_id + fact['fact_id'])}", force,
    ) if selected else []
    return selected, assessments


def process_paper(
    row: dict,
    client: PMCClient,
    llm: DeepSeekJSON,
    output_root: Path,
    force: bool,
) -> dict:
    pmcid = str(row.get("pmcid", "")).upper()
    package_dir = output_root / "reasoning_inputs" / pmcid
    cache_dir = output_root / ".step3_cache" / pmcid
    package_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    article = extract_article_data(get_target_xml(client, pmcid))

    fact_result = llm.call(
        FACT_PROMPT,
        {
            "target_paper": {"pmcid": pmcid, "title": article["title"], "abstract": article["abstract"]},
            "introduction": article["introduction"],
            "introduction_paragraphs": article["introduction_paragraphs"],
            "reference_catalog": [
                {
                    "reference_id": ref.reference_id,
                    "citation_text": ref.citation_text,
                    "title": ref.title,
                    "journal": ref.journal,
                    "year": ref.year,
                    "pmid": ref.pmid,
                    "pmcid": ref.pmcid,
                    "doi": ref.doi,
                    "cited_in_introduction": bool(ref.citation_contexts),
                }
                for ref in article["references"].values()
                if ref.citation_contexts
            ],
        },
        cache_dir / "candidate_facts_v3.json",
        force,
    )
    facts = validate_facts(fact_result, article["introduction"], set(article["references"]))

    nearby_ids = list(dict.fromkeys(
        ref_id for fact in facts for ref_id in fact.get("nearby_reference_ids", [])
    ))
    references = []
    for ref_id in nearby_ids:
        reference = article["references"][ref_id]
        references.append(resolve_reference(client, reference))
    initial_assessments = assess_batches(
        llm, facts, references, cache_dir, "introduction", force,
    )

    reference_by_id = {item.reference_id: item for item in references}
    traced_references: list[ReferenceRecord] = []
    traced_assessments: list[dict] = []
    trace_audit = []
    for assessment in initial_assessments:
        if assessment.get("support_level") != "DIRECT" or assessment.get("article_type") != "REVIEW":
            continue
        fact = next((item for item in facts if item["fact_id"] == assessment.get("fact_id")), None)
        review = reference_by_id.get(assessment.get("reference_id"))
        if fact is None or review is None:
            continue
        originals, original_assessments = trace_review(
            client, llm, review, fact, cache_dir, force,
        )
        traced_references.extend(originals)
        traced_assessments.extend(original_assessments)
        trace_audit.append({
            "fact_id": fact["fact_id"],
            "review_reference_id": review.reference_id,
            "candidate_original_reference_ids": [item.reference_id for item in originals],
        })

    all_references = references + traced_references
    all_assessments = initial_assessments + traced_assessments
    all_reference_by_id = {item.reference_id: item for item in all_references}
    coverage = []
    for fact in facts:
        eligible = []
        for item in all_assessments:
            if item.get("fact_id") != fact["fact_id"]:
                continue
            reference = all_reference_by_id.get(item.get("reference_id"))
            if reference and eligible_assessment(item, reference):
                eligible.append(item["reference_id"])
        coverage.append({
            "fact_id": fact["fact_id"],
            "eligible_oa_primary_reference_ids": list(dict.fromkeys(eligible)),
            "covered": bool(eligible),
        })
    passed = all(item["covered"] for item in coverage)
    failed_facts = [item["fact_id"] for item in coverage if not item["covered"]]
    result = {
        "schema_version": STEP3_VERSION,
        "target_paper": {"pmcid": pmcid, "title": article["title"]},
        "candidate_reasoning": fact_result,
        "references": [item.as_dict() for item in all_references],
        "evidence_assessments": all_assessments,
        "review_to_primary_tracing": trace_audit,
        "fact_coverage": coverage,
        "decision": "INCLUDE" if passed else "EXCLUDE",
        "exclusion_code": "" if passed else "E_INCOMPLETE_OA_PRIMARY_FACT_COVERAGE",
        "failed_fact_ids": failed_facts,
        "decision_rule": "Return 2-6 qualified Known Facts; every returned Fact needs DIRECT support from an OA/PMC primary experimental article with a relevant experimental figure caption. Minimal-subset ablation is explicitly deferred to Minimum_set.",
        "figure_screening_limitation": "Figure relevance is caption-based; DeepSeek receives no image pixels.",
    }
    (package_dir / "feasibility_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Fact-level OA primary-evidence feasibility gate")
    parser.add_argument("--input", type=Path, required=True, help="Target_M data directory")
    parser.add_argument(
        "--source-csv", type=Path,
        help="CSV to scan; default: <input>/step1/eligible.csv",
    )
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--target-count", type=int, default=5, help="Stop after finding this many qualified papers")
    parser.add_argument(
        "--limit", type=parse_scan_range, metavar="N|START-END",
        help=(
            "CSV scan interval: N scans papers 1..N; START-END scans that "
            "1-based inclusive interval, for example 100-200"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--email", default=get_ncbi_email())
    args = parser.parse_args()
    if args.target_count <= 0:
        parser.error("--target-count must be positive")
    api_key = load_deepseek_api_key()
    if not api_key:
        parser.error("Missing DEEPSEEK_API_KEY or key file")
    output_dir = args.input / "step3"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.source_csv or args.input / "step1" / "eligible.csv"
    rows = read_csv_rows(source)
    selected_interval = None
    if args.limit:
        start, end = args.limit
        selected_interval = {"start_1_based": start, "end_1_based_inclusive": end}
        rows = rows[start - 1:end]

    previous = []
    for path in (output_dir / "final.jsonl", output_dir / "excluded.jsonl"):
        if path.exists():
            previous.extend(read_jsonl(path))
    cumulative = {
        str(item.get("pmcid") or item.get("doi") or item.get("title")): item
        for item in previous
        if item.get("step3_schema_version") == STEP3_VERSION
    }
    if not args.force:
        processed_by_current_version = {
            key for key, item in cumulative.items()
            if item.get("step3_schema_version") == STEP3_VERSION
        }
        rows = [
            row for row in rows
            if str(row.get("pmcid") or row.get("doi") or row.get("title"))
            not in processed_by_current_version
        ]
    existing_qualified = sum(
        item.get("step3_decision") == "INCLUDE" for item in cumulative.values()
    )

    client = PMCClient(args.email, load_ncbi_api_key())
    llm = DeepSeekJSON(api_key, args.model)
    included, excluded = [], []
    errors = []
    scanned_this_run = 0
    scan_bar = tqdm(total=len(rows), desc="Scanned CSV papers", unit="paper", position=0)
    qualified_bar = tqdm(
        total=args.target_count, initial=min(existing_qualified, args.target_count),
        desc="Qualified papers", unit="paper", position=1,
    )
    try:
        if existing_qualified < args.target_count:
            for row in rows:
                pmcid = str(row.get("pmcid", ""))
                scan_bar.set_postfix_str(pmcid)
                try:
                    audit = process_paper(row, client, llm, output_dir, args.force)
                    final_row = {
                        **row,
                        "step3_schema_version": STEP3_VERSION,
                        "step3_decision": audit["decision"],
                        "step3_exclusion_code": audit["exclusion_code"],
                        "candidate_fact_count": len(audit["candidate_reasoning"]["candidate_facts"]),
                        "covered_candidate_fact_count": sum(item["covered"] for item in audit["fact_coverage"]),
                        "failed_fact_ids": audit["failed_fact_ids"],
                        "feasibility_audit_path": f"step3/reasoning_inputs/{pmcid}/feasibility_audit.json",
                    }
                    if audit["decision"] == "INCLUDE":
                        included.append(final_row)
                        append_jsonl_durable(output_dir / "final.jsonl", final_row)
                        qualified_bar.update(1)
                        tqdm.write(f"QUALIFIED {pmcid}  {row.get('title', '')}")
                    else:
                        excluded.append(final_row)
                        append_jsonl_durable(output_dir / "excluded.jsonl", final_row)
                        tqdm.write(f"EXCLUDED  {pmcid}  failed={','.join(audit['failed_fact_ids'])}")
                except Exception as exc:
                    errors.append({"pmcid": pmcid, "error_type": type(exc).__name__, "error": str(exc)})
                    tqdm.write(f"ERROR     {pmcid}  {type(exc).__name__}: {exc}")
                finally:
                    scan_bar.update(1)
                    scanned_this_run += 1
                if existing_qualified + len(included) >= args.target_count:
                    break
    finally:
        client.close()
        scan_bar.close()
        qualified_bar.close()

    for item in included + excluded:
        cumulative[str(item.get("pmcid") or item.get("doi") or item.get("title"))] = item
    included_all = [item for item in cumulative.values() if item.get("step3_decision") == "INCLUDE"]
    excluded_all = [item for item in cumulative.values() if item.get("step3_decision") != "INCLUDE"]
    write_jsonl(output_dir / "final.jsonl", included_all)
    write_csv(output_dir / "final.csv", included_all)
    write_jsonl(output_dir / "excluded.jsonl", excluded_all)
    write_csv(output_dir / "excluded.csv", excluded_all)
    summary = {
        "schema_version": STEP3_VERSION,
        "input": str(source),
        "selected_csv_interval": selected_interval,
        "available_unprocessed_rows": len(rows),
        "scanned_this_run": scanned_this_run,
        "included_this_run": len(included), "excluded_this_run": len(excluded),
        "cumulative_included": len(included_all), "cumulative_excluded": len(excluded_all),
        "target_count": args.target_count,
        "target_reached": len(included_all) >= args.target_count,
        "errors": errors,
        "model": args.model,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
