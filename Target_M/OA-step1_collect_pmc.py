from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import httpx
from tqdm import tqdm

from pmc_m.cli import parse_year_range
from pmc_m.config import DEFAULT_NCBI_KEY_FILE, get_ncbi_email, load_ncbi_api_key
from pmc_m.pmc import PMCClient, collect_range, _text
from step1_collect_pmc import allocate_year_quotas, record, screen_paper


OA_FIELDS = [
    "pmcid", "pmid", "doi", "year", "title", "abstract", "journal", "category",
    "article_types", "source_categories", "open_access", "license", "license_url",
    "introduction", "methods", "results", "discussion", "conclusions", "full_text",
    "rule_decision", "exclusion_code",
    "intro_reference_count", "intro_reference_resolvable_count",
    "intro_reference_oa_count", "intro_reference_pmc_count",
    "intro_reference_unresolved_count", "intro_reference_resolution_rate",
    "intro_reference_oa_coverage_all", "intro_reference_pmc_coverage_all",
    "intro_reference_oa_audit",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = row.copy()
            item["article_types"] = "|".join(item.get("article_types", []))
            item["source_categories"] = "|".join(item.get("source_categories", []))
            item["intro_reference_oa_audit"] = json.dumps(
                item.get("intro_reference_oa_audit", []), ensure_ascii=False
            )
            writer.writerow(item)


class OAReferenceAuditor:
    """Audit unique Introduction references with persistent PMID/DOI OA caching."""

    def __init__(self, client: PMCClient, email: str, cache_path: Path):
        self.client = client
        self.email = email
        self.cache_path = cache_path
        self.cache = (
            json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if cache_path.exists() else {}
        )

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.cache_path)

    @staticmethod
    def _introduction(article: ET.Element) -> ET.Element | None:
        body = article.find("./body")
        if body is None:
            return None
        for section in body.findall(".//sec"):
            label = (
                f"{section.attrib.get('sec-type', '')} {_text(section.find('./title'))}"
            ).lower()
            if "intro" in label or "background" in label:
                return section
        sections = body.findall("./sec")
        return sections[0] if sections else body

    def extract_unique_references(self, root: ET.Element) -> list[dict]:
        article = root if root.tag == "article" else root.find(".//article")
        if article is None:
            return []
        introduction = self._introduction(article)
        if introduction is None:
            return []
        cited_ids = []
        for xref in introduction.findall(".//xref[@ref-type='bibr']"):
            cited_ids.extend(xref.attrib.get("rid", "").split())
        reference_nodes = {
            node.attrib.get("id", ""): node
            for node in article.findall(".//back//ref-list//ref")
            if node.attrib.get("id")
        }
        references = []
        seen = set()
        for ref_id in cited_ids:
            node = reference_nodes.get(ref_id)
            if node is None:
                continue
            citation = node.find("./element-citation")
            if citation is None:
                citation = node.find("./mixed-citation")
            if citation is None:
                citation = node
            identifiers = {
                item.attrib.get("pub-id-type", "").lower(): _text(item)
                for item in citation.findall(".//pub-id")
            }
            pmcid = identifiers.get("pmcid", "") or identifiers.get("pmc", "")
            if pmcid and not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            pmid = identifiers.get("pmid", "")
            doi = identifiers.get("doi", "").lower().removeprefix("https://doi.org/")
            citation_text = html.unescape(_text(citation))
            identity = pmcid or pmid or doi or hashlib.sha256(
                citation_text.lower().encode("utf-8")
            ).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            references.append({
                "reference_id": ref_id,
                "title": html.unescape(_text(citation.find(".//article-title"))),
                "pmcid": pmcid.upper(), "pmid": pmid, "doi": doi,
                "citation_text": citation_text,
            })
        return references

    def resolve_pmcid(self, pmid: str, doi: str) -> tuple[str, str]:
        key = f"resolve:{pmid or doi}"
        if key in self.cache:
            item = self.cache[key]
            return item.get("pmcid", ""), item.get("pmid", pmid)
        resolved_pmid = pmid
        pmcid = ""
        try:
            if not resolved_pmid and doi:
                payload = self.client._request("esearch.fcgi", {
                    "db": "pubmed", "term": f'"{doi}"[AID]',
                    "retmode": "json", "retmax": 1,
                }).json().get("esearchresult", {})
                ids = payload.get("idlist", [])
                resolved_pmid = ids[0] if ids else ""
            if resolved_pmid:
                xml = ET.fromstring(self.client._request("elink.fcgi", {
                    "dbfrom": "pubmed", "db": "pmc", "id": resolved_pmid,
                    "retmode": "xml", "linkname": "pubmed_pmc",
                }).content)
                value = _text(xml.find(".//LinkSetDb/Link/Id"))
                pmcid = f"PMC{value}" if value else ""
        except (httpx.HTTPError, ET.ParseError, ValueError):
            pass
        self.cache[key] = {"pmcid": pmcid, "pmid": resolved_pmid}
        return pmcid, resolved_pmid

    def unpaywall_is_oa(self, doi: str) -> tuple[bool, str]:
        key = f"unpaywall:{doi}"
        if key in self.cache:
            item = self.cache[key]
            return bool(item.get("is_oa")), item.get("oa_url", "")
        is_oa, oa_url = False, ""
        if doi:
            try:
                response = self.client.client.get(
                    f"https://api.unpaywall.org/v2/{doi}",
                    params={"email": self.email},
                )
                if response.status_code == 200:
                    payload = response.json()
                    best = payload.get("best_oa_location") or {}
                    is_oa = payload.get("is_oa") is True
                    oa_url = best.get("url_for_pdf") or best.get("url") or ""
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.05)
        self.cache[key] = {"is_oa": is_oa, "oa_url": oa_url}
        return is_oa, oa_url

    def audit_paper(self, pmcid: str) -> dict:
        paper_key = f"paper:{pmcid}:oa-v1"
        if paper_key in self.cache:
            return self.cache[paper_key]
        root = ET.fromstring(self.client._request("efetch.fcgi", {
            "db": "pmc", "id": pmcid.upper().removeprefix("PMC"), "retmode": "xml",
        }).content)
        references = self.extract_unique_references(root)
        audited = []
        for reference in references:
            resolved_pmcid = reference["pmcid"]
            resolved_pmid = reference["pmid"]
            if not resolved_pmcid and (resolved_pmid or reference["doi"]):
                resolved_pmcid, resolved_pmid = self.resolve_pmcid(
                    resolved_pmid, reference["doi"]
                )
            oa_url = ""
            legally_open = bool(resolved_pmcid)
            oa_source = "PMC" if resolved_pmcid else ""
            if not legally_open and reference["doi"]:
                legally_open, oa_url = self.unpaywall_is_oa(reference["doi"])
                oa_source = "UNPAYWALL" if legally_open else ""
            resolvable = bool(reference["pmcid"] or resolved_pmid or reference["doi"])
            audited.append({
                **reference,
                "resolved_pmid": resolved_pmid,
                "resolved_pmcid": resolved_pmcid,
                "resolvable": resolvable,
                "legally_open_fulltext": legally_open,
                "oa_source": oa_source,
                "oa_url": oa_url,
            })
        total = len(audited)
        resolvable_count = sum(item["resolvable"] for item in audited)
        oa_count = sum(item["legally_open_fulltext"] for item in audited)
        pmc_count = sum(bool(item["resolved_pmcid"]) for item in audited)
        result = {
            "intro_reference_count": total,
            "intro_reference_resolvable_count": resolvable_count,
            "intro_reference_oa_count": oa_count,
            "intro_reference_pmc_count": pmc_count,
            "intro_reference_unresolved_count": total - resolvable_count,
            "intro_reference_resolution_rate": resolvable_count / total if total else 0.0,
            "intro_reference_oa_coverage_all": oa_count / total if total else 0.0,
            "intro_reference_pmc_coverage_all": pmc_count / total if total else 0.0,
            "intro_reference_oa_audit": audited,
        }
        self.cache[paper_key] = result
        self.save()
        return result


def oa_screen_paper(
    paper,
    auditor: OAReferenceAuditor,
    min_oa_coverage: float,
    min_pmc_coverage: float,
    min_resolution_rate: float,
    min_intro_references: int,
    min_pmc_references: int,
) -> tuple[dict, bool]:
    row, passed = screen_paper(paper)
    if not passed:
        return row, False
    try:
        audit = auditor.audit_paper(paper.pmcid)
    except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
        row.update({
            "rule_decision": "EXCLUDE",
            "exclusion_code": "E_INTRO_REFERENCE_OA_AUDIT_FAILED",
            "intro_reference_oa_audit": [{"error": f"{type(exc).__name__}: {exc}"}],
        })
        return row, False
    row.update(audit)
    code = ""
    if audit["intro_reference_count"] < min_intro_references:
        code = "E_TOO_FEW_INTRO_REFERENCES"
    elif audit["intro_reference_resolution_rate"] < min_resolution_rate:
        code = "E_LOW_INTRO_REFERENCE_RESOLUTION"
    elif audit["intro_reference_oa_coverage_all"] < min_oa_coverage:
        code = "E_LOW_INTRO_REFERENCE_OA_COVERAGE"
    elif audit["intro_reference_pmc_coverage_all"] < min_pmc_coverage:
        code = "E_LOW_INTRO_REFERENCE_PMC_COVERAGE"
    elif audit["intro_reference_pmc_count"] < min_pmc_references:
        code = "E_TOO_FEW_PMC_INTRO_REFERENCES"
    if code:
        row.update({"rule_decision": "EXCLUDE", "exclusion_code": code})
    return row, not bool(code)


def collect_eligible_year_oa(
    client: PMCClient,
    auditor: OAReferenceAuditor,
    year: int,
    quota: int,
    thresholds: dict,
) -> tuple[list[dict], list[dict], dict]:
    if quota == 0:
        return [], [], {"year": year, "eligible_quota": 0, "candidate_rounds": []}
    candidate_count = max(quota + 25, math.ceil(quota * 1.4))
    previous_fetched = -1
    rounds = []
    screened_by_pmcid: dict[str, tuple[dict, bool]] = {}
    scan_bar = tqdm(desc=f"{year} uniquely OA-audited", unit="paper", position=0)
    qualified_bar = tqdm(total=quota, desc=f"{year} OA-qualified", unit="paper", position=1)
    try:
        while True:
            sampling = {}
            scan_bar.set_postfix_str("collecting PMC candidates")
            papers = collect_range(
                client, year, year, candidate_count,
                min_per_category=max(1, candidate_count // 20),
                sampling_report=sampling,
            )
            for paper in papers:
                key = paper.pmcid or paper.doi or paper.title
                scan_bar.set_postfix_str(paper.pmcid or paper.title[:30])
                if key in screened_by_pmcid:
                    continue
                result = oa_screen_paper(paper, auditor, **thresholds)
                screened_by_pmcid[key] = result
                scan_bar.update(1)
                if result[1]:
                    qualified_bar.update(1)
                    tqdm.write(
                        f"OA-QUALIFIED {paper.pmcid}  "
                        f"OA={result[0]['intro_reference_oa_coverage_all']:.1%}  "
                        f"PMC={result[0]['intro_reference_pmc_coverage_all']:.1%}"
                    )
            screened = list(screened_by_pmcid.values())
            eligible = [row for row, passed in screened if passed]
            excluded = [row for row, passed in screened if not passed]
            rounds.append({
                "requested_candidates": candidate_count,
                "fetched_candidates": len(papers),
                "unique_candidates_audited": len(screened_by_pmcid),
                "eligible_candidates": len(eligible),
                "excluded_candidates": len(excluded),
                "exclusion_codes": dict(Counter(row["exclusion_code"] for row in excluded)),
            })
            tqdm.write(
                f"{year}: {len(eligible)}/{quota} OA-qualified from "
                f"{len(screened_by_pmcid)} unique audited candidates"
            )
            if len(eligible) >= quota:
                return eligible[:quota], excluded, {
                    "year": year, "eligible_quota": quota,
                    "candidate_rounds": rounds, "final_topic_sampling": sampling,
                }
            if len(papers) <= previous_fetched or len(papers) < candidate_count:
                raise RuntimeError(
                    f"Cannot fill {year} quota under OA thresholds: {len(eligible)}/{quota}. "
                    "Lower a threshold or expand the candidate year range."
                )
            previous_fetched = len(papers)
            candidate_count = math.ceil(candidate_count * 1.5)
    finally:
        scan_bar.close()
        qualified_bar.close()


def ratio(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("coverage thresholds must be between 0 and 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 with Introduction-reference legal-OA and PMC coverage gates"
    )
    parser.add_argument("--year-range", type=parse_year_range, required=True, metavar="START-END")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("../data"))
    parser.add_argument("--email", default=get_ncbi_email())
    parser.add_argument("--ncbi-api-key", default=None, help=f"Default: {DEFAULT_NCBI_KEY_FILE}")
    parser.add_argument("--min-intro-oa-coverage", type=ratio, default=0.90)
    parser.add_argument("--min-intro-pmc-coverage", type=ratio, default=0.70)
    parser.add_argument("--min-intro-resolution-rate", type=ratio, default=0.95)
    parser.add_argument("--min-intro-references", type=int, default=5)
    parser.add_argument("--min-intro-pmc-references", type=int, default=2)
    args = parser.parse_args()
    if args.count <= 0 or args.min_intro_references <= 0 or args.min_intro_pmc_references < 0:
        parser.error("counts must be positive (minimum PMC references may be zero)")

    start_year, end_year = args.year_range
    quotas = allocate_year_quotas(start_year, end_year, args.count)
    output_dir = args.output / "step1"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = args.ncbi_api_key if args.ncbi_api_key is not None else load_ncbi_api_key()
    client = PMCClient(args.email, api_key)
    auditor = OAReferenceAuditor(client, args.email, output_dir / "oa_reference_cache.json")
    thresholds = {
        "min_oa_coverage": args.min_intro_oa_coverage,
        "min_pmc_coverage": args.min_intro_pmc_coverage,
        "min_resolution_rate": args.min_intro_resolution_rate,
        "min_intro_references": args.min_intro_references,
        "min_pmc_references": args.min_intro_pmc_references,
    }
    try:
        print(f"[OA Step 1 {start_year}-{end_year}] target={args.count}; quotas={quotas}")
        print(json.dumps(thresholds, ensure_ascii=False), flush=True)
        eligible, excluded, reports = [], [], {}
        for year, quota in quotas.items():
            year_ok, year_excluded, report = collect_eligible_year_oa(
                client, auditor, year, quota, thresholds
            )
            eligible.extend(year_ok)
            excluded.extend(year_excluded)
            reports[str(year)] = report
        write_jsonl(output_dir / "eligible.jsonl", eligible)
        write_csv(output_dir / "eligible.csv", eligible)
        write_jsonl(output_dir / "excluded.jsonl", excluded)
        write_csv(output_dir / "excluded.csv", excluded)
        summary = {
            "schema_version": "target-m-oa-step1-v1",
            "year_range": {"start": start_year, "end": end_year},
            "requested_count": args.count,
            "eligible_for_llm": len(eligible),
            "rule_excluded": len(excluded),
            "oa_thresholds": thresholds,
            "eligible_by_year": dict(Counter(row["year"] for row in eligible)),
            "eligible_by_category": dict(Counter(row["category"] for row in eligible)),
            "exclusion_codes": dict(Counter(row["exclusion_code"] for row in excluded)),
            "sampling_by_year": reports,
            "oa_definition": "PMCID or legal OA location reported by Unpaywall; unresolved/failed checks count as non-OA",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        auditor.save()
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
