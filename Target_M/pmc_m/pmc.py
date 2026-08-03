from __future__ import annotations

import html
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

from .rules import CATEGORIES

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Apply status exclusions during candidate discovery. The deterministic rules
# repeat these checks after EFetch so unusual JATS labels cannot slip through.
SEARCH_EXCLUSIONS = (
    'NOT preprint[pt] NOT articletyperetraction '
    'NOT hasretractionin NOT "retracted publication"[pt]'
)


@dataclass
class Paper:
    pmcid: str
    year: int
    title: str
    abstract: str
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    article_types: list[str] = field(default_factory=list)
    source_categories: set[str] = field(default_factory=set)
    open_access: bool = False
    license: str = ""
    license_url: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusions: str = ""
    full_text: str = ""
    primary_category: str = ""

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["source_categories"] = sorted(self.source_categories)
        return data


class PMCClient:
    def __init__(self, email: str, api_key: str = "", timeout: float = 60.0):
        if not email:
            raise ValueError("NCBI_EMAIL/--email is required by the NCBI E-utilities policy.")
        self.common = {"tool": "pmc_cell_m_builder", "email": email}
        if api_key:
            self.common["api_key"] = api_key
        self.min_interval = 0.11 if api_key else 0.36
        self.last_call = 0.0
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": f"pmc_cell_m_builder/1.0 ({email})"})

    def close(self) -> None:
        self.client.close()

    def _request(self, endpoint: str, params: dict, retries: int = 5) -> httpx.Response:
        merged = {**self.common, **params}
        for attempt in range(retries):
            wait = self.min_interval - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.client.get(f"{BASE}/{endpoint}", params=merged)
                self.last_call = time.monotonic()
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.TimeoutException):
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("unreachable")

    def search(self, year: int, category_query: str, limit: int | None = None) -> list[str]:
        # PMC can cap one ESearch result set at 10k. Monthly segmentation keeps sets bounded.
        if limit is not None:
            term = (
                f"({category_query}) AND open_access[filter] {SEARCH_EXCLUSIONS} "
                f"AND ({year}/01/01[PDAT] : {year}/12/31[PDAT])"
            )
            ids: list[str] = []
            target = min(limit, 10000)
            while len(ids) < target:
                page_size = min(1000, target - len(ids))
                payload = self._request("esearch.fcgi", {
                    "db": "pmc", "term": term, "retmode": "json",
                    "retstart": len(ids), "retmax": page_size, "sort": "relevance",
                }).json()["esearchresult"]
                page = payload.get("idlist", [])
                ids.extend(page)
                if not page or len(ids) >= int(payload.get("count", 0)):
                    break
            return list(dict.fromkeys(ids))

        ids: list[str] = []
        for month in range(1, 13):
            start = f"{year}/{month:02d}/01"
            end_month = month + 1
            end_year = year
            if end_month == 13:
                end_month, end_year = 1, year + 1
            end = f"{end_year}/{end_month:02d}/01"
            term = (
                f"({category_query}) AND open_access[filter] {SEARCH_EXCLUSIONS} "
                f"AND ({start}[PDAT] : {end}[PDAT])"
            )
            retstart = 0
            while True:
                page_size = min(1000, (limit - len(ids)) if limit else 1000)
                if page_size <= 0:
                    return list(dict.fromkeys(ids))
                payload = self._request("esearch.fcgi", {
                    "db": "pmc", "term": term, "retmode": "json",
                    "retstart": retstart, "retmax": page_size, "sort": "pub_date",
                }).json()["esearchresult"]
                page = payload.get("idlist", [])
                ids.extend(page)
                retstart += len(page)
                count = int(payload.get("count", 0))
                if not page or retstart >= count or retstart >= 10000:
                    break
            if limit and len(ids) >= limit:
                break
        return list(dict.fromkeys(ids))

    def search_count(self, year: int, category_query: str) -> int:
        """Return the complete annual hit count without downloading any IDs or articles."""
        term = (
            f"({category_query}) AND open_access[filter] {SEARCH_EXCLUSIONS} "
            f"AND ({year}/01/01[PDAT] : {year}/12/31[PDAT])"
        )
        payload = self._request("esearch.fcgi", {
            "db": "pmc", "term": term, "retmode": "json", "retmax": 0,
        }).json()["esearchresult"]
        return int(payload.get("count", 0))

    def fetch(self, ids: list[str], batch_size: int = 100) -> list[Paper]:
        papers: list[Paper] = []
        for offset in range(0, len(ids), batch_size):
            batch = ids[offset:offset + batch_size]
            xml_text = self._request("efetch.fcgi", {
                "db": "pmc", "id": ",".join(batch), "retmode": "xml",
            }).text
            root = ET.fromstring(xml_text)
            papers.extend(parse_articles(root))
        return papers


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def parse_articles(root: ET.Element) -> list[Paper]:
    papers: list[Paper] = []
    for article in root.findall(".//article"):
        front = article.find("./front")
        if front is None:
            continue
        meta = front.find("./article-meta")
        journal_meta = front.find("./journal-meta")
        if meta is None:
            continue
        ids = {}
        for item in meta.findall("./article-id"):
            ids[item.attrib.get("pub-id-type", "")] = _text(item)
        pmcid = ids.get("pmc", "") or ids.get("pmcid", "")
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        title = html.unescape(_text(meta.find("./title-group/article-title")))
        abstract_parts = [_text(x) for x in meta.findall("./abstract")]
        abstract = html.unescape(" ".join(x for x in abstract_parts if x))
        journal = _text(journal_meta.find("./journal-title-group/journal-title")) if journal_meta is not None else ""
        year_text = _text(meta.find("./pub-date[@pub-type='epub']/year")) or _text(meta.find("./pub-date/year"))
        try:
            year = int(year_text)
        except ValueError:
            year = 0
        types = [article.attrib.get("article-type", "")]
        types += [_text(x) for x in meta.findall("./article-categories/subj-group/subject")]
        types += [_text(x) for x in meta.findall("./custom-meta-group/custom-meta[meta-name='article-type']/meta-value")]
        license_node = meta.find("./permissions/license")
        license_text = _text(license_node)
        license_url = ""
        if license_node is not None:
            license_url = (
                license_node.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                or license_node.attrib.get("href", "")
            )
            if not license_url:
                ref = license_node.find(".//license-ref")
                if ref is not None:
                    license_url = (
                        ref.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                        or ref.attrib.get("href", "")
                        or _text(ref)
                    )
        sections = _extract_body_sections(article.find("./body"))
        papers.append(Paper(
            pmcid=pmcid, pmid=ids.get("pmid", ""), doi=ids.get("doi", ""), year=year,
            title=title, abstract=abstract, journal=journal,
            article_types=[x for x in types if x],
            license=license_text, license_url=license_url,
            introduction=sections["introduction"],
            methods=sections["methods"],
            results=sections["results"],
            discussion=sections["discussion"],
            conclusions=sections["conclusions"],
            full_text=sections["full_text"],
        ))
    return papers


def _extract_body_sections(body: ET.Element | None) -> dict[str, str]:
    output = {
        "introduction": "", "methods": "", "results": "",
        "discussion": "", "conclusions": "", "full_text": "",
    }
    if body is None:
        return output
    output["full_text"] = _text(body)
    buckets: dict[str, list[str]] = {key: [] for key in output if key != "full_text"}
    aliases = {
        "introduction": ("introduction", "background"),
        "methods": ("method", "materials", "experimental procedure", "study design"),
        "results": ("result", "finding"),
        "discussion": ("discussion", "interpretation"),
        "conclusions": ("conclusion", "summary", "closing remarks"),
    }
    # Top-level sections already include nested subsections, so extracting only
    # these avoids duplicating the same paragraphs several times.
    for section in body.findall("./sec"):
        title = _text(section.find("./title")).lower()
        section_text = _text(section)
        for bucket, terms in aliases.items():
            if any(term in title for term in terms):
                buckets[bucket].append(section_text)
                break
    for bucket, parts in buckets.items():
        output[bucket] = "\n\n".join(parts)
    return output


def _topic_coverage_sample(
    ids_by_category: dict[str, list[str]],
    limit: int,
    min_per_category: int,
    population_by_category: dict[str, int] | None = None,
) -> tuple[list[str], dict[str, str], dict]:
    """Guarantee minimum topic coverage, then approximate the real topic distribution."""
    category_order = list(ids_by_category)
    memberships: dict[str, set[str]] = {}
    for category, ids in ids_by_category.items():
        for pmc_id in ids:
            memberships.setdefault(pmc_id, set()).add(category)

    # 多标签论文按1/k计入每个主题，避免统计真实分布时重复计算同一篇论文。
    observed_fractional_weights = {
        category: sum(
            1.0 / len(memberships[pmc_id])
            for pmc_id in dict.fromkeys(ids_by_category[category])
        )
        for category in category_order
    }
    # 用完整PMC命中量保持真实年度分布，并用已观察到的多主题重叠率进行折减。
    population_weights = {}
    for category in category_order:
        retrieved = len(set(ids_by_category[category]))
        total = (
            population_by_category.get(category, retrieved)
            if population_by_category is not None else retrieved
        )
        overlap_factor = (
            observed_fractional_weights[category] / retrieved if retrieved else 0.0
        )
        population_weights[category] = total * overlap_factor
    selected: list[str] = []
    selected_set: set[str] = set()
    primary_category: dict[str, str] = {}
    counts = {category: 0 for category in category_order}
    positions = {category: 0 for category in category_order}

    def take_next(category: str) -> bool:
        ids = ids_by_category[category]
        position = positions[category]
        while position < len(ids) and ids[position] in selected_set:
            position += 1
        positions[category] = position
        if position >= len(ids):
            return False
        pmc_id = ids[position]
        positions[category] += 1
        selected.append(pmc_id)
        selected_set.add(pmc_id)
        primary_category[pmc_id] = category
        counts[category] += 1
        return True

    # 先处理候选量较少的主题，减少共享论文被热门主题提前占用的风险。
    coverage_order = sorted(
        category_order,
        key=lambda category: (
            len(set(ids_by_category[category])),
            category_order.index(category),
        ),
    )
    for category in coverage_order:
        while counts[category] < min_per_category and len(selected) < limit:
            if not take_next(category):
                break

    remaining = limit - len(selected)
    total_weight = sum(population_weights.values())
    proportional_targets = {
        category: counts[category] + (
            remaining * population_weights[category] / total_weight if total_weight else 0.0
        )
        for category in category_order
    }

    # 先补足按真实候选规模计算的目标，再把无法使用的配额动态转给仍有论文的主题。
    while len(selected) < limit:
        available = [
            category for category in category_order
            if any(pmc_id not in selected_set for pmc_id in ids_by_category[category][positions[category]:])
        ]
        if not available:
            break
        category = max(
            available,
            key=lambda item: (
                proportional_targets[item] - counts[item],
                population_weights[item] / (counts[item] + 1),
                -category_order.index(item),
            ),
        )
        if not take_next(category):
            break

    report = {
        "sampling_strategy": "topic_coverage",
        "requested_limit": limit,
        "min_per_category": min_per_category,
        "unique_candidate_count": len(memberships),
        "raw_candidate_by_category": {
            category: len(set(ids_by_category[category])) for category in category_order
        },
        "annual_pmc_hit_count_by_category": {
            category: (
                population_by_category.get(category, len(set(ids_by_category[category])))
                if population_by_category is not None
                else len(set(ids_by_category[category]))
            )
            for category in category_order
        },
        "fractional_candidate_weight_by_category": {
            category: round(population_weights[category], 3) for category in category_order
        },
        "sampled_by_primary_category": counts,
    }
    return selected, primary_category, report


def collect_year(
    client: PMCClient,
    year: int,
    max_per_category: int | None = None,
    max_per_year: int | None = None,
    min_per_category: int = 0,
    sampling_report: dict | None = None,
) -> list[Paper]:
    # Search cheaply first, merge duplicate PMC IDs across topics, then EFetch each
    # article only once. This is substantially faster than fetching topic by topic.
    ids_by_category: dict[str, list[str]] = {}
    population_by_category: dict[str, int] = {}
    for category in CATEGORIES:
        # 年度限额存在时，每个主题最多搜索到该限额；这里只获取廉价ID，
        # 随后通过轮询选出年度目标数，再对选中的唯一ID执行EFetch。
        search_limit = max_per_category
        if max_per_year is not None:
            search_limit = min(search_limit, max_per_year) if search_limit else max_per_year
        population_by_category[category.name] = client.search_count(year, category.query)
        ids = client.search(year, category.query, search_limit)
        ids_by_category[category.name] = ids

    id_categories: dict[str, set[str]] = {}
    for category_name, ids in ids_by_category.items():
        for pmc_numeric_id in ids:
            id_categories.setdefault(pmc_numeric_id, set()).add(category_name)

    if max_per_year is not None:
        selected_ids, primary_categories, report = _topic_coverage_sample(
            ids_by_category,
            max_per_year,
            min_per_category,
            population_by_category=population_by_category,
        )
    else:
        selected_ids = list(id_categories)
        primary_categories = {}
        report = {
            "sampling_strategy": "all_unique_candidates",
            "requested_limit": None,
            "min_per_category": 0,
            "unique_candidate_count": len(id_categories),
            "raw_candidate_by_category": {
                category: len(set(ids)) for category, ids in ids_by_category.items()
            },
            "annual_pmc_hit_count_by_category": population_by_category,
        }
    if sampling_report is not None:
        sampling_report.update(report)

    merged: dict[str, Paper] = {}
    for paper in client.fetch(selected_ids):
        numeric_id = paper.pmcid.removeprefix("PMC")
        paper.source_categories.update(id_categories.get(numeric_id, set()))
        paper.primary_category = primary_categories.get(numeric_id, "")
        # Every ID came from an ESearch query containing open_access[filter].
        paper.open_access = True
        key = paper.pmcid or paper.doi or paper.title
        if key:
            merged[key] = paper
    return list(merged.values())


def collect_range(
    client: PMCClient,
    start_year: int,
    end_year: int,
    count: int,
    min_per_category: int = 0,
    sampling_report: dict | None = None,
) -> list[Paper]:
    """Collect one deduplicated sample across an inclusive year range."""
    ids_by_category: dict[str, list[str]] = {category.name: [] for category in CATEGORIES}
    population_by_category: dict[str, int] = {category.name: 0 for category in CATEGORIES}
    for year in range(start_year, end_year + 1):
        for category in CATEGORIES:
            population_by_category[category.name] += client.search_count(year, category.query)
            ids_by_category[category.name].extend(client.search(year, category.query, count))

    for category_name, ids in ids_by_category.items():
        ids_by_category[category_name] = list(dict.fromkeys(ids))

    id_categories: dict[str, set[str]] = {}
    for category_name, ids in ids_by_category.items():
        for pmc_numeric_id in ids:
            id_categories.setdefault(pmc_numeric_id, set()).add(category_name)

    selected_ids, primary_categories, report = _topic_coverage_sample(
        ids_by_category,
        count,
        min_per_category,
        population_by_category=population_by_category,
    )
    report.update({"start_year": start_year, "end_year": end_year})
    if sampling_report is not None:
        sampling_report.update(report)

    merged: dict[str, Paper] = {}
    for paper in client.fetch(selected_ids):
        if not start_year <= paper.year <= end_year:
            continue
        numeric_id = paper.pmcid.removeprefix("PMC")
        paper.source_categories.update(id_categories.get(numeric_id, set()))
        paper.primary_category = primary_categories.get(numeric_id, "")
        paper.open_access = True
        key = paper.pmcid or paper.doi or paper.title
        if key:
            merged[key] = paper
    return list(merged.values())
