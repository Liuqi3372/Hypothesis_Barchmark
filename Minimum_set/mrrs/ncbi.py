from __future__ import annotations

import html
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _text(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


@dataclass
class Reference:
    ref_id: str
    ordinal: int
    citation_text: str
    title: str = ""
    journal: str = ""
    year: str = ""
    authors: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    evidence_source: str = "citation_only"
    abstract: str = ""
    full_text_excerpt: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class NCBIClient:
    def __init__(self, email: str, api_key: str = "", timeout: float = 90.0):
        if not email:
            raise ValueError("NCBI contact email is required")
        self.common = {"tool": "mrrs_builder", "email": email}
        if api_key:
            self.common["api_key"] = api_key
        self.interval = 0.11 if api_key else 0.36
        self.last_call = 0.0
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": f"mrrs_builder/0.1 ({email})"},
        )

    def close(self) -> None:
        self.client.close()

    def _get(self, endpoint: str, params: dict, retries: int = 5) -> httpx.Response:
        for attempt in range(retries):
            wait = self.interval - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.client.get(
                    f"{EUTILS}/{endpoint}", params={**self.common, **params}
                )
                self.last_call = time.monotonic()
                response.raise_for_status()
                return response
            except httpx.HTTPError:
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise RuntimeError("unreachable")

    def fetch_pmc_xml(self, pmcid: str) -> ET.Element:
        value = pmcid.upper().removeprefix("PMC")
        xml = self._get("efetch.fcgi", {
            "db": "pmc", "id": value, "retmode": "xml",
        }).content
        return ET.fromstring(xml)

    def extract_references(self, pmcid: str) -> list[Reference]:
        root = self.fetch_pmc_xml(pmcid)
        return self.extract_references_from_root(root)

    def extract_references_from_root(self, root: ET.Element) -> list[Reference]:
        references = []
        for ordinal, ref_node in enumerate(root.findall(".//back//ref-list//ref"), 1):
            citation = ref_node.find("./element-citation")
            if citation is None:
                citation = ref_node.find("./mixed-citation")
            if citation is None:
                citation = ref_node.find("./citation")
            if citation is None:
                citation = ref_node
            ids = {}
            for pub_id in citation.findall(".//pub-id"):
                ids[pub_id.attrib.get("pub-id-type", "").lower()] = _text(pub_id)
            names = []
            for name in citation.findall(".//person-group/name"):
                value = " ".join(filter(None, [
                    _text(name.find("./surname")), _text(name.find("./given-names"))
                ]))
                if value:
                    names.append(value)
            pmcid_value = ids.get("pmcid", "") or ids.get("pmc", "")
            if pmcid_value and not pmcid_value.upper().startswith("PMC"):
                pmcid_value = f"PMC{pmcid_value}"
            references.append(Reference(
                ref_id=ref_node.attrib.get("id", "") or f"R{ordinal}",
                ordinal=ordinal,
                citation_text=html.unescape(_text(citation)),
                title=html.unescape(_text(citation.find(".//article-title"))),
                journal=html.unescape(_text(citation.find(".//source"))),
                year=_text(citation.find(".//year")),
                authors="; ".join(names),
                pmid=ids.get("pmid", ""), pmcid=pmcid_value,
                doi=ids.get("doi", ""),
            ))
        return references

    def extract_article_content(self, root: ET.Element) -> dict:
        article = root.find(".//article")
        if article is None:
            raise ValueError("PMC XML contains no article")
        body = article.find("./body")
        meta = article.find("./front/article-meta")
        journal_meta = article.find("./front/journal-meta")
        title = _text(meta.find("./title-group/article-title")) if meta is not None else ""
        journal = (
            _text(journal_meta.find("./journal-title-group/journal-title"))
            if journal_meta is not None else ""
        )
        abstract = " ".join(
            _text(node) for node in article.findall("./front/article-meta/abstract")
        ).strip()
        sections = {
            "introduction": [], "methods": [], "results": [],
            "discussion": [], "conclusions": [],
        }
        aliases = {
            "introduction": ("introduction", "background"),
            "methods": ("method", "materials", "experimental procedure", "study design"),
            "results": ("result", "finding"),
            "discussion": ("discussion", "interpretation"),
            "conclusions": ("conclusion", "summary", "closing remarks"),
        }
        if body is not None:
            for section in body.findall("./sec"):
                section_title = _text(section.find("./title")).lower()
                section_type = section.attrib.get("sec-type", "").lower()
                label = f"{section_type} {section_title}"
                for bucket, terms in aliases.items():
                    if any(term in label for term in terms):
                        sections[bucket].append(_text(section))
                        break
        return {
            "title": html.unescape(title),
            "journal": html.unescape(journal),
            "abstract": html.unescape(abstract),
            **{key: "\n\n".join(value) for key, value in sections.items()},
            "full_text": html.unescape(_text(body)),
        }

    def extract_introduction_citations(self, root: ET.Element) -> dict:
        """Return introduction text plus every in-text bibliographic citation context."""
        body = root.find(".//article/body")
        if body is None:
            return {"introduction_text": "", "citation_mentions": []}
        introduction = None
        for section in body.findall(".//sec"):
            title = _text(section.find("./title")).lower()
            section_type = section.attrib.get("sec-type", "").lower()
            if section_type in {"intro", "introduction"} or title in {
                "introduction", "background"
            }:
                introduction = section
                break
        if introduction is None:
            top_sections = body.findall("./sec")
            introduction = top_sections[0] if top_sections else body
        mentions = []
        seen = set()
        paragraphs = introduction.findall(".//p")
        for paragraph_index, paragraph in enumerate(paragraphs, 1):
            context = html.unescape(_text(paragraph))
            for xref in paragraph.findall(".//xref[@ref-type='bibr']"):
                for ref_id in xref.attrib.get("rid", "").split():
                    key = (ref_id, paragraph_index)
                    if ref_id and key not in seen:
                        seen.add(key)
                        mentions.append({
                            "ref_id": ref_id,
                            "paragraph_index": paragraph_index,
                            "citation_marker": _text(xref),
                            "context": context,
                        })
        return {
            "introduction_text": html.unescape(_text(introduction)),
            "citation_mentions": mentions,
        }

    def resolve_pmid(self, reference: Reference) -> str:
        if reference.pmid:
            return reference.pmid
        if not reference.doi:
            return ""
        payload = self._get("esearch.fcgi", {
            "db": "pubmed", "term": f'"{reference.doi}"[AID]',
            "retmode": "json", "retmax": 1,
        }).json().get("esearchresult", {})
        ids = payload.get("idlist", [])
        return ids[0] if ids else ""

    def resolve_pmcid(self, pmid: str) -> str:
        if not pmid:
            return ""
        xml = self._get("elink.fcgi", {
            "dbfrom": "pubmed", "db": "pmc", "id": pmid,
            "retmode": "xml", "linkname": "pubmed_pmc",
        }).content
        root = ET.fromstring(xml)
        value = _text(root.find(".//LinkSetDb/Link/Id"))
        return f"PMC{value}" if value else ""

    def fetch_pubmed_abstract(self, pmid: str) -> str:
        if not pmid:
            return ""
        xml = self._get("efetch.fcgi", {
            "db": "pubmed", "id": pmid, "retmode": "xml",
        }).content
        root = ET.fromstring(xml)
        parts = [_text(node) for node in root.findall(".//Abstract/AbstractText")]
        return html.unescape(" ".join(part for part in parts if part))

    def fetch_pmc_excerpt(self, pmcid: str, maximum: int = 30000) -> str:
        if not pmcid:
            return ""
        root = self.fetch_pmc_xml(pmcid)
        body = root.find(".//article/body")
        text = _text(body)
        return text[:maximum]

    def enrich_reference(self, reference: Reference) -> Reference:
        try:
            reference.pmid = self.resolve_pmid(reference)
            if not reference.pmcid and reference.pmid:
                reference.pmcid = self.resolve_pmcid(reference.pmid)
            if reference.pmcid:
                reference.full_text_excerpt = self.fetch_pmc_excerpt(
                    reference.pmcid, maximum=10_000_000
                )
            if reference.pmid:
                reference.abstract = self.fetch_pubmed_abstract(reference.pmid)
        except (httpx.HTTPError, ET.ParseError, ValueError):
            pass
        if reference.full_text_excerpt:
            reference.evidence_source = "pmc_full_text_excerpt"
        elif reference.abstract:
            reference.evidence_source = "pubmed_abstract"
        return reference
