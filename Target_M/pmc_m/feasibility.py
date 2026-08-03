from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
from openai import OpenAI

from .pmc import PMCClient, _text


FACT_PROMPT = """You are a rigorous cell-biology evidence curator. Use only the supplied target-paper Abstract and Introduction.

Your task is to construct a QUALIFIED set of 2-6 Known Facts for later reasoning. This is not a minimal-subset or ablation task. Every returned Fact must satisfy every rule below.

Fact eligibility rules:
1. The Fact is established prior knowledge, never a result newly reported by the target paper.
2. The Fact states one specific, atomic, experimentally established biological relationship or effect (for example, entity/process A regulates, is required for, localizes with, or measurably changes B under a stated condition).
3. The Fact is necessary to define the target paper's Knowledge Gap and has a clear role in the later reasoning chain.
4. The Fact has a verbatim factual anchor copied from the Introduction. Copy the shortest complete clause or sentence that preserves the claim; do not paraphrase it.
5. nearby_reference_ids contains only citations that directly accompany and support that anchored claim. Do not include every citation in the paragraph.
6. Prefer citations whose reference-catalog entry appears to be an original experimental article and has a PMID, PMCID, or DOI that can be resolved. A review may be selected only when its cited original experiment can subsequently be traced.
7. One returned Fact must contain exactly ONE independently testable subject-relation-object claim. If one Introduction sentence contains two experimentally separable claims, split it into two Facts and count them separately. The two split Facts may share the same verbatim anchor and may cite the same reference; shared evidence is allowed and does not merge them back into one Fact.

Never return these as Known Facts:
- broad textbook background, a general definition, field motivation, disease prevalence, or a statement that merely introduces a topic;
- vague statements such as "X is important", "X has many roles", or "little is known";
- review-style summaries combining several mechanisms or claims into one Fact;
- an open question, knowledge gap, hypothesis, speculation, or the target paper's own new finding;
- a claim for which the Introduction supplies only a background-related citation rather than direct evidence.

Before returning JSON, silently split composite claims, then check each atomic Fact against the reference_catalog and discard weak/background-only Facts. It is better to return 2 strong atomic Facts than 6 weak Facts, but never return fewer than 2 or more than 6. Number them consecutively F1, F2, ... only after splitting and quality filtering.

Also state the Knowledge Gap and a testable Hypothesis only to evaluate whether this qualified evidence chain is feasible. Do not use Results, Discussion, or the target paper's conclusion.
Return JSON only:
{"candidate_facts":[{"fact_id":"F1","statement":"one atomic claim","anchor_text":"verbatim Introduction text","anchor_paragraph_index":1,"nearby_reference_ids":["R1"],"role_in_gap":"...","atomic_claim_count":1,"source_claim_group_id":"A1","atomicity_check":"why no independently testable second relation remains"}],"knowledge_gap":"...","hypothesis":"..."}"""

EVIDENCE_PROMPT = """You are auditing evidence for Candidate Known Facts. Assess only and every pair listed in required_pairs, independently.
DIRECT means the reference itself reports evidence that explicitly establishes the Fact. BACKGROUND means related but not direct. NONE means unsupported.
Classify article_type as PRIMARY_EXPERIMENTAL, REVIEW, NON_EXPERIMENTAL, or UNCERTAIN. A review never counts as final evidence even when it states the Fact.
Figure relevance is based only on supplied captions because no image pixels are available. Set has_relevant_experimental_figure true only when a caption describes measured, imaged, quantified, or otherwise real experimental data directly relevant to the Fact; schematics, models, simulations, networks, decorative images, and unrelated figures do not count.
Do not infer beyond the supplied text. Return JSON only:
{"assessments":[{"fact_id":"F1","reference_id":"R1","support_level":"DIRECT|BACKGROUND|NONE","article_type":"PRIMARY_EXPERIMENTAL|REVIEW|NON_EXPERIMENTAL|UNCERTAIN","evidence_quote":"short exact evidence","evidence_locator":"abstract|full text|caption|none","has_relevant_experimental_figure":true,"relevant_figure_ids":["fig1"],"reason":"..."}]}"""

TRACE_PROMPT = """You are tracing a review article to the original experiments behind one Candidate Known Fact.
Use only the supplied review citation contexts. Select at most 8 cited references that are most likely to be original experimental sources directly supporting the Fact. Do not select citations merely because they discuss the broad topic. Return JSON only:
{"selected_reference_ids":[{"reference_id":"R12","reason":"..."}]}"""


@dataclass
class ReferenceRecord:
    reference_id: str
    citation_text: str
    title: str = ""
    journal: str = ""
    year: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    abstract: str = ""
    full_text: str = ""
    article_types: list[str] | None = None
    figure_captions: list[dict] | None = None
    citation_contexts: list[dict] | None = None
    source_reference_id: str = ""
    trace_depth: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class DeepSeekJSON:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def call(self, prompt: str, payload: dict, cache_path: Path, force: bool = False) -> dict:
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        last_error = None
        for _attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=16384,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"},
                stream=False,
            )
            try:
                result = json.loads((response.choices[0].message.content or "{}").strip())
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"Model did not return complete JSON after 3 attempts: {last_error}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def extract_article_data(root: ET.Element) -> dict:
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        raise ValueError("PMC XML has no article")
    meta = article.find("./front/article-meta")
    ids = {
        node.attrib.get("pub-id-type", "").lower(): _text(node)
        for node in meta.findall("./article-id")
    } if meta is not None else {}
    body = article.find("./body")
    introduction = None
    for section in article.findall("./body//sec"):
        label = f"{section.attrib.get('sec-type', '')} {_text(section.find('./title'))}".lower()
        if "intro" in label or "background" in label:
            introduction = section
            break
    if introduction is None and body is not None:
        introduction = body.find("./sec") or body

    references = {}
    for ordinal, ref in enumerate(article.findall(".//back//ref-list//ref"), start=1):
        citation = ref.find("./element-citation")
        if citation is None:
            citation = ref.find("./mixed-citation")
        if citation is None:
            citation = ref
        pub_ids = {
            node.attrib.get("pub-id-type", "").lower(): _text(node)
            for node in citation.findall(".//pub-id")
        }
        pmcid = pub_ids.get("pmcid", "") or pub_ids.get("pmc", "")
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        ref_id = ref.attrib.get("id", "") or f"R{ordinal}"
        references[ref_id] = ReferenceRecord(
            reference_id=ref_id,
            citation_text=html.unescape(_text(citation)),
            title=html.unescape(_text(citation.find(".//article-title"))),
            journal=html.unescape(_text(citation.find(".//source"))),
            year=_text(citation.find(".//year")),
            pmid=pub_ids.get("pmid", ""), pmcid=pmcid, doi=pub_ids.get("doi", ""),
            citation_contexts=[],
        )

    paragraph_rows = []
    scope = introduction if introduction is not None else body
    if scope is not None:
        for index, paragraph in enumerate(scope.findall(".//p"), start=1):
            context = html.unescape(_text(paragraph))
            ref_ids = []
            for xref in paragraph.findall(".//xref[@ref-type='bibr']"):
                ref_ids.extend(xref.attrib.get("rid", "").split())
            paragraph_rows.append({"paragraph_index": index, "text": context, "reference_ids": list(dict.fromkeys(ref_ids))})
            for ref_id in ref_ids:
                if ref_id in references:
                    references[ref_id].citation_contexts.append({"paragraph_index": index, "context": context})

    figures = []
    for index, fig in enumerate(article.findall(".//fig"), start=1):
        figures.append({
            "figure_id": fig.attrib.get("id", f"fig{index}"),
            "label": _text(fig.find("./label")),
            "caption": html.unescape(_text(fig.find("./caption"))),
        })
    article_types = [article.attrib.get("article-type", "")]
    if meta is not None:
        article_types.extend(_text(node) for node in meta.findall("./article-categories//subject"))
    abstract = " ".join(_text(node) for node in article.findall("./front/article-meta/abstract"))
    return {
        "ids": ids,
        "title": html.unescape(_text(article.find("./front/article-meta/title-group/article-title"))),
        "abstract": html.unescape(abstract),
        "introduction": html.unescape(_text(introduction)),
        "introduction_paragraphs": paragraph_rows,
        "full_text": html.unescape(_text(body)),
        "article_types": [value for value in article_types if value],
        "figures": figures,
        "references": references,
    }


def resolve_reference(client: PMCClient, reference: ReferenceRecord) -> ReferenceRecord:
    try:
        if not reference.pmid and reference.doi:
            result = client._request("esearch.fcgi", {"db": "pubmed", "term": f'"{reference.doi}"[AID]', "retmode": "json", "retmax": 1}).json()
            ids = result.get("esearchresult", {}).get("idlist", [])
            reference.pmid = ids[0] if ids else ""
        if not reference.pmcid and reference.pmid:
            xml = ET.fromstring(client._request("elink.fcgi", {"dbfrom": "pubmed", "db": "pmc", "id": reference.pmid, "retmode": "xml", "linkname": "pubmed_pmc"}).content)
            value = _text(xml.find(".//LinkSetDb/Link/Id"))
            reference.pmcid = f"PMC{value}" if value else ""
        if reference.pmcid:
            xml = ET.fromstring(client._request("efetch.fcgi", {"db": "pmc", "id": reference.pmcid.removeprefix("PMC"), "retmode": "xml"}).content)
            data = extract_article_data(xml)
            reference.title = reference.title or data["title"]
            reference.abstract = data["abstract"]
            reference.full_text = data["full_text"]
            reference.article_types = data["article_types"]
            reference.figure_captions = data["figures"]
        elif reference.pmid:
            xml = ET.fromstring(client._request("efetch.fcgi", {"db": "pubmed", "id": reference.pmid, "retmode": "xml"}).content)
            reference.abstract = html.unescape(" ".join(_text(node) for node in xml.findall(".//Abstract/AbstractText")))
    except (httpx.HTTPError, ET.ParseError, ValueError):
        pass
    reference.article_types = reference.article_types or []
    reference.figure_captions = reference.figure_captions or []
    reference.citation_contexts = reference.citation_contexts or []
    return reference


def evidence_payload(facts: list[dict], references: list[ReferenceRecord]) -> dict:
    return {
        "facts": facts,
        "references": [
            {
                "reference_id": ref.reference_id,
                "title": ref.title,
                "article_types": ref.article_types,
                "abstract": ref.abstract[:12000],
                "full_text": ref.full_text[:30000],
                "figure_captions": ref.figure_captions,
                "full_text_available": bool(ref.full_text),
            }
            for ref in references
        ],
        "required_pairs": [
            {"fact_id": fact["fact_id"], "reference_id": ref.reference_id}
            for fact in facts for ref in references
            if not fact.get("nearby_reference_ids")
            or ref.reference_id in fact.get("nearby_reference_ids", [])
            or ref.trace_depth > 0
        ],
    }


def validate_facts(result: dict, introduction: str, known_reference_ids: set[str]) -> list[dict]:
    facts = result.get("candidate_facts", [])
    if not 2 <= len(facts) <= 6:
        raise ValueError("Candidate Known Facts must contain 2-6 items")
    for fact in facts:
        anchor = fact.get("anchor_text", "")
        if not anchor or normalize_anchor(anchor) not in normalize_anchor(introduction):
            raise ValueError(f"Fact {fact.get('fact_id')} has no verbatim Introduction anchor")
        ids = set(fact.get("nearby_reference_ids", []))
        if not ids or not ids <= known_reference_ids:
            raise ValueError(f"Fact {fact.get('fact_id')} has invalid nearby references")
        if fact.get("atomic_claim_count") != 1:
            raise ValueError(f"Fact {fact.get('fact_id')} is not declared atomic")
    return facts


def normalize_anchor(value: str) -> str:
    """Normalize XML/Unicode typography while preserving the words in an anchor."""
    value = unicodedata.normalize("NFKC", html.unescape(value or ""))
    value = value.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u00a0": " ",
    }))
    return re.sub(r"\s+", " ", value).strip()


def eligible_assessment(item: dict, reference: ReferenceRecord) -> bool:
    return bool(
        item.get("support_level") == "DIRECT"
        and item.get("article_type") == "PRIMARY_EXPERIMENTAL"
        and item.get("has_relevant_experimental_figure") is True
        and reference.full_text
        and reference.pmcid
    )
