import xml.etree.ElementTree as ET

import pytest

from pmc_m.feasibility import ReferenceRecord, eligible_assessment, extract_article_data, validate_facts
from step3_reference_feasibility import parse_scan_range


def test_limit_accepts_prefix_or_inclusive_csv_interval():
    assert parse_scan_range("100") == (1, 100)
    assert parse_scan_range("100-200") == (100, 200)
    with pytest.raises(Exception):
        parse_scan_range("200-100")


def test_every_fact_requires_a_verbatim_anchor_and_known_reference():
    introduction = "Established membrane contact supports lipid transfer."
    result = {"candidate_facts": [
        {"fact_id": "F1", "anchor_text": introduction, "nearby_reference_ids": ["R1"], "atomic_claim_count": 1},
        {"fact_id": "F2", "anchor_text": introduction, "nearby_reference_ids": ["R1"], "atomic_claim_count": 1},
    ]}
    assert len(validate_facts(result, introduction, {"R1"})) == 2
    result["candidate_facts"][0]["anchor_text"] = "invented anchor"
    with pytest.raises(ValueError):
        validate_facts(result, introduction, {"R1"})


def test_anchor_validation_tolerates_xml_typography_and_whitespace():
    introduction = "Protein A\u00a0regulates B\u2014under stress. A second established fact [2]."
    result = {"candidate_facts": [
        {"fact_id": "F1", "anchor_text": "Protein A regulates B-under stress.", "nearby_reference_ids": ["R1"], "atomic_claim_count": 1},
        {"fact_id": "F2", "anchor_text": "A second established fact [2].", "nearby_reference_ids": ["R2"], "atomic_claim_count": 1},
    ]}
    assert len(validate_facts(result, introduction, {"R1", "R2"})) == 2


def test_review_never_counts_as_final_coverage():
    reference = ReferenceRecord(
        reference_id="R1", citation_text="x", pmcid="PMC1", full_text="evidence"
    )
    review = {
        "support_level": "DIRECT", "article_type": "REVIEW",
        "has_relevant_experimental_figure": True,
    }
    primary = {**review, "article_type": "PRIMARY_EXPERIMENTAL"}
    assert not eligible_assessment(review, reference)
    assert eligible_assessment(primary, reference)


def test_extracts_introduction_citations_and_figure_captions():
    root = ET.fromstring("""
    <article><front><article-meta><article-id pub-id-type="pmc">1</article-id>
    <title-group><article-title>Test</article-title></title-group><abstract><p>A</p></abstract>
    </article-meta></front><body><sec sec-type="intro"><title>Introduction</title>
    <p>Known fact <xref ref-type="bibr" rid="R1">1</xref>.</p></sec></body>
    <fig id="F1"><label>Figure 1</label><caption><p>Measured cells.</p></caption></fig>
    <back><ref-list><ref id="R1"><element-citation><article-title>Source</article-title>
    <pub-id pub-id-type="pmcid">PMC2</pub-id></element-citation></ref></ref-list></back></article>
    """)
    data = extract_article_data(root)
    assert data["introduction_paragraphs"][0]["reference_ids"] == ["R1"]
    assert data["references"]["R1"].pmcid == "PMC2"
    assert data["figures"][0]["caption"] == "Measured cells."
