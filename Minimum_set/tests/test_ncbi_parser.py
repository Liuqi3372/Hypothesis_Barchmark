import xml.etree.ElementTree as ET

from mrrs.ncbi import NCBIClient


def test_reference_parser_extracts_identifiers(monkeypatch):
    root = ET.fromstring("""<article><back><ref-list>
      <ref id="R1"><element-citation>
        <person-group><name><surname>Smith</surname><given-names>J</given-names></name></person-group>
        <article-title>A mechanistic paper</article-title><source>Cell</source><year>2020</year>
        <pub-id pub-id-type="pmid">123</pub-id><pub-id pub-id-type="pmc">456</pub-id>
        <pub-id pub-id-type="doi">10.1/test</pub-id>
      </element-citation></ref>
    </ref-list></back></article>""")
    client = object.__new__(NCBIClient)
    monkeypatch.setattr(client, "fetch_pmc_xml", lambda pmcid: root)
    reference = client.extract_references("PMC1")[0]
    assert reference.ref_id == "R1"
    assert reference.title == "A mechanistic paper"
    assert reference.pmid == "123"
    assert reference.pmcid == "PMC456"
    assert reference.doi == "10.1/test"


def test_introduction_citations_include_paragraph_context():
    root = ET.fromstring("""<pmc-articleset><article><body>
      <sec sec-type="intro"><title>Introduction</title>
        <p>Known mechanism <xref ref-type="bibr" rid="R1 R2">1,2</xref> motivates the gap.</p>
      </sec><sec><title>Results</title><p><xref ref-type="bibr" rid="R3">3</xref></p></sec>
    </body></article></pmc-articleset>""")
    client = object.__new__(NCBIClient)
    result = client.extract_introduction_citations(root)
    assert [item["ref_id"] for item in result["citation_mentions"]] == ["R1", "R2"]
    assert "motivates the gap" in result["citation_mentions"][0]["context"]
