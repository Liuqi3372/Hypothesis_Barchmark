import xml.etree.ElementTree as ET

from pmc_m.pmc import parse_articles


def test_parse_jats_article():
    root = ET.fromstring("""<pmc-articleset><article article-type="research-article">
      <front><journal-meta><journal-title-group><journal-title>Cell Test</journal-title></journal-title-group></journal-meta>
      <article-meta>
        <article-id pub-id-type="pmc">123</article-id><article-id pub-id-type="pmid">456</article-id>
        <article-id pub-id-type="doi">10.1/x</article-id>
        <title-group><article-title>A <italic>cell</italic> study</article-title></title-group>
        <pub-date pub-type="epub"><year>2025</year></pub-date>
        <permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/"><p>CC BY 4.0</p></license></permissions>
        <abstract><p>We measured cells and found a mechanistic result.</p></abstract>
      </article-meta></front>
      <body>
        <sec><title>Methods</title><p>We cultured cells and performed CRISPR perturbation.</p></sec>
        <sec><title>Results</title><p>Gene loss changed the cellular phenotype.</p></sec>
        <sec><title>Conclusion</title><p>The gene controls this cell process.</p></sec>
      </body>
      </article></pmc-articleset>""")
    paper = parse_articles(root)[0]
    assert paper.pmcid == "PMC123"
    assert paper.title == "A cell study"
    assert paper.year == 2025
    assert "mechanistic result" in paper.abstract
    assert paper.license_url == "https://creativecommons.org/licenses/by/4.0/"
    assert "CRISPR perturbation" in paper.methods
    assert "cellular phenotype" in paper.results
    assert "controls this cell process" in paper.conclusions
    assert "Methods" in paper.full_text
