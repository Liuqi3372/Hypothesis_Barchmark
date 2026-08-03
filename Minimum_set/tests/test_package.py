import xml.etree.ElementTree as ET

from mrrs.package import article_intro, extract_figures


def test_package_extracts_target_introduction_and_figure_caption():
    root = ET.fromstring("""
    <pmc-articleset><article><body><sec sec-type="intro"><title>Introduction</title>
    <p>Known fact <xref ref-type="bibr" rid="R1">1</xref>.</p></sec></body>
    <fig id="F1"><label>Figure 1</label><caption><p>Measured cells.</p></caption>
    <graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="f1.jpg"/></fig>
    </article></pmc-articleset>
    """)
    assert "Known fact" in article_intro(root)
    figures = extract_figures(root)
    assert figures[0]["caption"] == "Measured cells."
    assert figures[0]["source_href"] == "f1.jpg"
