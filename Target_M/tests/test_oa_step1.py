import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path


def load_oa_step1():
    path = Path(__file__).resolve().parents[1] / "OA-step1_collect_pmc.py"
    spec = importlib.util.spec_from_file_location("oa_step1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_unique_introduction_references_enter_oa_denominator():
    module = load_oa_step1()
    root = ET.fromstring("""
    <article><body>
      <sec sec-type="intro"><title>Introduction</title><p>
        Known facts <xref ref-type="bibr" rid="R1 R2">1,2</xref>.
        Repeated <xref ref-type="bibr" rid="R1">1</xref>.
      </p></sec>
      <sec><title>Results</title><p><xref ref-type="bibr" rid="R3">3</xref></p></sec>
    </body><back><ref-list>
      <ref id="R1"><element-citation><pub-id pub-id-type="pmid">1</pub-id></element-citation></ref>
      <ref id="R2"><element-citation><pub-id pub-id-type="doi">10.1/x</pub-id></element-citation></ref>
      <ref id="R3"><element-citation><pub-id pub-id-type="pmid">3</pub-id></element-citation></ref>
    </ref-list></back></article>
    """)
    auditor = object.__new__(module.OAReferenceAuditor)
    references = auditor.extract_unique_references(root)
    assert [item["reference_id"] for item in references] == ["R1", "R2"]
