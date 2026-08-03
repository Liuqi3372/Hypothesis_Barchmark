from __future__ import annotations

import hashlib
import html
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from .ncbi import NCBIClient, _text


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def article_text(root: ET.Element) -> str:
    return html.unescape(_text(root.find(".//article/body")))


def article_intro(root: ET.Element) -> str:
    body = root.find(".//article/body")
    if body is None:
        return ""
    for section in body.findall(".//sec"):
        label = f"{section.attrib.get('sec-type', '')} {_text(section.find('./title'))}".lower()
        if "intro" in label or "background" in label:
            return html.unescape(_text(section))
    first = body.find("./sec")
    return html.unescape(_text(first if first is not None else body))


def extract_figures(root: ET.Element) -> list[dict]:
    figures = []
    for index, fig in enumerate(root.findall(".//article//fig"), start=1):
        graphic = fig.find(".//graphic")
        href = ""
        if graphic is not None:
            href = graphic.attrib.get("{http://www.w3.org/1999/xlink}href", "")
        figures.append({
            "figure_id": fig.attrib.get("id", f"F{index}"),
            "label": _text(fig.find("./label")) or f"Figure {index}",
            "caption": html.unescape(_text(fig.find("./caption"))),
            "source_href": href,
        })
    return figures


def download_figures(client: NCBIClient, pmcid: str, root: ET.Element, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    archive_members: dict[str, tarfile.TarInfo] = {}
    archive = None
    try:
        oa_response = client.client.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            params={"id": pmcid},
        )
        oa_response.raise_for_status()
        oa_root = ET.fromstring(oa_response.content)
        link = oa_root.find(".//link[@format='tgz']")
        if link is not None and link.attrib.get("href"):
            archive_url = link.attrib["href"].replace("ftp://", "https://")
            archive_response = client.client.get(archive_url)
            archive_response.raise_for_status()
            archive = tarfile.open(fileobj=io.BytesIO(archive_response.content), mode="r:gz")
            archive_members = {
                Path(member.name).name.lower(): member
                for member in archive.getmembers()
                if member.isfile()
            }
    except (httpx.HTTPError, ET.ParseError, tarfile.TarError):
        archive = None
        archive_members = {}

    manifest = []
    for figure in extract_figures(root):
        href = figure["source_href"]
        item = {**figure, "download_status": "MISSING", "image_path": "", "sha256": ""}
        source_name = Path(href).name.lower()
        member = archive_members.get(source_name)
        if member is None and source_name:
            source_stem = Path(source_name).stem
            member = next(
                (candidate for name, candidate in archive_members.items()
                 if Path(name).stem == source_stem),
                None,
            )
        if archive is not None and member is not None:
            stream = archive.extractfile(member)
            if stream is not None:
                suffix = Path(member.name).suffix.lower() or Path(href).suffix.lower() or ".jpg"
                target = output / f"{figure['figure_id']}{suffix}"
                target.write_bytes(stream.read())
                item.update({
                    "download_status": "DOWNLOADED",
                    "image_path": target.name,
                    "sha256": sha256(target),
                })
        manifest.append(item)
    if archive is not None:
        archive.close()
    return manifest


def build_package(
    row: dict,
    audit: dict,
    client: NCBIClient,
    package_dir: Path,
) -> dict:
    pmcid = str(row["pmcid"]).upper()
    target_root = client.fetch_pmc_xml(pmcid)
    target_dir = package_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_xml = target_dir / "article.xml"
    ET.ElementTree(target_root).write(target_xml, encoding="utf-8", xml_declaration=True)
    (target_dir / "introduction.txt").write_text(article_intro(target_root), encoding="utf-8")
    write_json(target_dir / "metadata.json", {
        key: row.get(key) for key in ("pmcid", "pmid", "doi", "year", "title", "journal", "category", "abstract")
    })

    candidate_reasoning = audit["candidate_reasoning"]
    write_json(package_dir / "candidate_reasoning.json", candidate_reasoning)
    write_json(package_dir / "fact_coverage.json", audit["fact_coverage"])

    eligible_ids = list(dict.fromkeys(
        ref_id
        for coverage in audit["fact_coverage"]
        for ref_id in coverage.get("eligible_oa_primary_reference_ids", [])
    ))
    evidence_by_pair = {
        (item.get("fact_id"), item.get("reference_id")): item
        for item in audit.get("evidence_assessments", [])
    }
    reference_by_id = {item["reference_id"]: item for item in audit.get("references", [])}
    references_manifest = []
    fact_reference_map = []
    for index, ref_id in enumerate(eligible_ids, start=1):
        source = reference_by_id[ref_id]
        ref_pmcid = str(source["pmcid"]).upper()
        ref_dir = package_dir / "references" / f"ref{index}_{ref_pmcid}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_root = client.fetch_pmc_xml(ref_pmcid)
        xml_path = ref_dir / "article.xml"
        ET.ElementTree(ref_root).write(xml_path, encoding="utf-8", xml_declaration=True)
        (ref_dir / "full_text.txt").write_text(article_text(ref_root), encoding="utf-8")
        figures = download_figures(client, ref_pmcid, ref_root, ref_dir / "figures")
        write_json(ref_dir / "figures_manifest.json", figures)
        metadata = {
            "package_reference_id": f"REF{index}", "source_reference_id": ref_id,
            **{key: source.get(key) for key in ("pmcid", "pmid", "doi", "title", "journal", "year", "citation_text", "source_reference_id", "trace_depth")},
            "xml_sha256": sha256(xml_path),
        }
        write_json(ref_dir / "metadata.json", metadata)
        references_manifest.append({
            **metadata, "relative_path": ref_dir.relative_to(package_dir).as_posix(),
            "figure_count": len(figures),
            "downloaded_figure_count": sum(item["download_status"] == "DOWNLOADED" for item in figures),
        })
        for fact in candidate_reasoning["candidate_facts"]:
            assessment = evidence_by_pair.get((fact["fact_id"], ref_id))
            if assessment and assessment.get("support_level") == "DIRECT":
                fact_reference_map.append({
                    "fact_id": fact["fact_id"], "package_reference_id": f"REF{index}",
                    "source_reference_id": ref_id, "step3_assessment": assessment,
                })

    write_json(package_dir / "references_manifest.json", references_manifest)
    write_json(package_dir / "fact_reference_map.json", fact_reference_map)
    manifest = {
        "schema_version": "minimum-set-package-1.0",
        "package_id": package_dir.name,
        "target_pmcid": pmcid,
        "source_final_row": row,
        "candidate_fact_count": len(candidate_reasoning["candidate_facts"]),
        "eligible_reference_count": len(references_manifest),
        "target": {
            "xml": "target/article.xml", "metadata": "target/metadata.json",
            "introduction": "target/introduction.txt",
        },
        "candidate_reasoning": "candidate_reasoning.json",
        "fact_coverage": "fact_coverage.json",
        "fact_reference_map": "fact_reference_map.json",
        "references_manifest": "references_manifest.json",
    }
    write_json(package_dir / "manifest.json", manifest)
    return manifest
