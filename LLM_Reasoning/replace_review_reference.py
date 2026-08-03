"""Replace the review-only ref2 package with an experimental PMC article."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "test"
OLD_PACKAGE = "ref2_CR23_PMC6746329"
NEW_PACKAGE = "ref2_CR27_PMC4449813"
PMCID = "PMC4449813"
OA_API = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={PMCID}"
XLINK = "{http://www.w3.org/1999/xlink}href"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def text_of(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def main() -> None:
    manifest_path = DATA / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_dir = DATA / "references" / OLD_PACKAGE
    archive_dir = DATA / "excluded_reviews" / OLD_PACKAGE
    if old_dir.exists() and not archive_dir.exists():
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(archive_dir))

    package_dir = DATA / "references" / NEW_PACKAGE
    source_dir = package_dir / "source"
    figures_dir = package_dir / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(OA_API, timeout=60) as response:
        oa_root = ET.fromstring(response.read())
    link = oa_root.find(".//link[@format='tgz']")
    if link is None:
        raise RuntimeError(f"No OA package found for {PMCID}")
    package_url = link.attrib["href"].replace("ftp://", "https://")

    with tempfile.TemporaryDirectory(prefix="pmc_replace_") as temp_name:
        temp_dir = Path(temp_name)
        try:
            archive_path = temp_dir / "article.tar.gz"
            urllib.request.urlretrieve(package_url, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(temp_dir / "unpacked", filter="data")
            nxml_path = next((temp_dir / "unpacked").rglob("*.nxml"))
            xml_bytes = nxml_path.read_bytes()
        except Exception:
            efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=4449813"
            with urllib.request.urlopen(efetch, timeout=60) as response:
                xml_bytes = response.read()
        with urllib.request.urlopen(f"https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/", timeout=60) as response:
            article_html = response.read().decode("utf-8", errors="replace")
        cdn_urls = {
            Path(html.unescape(url)).name: html.unescape(url)
            for url in re.findall(r'https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[^"<]+', article_html)
        }
        (source_dir / "article.xml").write_bytes(xml_bytes)
        tree = ET.fromstring(xml_bytes)
        full_text = text_of(tree)
        (source_dir / "full_text.txt").write_text(full_text, encoding="utf-8")

        figures = []
        for index, fig in enumerate(tree.findall(".//fig"), start=1):
            graphic = fig.find(".//graphic")
            if graphic is None or not graphic.attrib.get(XLINK):
                continue
            href = graphic.attrib[XLINK]
            candidates = list((temp_dir / "unpacked").rglob(href)) if (temp_dir / "unpacked").exists() else []
            if not candidates:
                candidates = list((temp_dir / "unpacked").rglob(f"{Path(href).stem}.*"))
            source_image = candidates[0] if candidates else temp_dir / Path(href).name
            if not candidates:
                image_url = cdn_urls.get(Path(href).name)
                if not image_url:
                    continue
                response = requests.get(image_url, timeout=60)
                response.raise_for_status()
                source_image.write_bytes(response.content)
            figure_id = fig.attrib.get("id", f"F{index}")
            uid = f"{NEW_PACKAGE}__{figure_id}__asset1"
            target_image = figures_dir / f"{uid}{source_image.suffix.lower()}"
            shutil.copy2(source_image, target_image)
            figures.append(
                {
                    "figure_uid": uid,
                    "figure_id_in_article": figure_id,
                    "figure_label": text_of(fig.find("label")) or f"Figure {index}",
                    "asset_index": 1,
                    "caption": text_of(fig.find("caption")),
                    "image_path": target_image.relative_to(DATA).as_posix(),
                    "source_href": href,
                    "mime_type": "image/" + target_image.suffix.lower().lstrip(".").replace("jpg", "jpeg"),
                    "sha256": hashlib.sha256(target_image.read_bytes()).hexdigest(),
                }
            )

    write_json(package_dir / "figures_manifest.json", figures)
    title = text_of(tree.find(".//article-title"))
    abstract = text_of(tree.find(".//abstract"))
    replacement = {
        "package_id": NEW_PACKAGE,
        "reference_number": 2,
        "ref_id": "CR27",
        "pmcid": PMCID,
        "pmid": "25961502",
        "doi": "10.1038/ncb3166",
        "title": title,
        "abstract": abstract,
        "citation_text": f"Kaushik S, Cuervo AM. {title}. Nat Cell Biol. 2015;17:759-770.",
        "directly_supported_fact_ids": ["F4"],
        "source_xml_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "source_mode": "pmc_open_access_package",
        "package_path": f"references/{NEW_PACKAGE}",
        "full_text_path": f"references/{NEW_PACKAGE}/source/full_text.txt",
        "article_xml_path": f"references/{NEW_PACKAGE}/source/article.xml",
        "figures_manifest_path": f"references/{NEW_PACKAGE}/figures_manifest.json",
        "figure_count": len(figures),
        "replacement_note": f"Replaces review {OLD_PACKAGE}; experimental primary research supporting F4.",
    }
    manifest["references"] = [
        replacement if item["package_id"] == OLD_PACKAGE else item
        for item in manifest["references"]
    ]
    manifest.setdefault("reference_replacements", []).append(
        {"removed_package": OLD_PACKAGE, "replacement_package": NEW_PACKAGE, "reason": "Review figures were all schematics; primary experimental evidence required."}
    )
    write_json(manifest_path, manifest)
    print(f"Replaced {OLD_PACKAGE} with {NEW_PACKAGE}; extracted {len(figures)} figures.")


if __name__ == "__main__":
    main()
