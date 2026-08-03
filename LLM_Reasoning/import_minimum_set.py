from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from batch_io import parse_limit_interval, read_csv_rows, select_interval, write_csv_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = PROJECT_ROOT / "Minimum_set" / "data" / "step1" / "m1_PMC11672324"
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "Minimum_set" / "data" / "step2" / "PMC11672324_ground_truth.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "PMC11672324"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_dataset(package_root: Path, ground_truth_path: Path, output_root: Path) -> Path:
    manifest = read_json(package_root / "manifest.json")
    reference_manifest = read_json(package_root / "references_manifest.json")
    ground_truth_document = read_json(ground_truth_path)
    selected = ground_truth_document["minimum_reference_set"]["selected_references"]
    source_by_id = {item["package_reference_id"]: item for item in reference_manifest}

    output_root.mkdir(parents=True, exist_ok=True)
    references = []
    for number, selected_ref in enumerate(selected, 1):
        source = source_by_id[selected_ref["package_reference_id"]]
        source_dir = package_root / source["relative_path"]
        package_id = f"ref{number}_{selected_ref['package_reference_id']}_{selected_ref['pmcid']}"
        destination = output_root / "references" / package_id
        source_output = destination / "source"
        figures_output = destination / "figures"
        source_output.mkdir(parents=True, exist_ok=True)
        figures_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / "article.xml", source_output / "article.xml")
        shutil.copy2(source_dir / "full_text.txt", source_output / "full_text.txt")

        figures = []
        for asset_index, item in enumerate(read_json(source_dir / "figures_manifest.json"), 1):
            image_source = (
                source_dir / "figures" / item["image_path"]
                if item.get("image_path") else None
            )
            image_available = bool(image_source and image_source.exists())
            suffix = image_source.suffix.lower() if image_available else ".jpg"
            figure_uid = f"{package_id}__{item['figure_id']}__asset1"
            image_name = f"{figure_uid}{suffix}"
            relative_image_path = ""
            checksum = ""
            if image_available:
                shutil.copy2(image_source, figures_output / image_name)
                relative_image_path = f"references/{package_id}/figures/{image_name}"
                checksum = file_sha256(image_source)
            figures.append({
                "figure_uid": figure_uid,
                "figure_id_in_article": item["figure_id"],
                "figure_label": item["label"],
                "asset_index": 1,
                "caption": item["caption"],
                "image_path": relative_image_path,
                "image_download_status": item.get("download_status", "MISSING"),
                "source_href": item.get("source_href", ""),
                "mime_type": "image/png" if suffix == ".png" else "image/jpeg",
                "sha256": checksum,
            })
        write_json(destination / "figures_manifest.json", figures)

        references.append({
            "package_id": package_id,
            "reference_number": number,
            "ref_id": selected_ref["package_reference_id"],
            "pmcid": selected_ref["pmcid"],
            "pmid": selected_ref.get("pmid", ""),
            "doi": selected_ref.get("doi", ""),
            "title": selected_ref["title"],
            "abstract": "",
            "citation_text": selected_ref.get("citation_text", ""),
            "directly_supported_fact_ids": selected_ref["directly_supported_fact_ids"],
            "source_xml_sha256": file_sha256(source_dir / "article.xml"),
            "source_mode": "minimum_set_step1",
            "package_path": f"references/{package_id}",
            "full_text_path": f"references/{package_id}/source/full_text.txt",
            "article_xml_path": f"references/{package_id}/source/article.xml",
            "figures_manifest_path": f"references/{package_id}/figures_manifest.json",
            "figure_count": len(figures),
        })

    target = ground_truth_document["target_paper"]
    gt = ground_truth_document["ground_truth"]
    sealed_target = {
        "target_paper": target,
        "knowledge_gap": {
            "statement": gt["knowledge_gap"],
            "depends_on_fact_ids": [item["fact_id"] for item in gt["known_facts"]],
        },
        "hypothesis": {
            "statement": gt["hypothesis"],
            "status": "GROUND_TRUTH",
            "depends_on_fact_ids": [item["fact_id"] for item in gt["known_facts"]],
        },
        "access_policy": {
            "allowed_for_stage1_to_stage3": False,
            "allowed_use": "post-hoc human evaluation only",
            "reason": "Prevent target-paper answer leakage into independent hypothesis generation.",
        },
    }
    write_json(output_root / "sealed_target" / "target_hypothesis.json", sealed_target)
    dataset = {
        "schema_version": "visual-reasoning-dataset-1.0",
        "dataset_id": f"{target['pmcid']}_minimum_reference_visual_reasoning",
        "source_minimum_set_ground_truth": str(ground_truth_path.resolve()),
        "reference_count": len(references),
        "references": references,
        "sealed_target_path": "sealed_target/target_hypothesis.json",
        "leakage_firewall": {
            "stage1_allowed_inputs": ["one image", "that image caption"],
            "context_allowed_inputs": ["one reference full text", "its figure captions"],
            "stage3_allowed_inputs": ["each reference context", "its standardized observations"],
            "forbidden_for_all_generation_stages": ["sealed_target/**", "target paper results", "target paper conclusion"],
        },
    }
    dataset_path = output_root / "dataset_manifest.json"
    write_json(dataset_path, dataset)
    print(f"Imported {len(references)} minimum references for {target['pmcid']}")
    print(f"Saved dataset manifest: {dataset_path}")
    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Minimum_set output into an LLM_Reasoning dataset")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-csv", type=Path, help="Minimum_set Step1 dataset_manifest.csv")
    parser.add_argument("--package-root", type=Path, help="Minimum_set Step1 package directory")
    parser.add_argument("--ground-truth-dir", type=Path, help="Minimum_set Step2 output directory")
    parser.add_argument("--output-root", type=Path, help="Parent directory for per-PMCID datasets")
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help="Optional 1-based inclusive CSV interval; omitted means all rows",
    )
    args = parser.parse_args()
    if not args.source_csv:
        import_dataset(args.package.resolve(), args.ground_truth.resolve(), args.output.resolve())
        return
    package_root = (args.package_root or args.source_csv.parent).resolve()
    ground_truth_dir = (
        args.ground_truth_dir
        or PROJECT_ROOT / "Minimum_set" / "data" / "OA90_test" / "step2"
    ).resolve()
    output_root = (args.output_root or Path(__file__).resolve().parent / "data" / "OA90_test").resolve()
    try:
        rows = select_interval(read_csv_rows(args.source_csv), args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    registry = []
    progress = tqdm(rows, desc="LLM import datasets", unit="paper")
    for row in progress:
        package_id = row["package_id"]
        pmcid = row.get("target_pmcid", "")
        progress.set_postfix_str(pmcid or package_id)
        ground_truth = ground_truth_dir / f"{pmcid}_ground_truth.json"
        if not ground_truth.exists():
            registry.append({
                "package_id": package_id, "pmcid": pmcid, "status": "SKIPPED",
                "data_root": "", "reason": "Minimum_set Ground Truth is missing or failed",
            })
            tqdm.write(f"SKIPPED  {pmcid}  no completed Ground Truth")
            continue
        data_root = output_root / pmcid
        import_dataset(package_root / package_id, ground_truth, data_root)
        registry.append({
            "package_id": package_id, "pmcid": pmcid, "status": "READY",
            "data_root": str(data_root), "reason": "",
        })
    registry_path = output_root / "dataset_registry.csv"
    write_csv_rows(
        registry_path, registry,
        ["package_id", "pmcid", "status", "data_root", "reason"],
    )
    print(f"Saved dataset registry: {registry_path}")


if __name__ == "__main__":
    main()
