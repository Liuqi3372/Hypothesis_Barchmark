from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from batch_io import parse_limit_interval, read_csv_rows, select_interval, write_csv_rows

from run_deepseek_reasoning import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    FIGURE_SCREENING_PROMPT,
    DeepSeekJSONClient,
    read_json,
    validate_screening,
    write_json,
)


def curate_reference_figures(
    data_root: Path,
    reference: dict,
    client: DeepSeekJSONClient,
    cache_root: Path,
    force: bool,
) -> dict:
    package_id = reference["package_id"]
    package_dir = data_root / reference["package_path"]
    figures = read_json(data_root / reference["figures_manifest_path"])
    payload = {
        "paper_id": package_id,
        "paper_title": reference["title"],
        "paper_abstract": reference.get("abstract", ""),
        "figures": [
            {
                "figure_uid": item["figure_uid"],
                "figure_label": item["figure_label"],
                "figure_id_in_article": item["figure_id_in_article"],
                "caption": item["caption"],
                "image_file_is_preserved_but_pixels_are_not_sent_to_deepseek": item["image_path"],
            }
            for item in figures
        ],
    }
    screening = client.call(
        FIGURE_SCREENING_PROMPT,
        payload,
        cache_root / f"{package_id}_figure_curation.json",
        force,
    )
    validate_screening(screening, figures)
    decision_by_id = {item["figure_uid"]: item for item in screening["figure_decisions"]}
    curated_dir = package_dir / "curated"
    curated_images_dir = curated_dir / "experimental_figures"
    curated_images_dir.mkdir(parents=True, exist_ok=True)

    curated_manifest = []
    excluded = []
    for figure in figures:
        decision = decision_by_id[figure["figure_uid"]]
        if decision["decision"] == "INCLUDE":
            source_image = data_root / figure["image_path"] if figure.get("image_path") else None
            image_available = bool(source_image and source_image.is_file())
            curated_image_path = ""
            if image_available:
                output_image = curated_images_dir / source_image.name
                shutil.copy2(source_image, output_image)
                curated_image_path = output_image.relative_to(data_root).as_posix()
            curated_manifest.append(
                {
                    **figure,
                    "original_image_path": figure["image_path"],
                    "image_path": curated_image_path,
                    "image_pixels_available": image_available,
                    "observation_input_mode": (
                        "LOCAL_IMAGE_PRESERVED_BUT_NOT_SENT_TO_DEEPSEEK"
                        if image_available else "CAPTION_AND_PAPER_TEXT_ONLY"
                    ),
                    "curation_decision": decision,
                }
            )
        else:
            excluded.append(
                {
                    "figure_uid": figure["figure_uid"],
                    "figure_label": figure["figure_label"],
                    "caption": figure["caption"],
                    "original_image_path": figure["image_path"],
                    "curation_decision": decision,
                }
            )

    write_json(curated_dir / "experimental_figures_manifest.json", curated_manifest)
    write_json(curated_dir / "excluded_figures.json", excluded)
    audit = {
        "schema_version": "figure-curation-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_package_id": package_id,
        "model": client.model,
        "curation_scope": "data preprocessing",
        "selection_basis": "caption and paper metadata; image pixels not sent to DeepSeek",
        "missing_image_pixels_do_not_block_text_grounded_mode": True,
        "inclusion_rule": "real experimental data plus relevance to the paper core question",
        "excluded_classes": [
            "SCHEMATIC",
            "SIMULATION",
            "NETWORK",
            "DECORATIVE_OR_IRRELEVANT",
            "UNCERTAIN",
        ],
        "original_figure_count": len(figures),
        "included_figure_count": len(curated_manifest),
        "excluded_figure_count": len(excluded),
        "included_figure_ids": [item["figure_uid"] for item in curated_manifest],
        "excluded_figure_ids": [item["figure_uid"] for item in excluded],
    }
    write_json(curated_dir / "figure_curation_audit.json", audit)
    return audit


def prepare_dataset(data_root: Path, model: str, force: bool) -> Path:
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    cache_root = data_root / ".preprocessing_cache" / "figure_curation"
    cache_root.mkdir(parents=True, exist_ok=True)
    client = DeepSeekJSONClient(model)
    audits = []
    for reference in manifest["references"]:
        print(f"[{reference['package_id']}] Curating figures inside data package", flush=True)
        audits.append(curate_reference_figures(data_root, reference, client, cache_root, force))

    curated_dataset_manifest = {
        "schema_version": "curated-visual-reasoning-dataset-1.0",
        "source_dataset_manifest": "dataset_manifest.json",
        "dataset_id": manifest["dataset_id"] + "_curated",
        "reference_count": manifest["reference_count"],
        "references": [
            {
                **reference,
                "curated_experimental_figures_manifest": (
                    Path(reference["package_path"])
                    / "curated"
                    / "experimental_figures_manifest.json"
                ).as_posix(),
                "figure_curation_audit": (
                    Path(reference["package_path"])
                    / "curated"
                    / "figure_curation_audit.json"
                ).as_posix(),
            }
            for reference in manifest["references"]
        ],
        "figure_curation_summary": audits,
        "sealed_target_path": manifest["sealed_target_path"],
        "sealed_target_allowed_for_reasoning": False,
    }
    output_path = data_root / "curated_dataset_manifest.json"
    write_json(output_path, curated_dataset_manifest)
    print(f"Saved curated dataset manifest: {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess reference figures and retain only relevant real experimental-data figures."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-csv", type=Path, help="LLM dataset_registry.csv for batch mode")
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help="Optional interval over READY registry rows; omitted means all",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.source_csv:
        prepare_dataset(args.data_root.resolve(), args.model, args.force)
    else:
        ready = [row for row in read_csv_rows(args.source_csv) if row.get("status") == "READY"]
        try:
            ready = select_interval(ready, args.limit)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        results = []
        for row in tqdm(ready, desc="LLM datasets curated", unit="paper"):
            data_root = Path(row["data_root"]).resolve()
            try:
                output = prepare_dataset(data_root, args.model, args.force)
                results.append({**row, "status": "CURATED", "curated_manifest": str(output), "error": ""})
            except Exception as exc:
                results.append({**row, "status": "ERROR", "curated_manifest": "", "error": f"{type(exc).__name__}: {exc}"})
                tqdm.write(f"ERROR {row.get('pmcid')} {type(exc).__name__}: {exc}")
        output_csv = args.source_csv.parent / "curation_registry.csv"
        write_csv_rows(
            output_csv, results,
            ["package_id", "pmcid", "status", "data_root", "curated_manifest", "error"],
        )
        print(f"Saved curation registry: {output_csv}")
