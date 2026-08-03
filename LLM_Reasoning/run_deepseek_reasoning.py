from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from batch_io import parse_limit_interval, read_csv_rows, select_interval, write_csv_rows
from typing import Any


PROJECT_ROOT = Path(r"C:/Users/sxx/Desktop/codex/barchmark-m-7.30")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "LLM_Reasoning" / "data" / "test"
DEFAULT_MODEL = "deepseek-v4-pro"

FIGURE_SCREENING_PROMPT = r"""You are a strict curator of cell-biology paper figures. You receive figure metadata and captions from one paper. Classify every figure independently using only supplied text metadata.

The downstream dataset may use only figures that contain real experimental measurements or real experimental images, such as microscopy, electron microscopy, immunofluorescence, histology, western blots, gels, flow cytometry, experimentally measured plots, or quantitative panels derived from biological experiments.

Exclude a figure when it is only one or more of the following:
- hand-drawn or conceptual illustration;
- mechanism schematic, pathway diagram, workflow, graphical abstract, or model cartoon;
- simulation-only or computational mock-up without experimental data;
- interaction network or other network visualization;
- decorative, unrelated, or insufficiently relevant image;
- review-only synthesis diagram with no original experimental data.

A composite figure may be retained when it contains at least one genuine experimental-data panel. Do not infer visual content that is not stated in the caption. If uncertain, exclude it.

Return JSON only:
{
  "paper_id": "package id",
  "figure_decisions": [
    {
      "figure_uid": "exact supplied ID",
      "classification": "EXPERIMENTAL_DATA or SCHEMATIC or SIMULATION or NETWORK or DECORATIVE_OR_IRRELEVANT or UNCERTAIN",
      "contains_real_experimental_data": true,
      "relevant_to_paper_core_question": true,
      "decision": "INCLUDE or EXCLUDE",
      "reason": "short text-grounded reason",
      "experimental_modalities": ["microscopy"]
    }
  ]
}

Include every supplied figure exactly once. INCLUDE is permitted only when both booleans are true and classification is EXPERIMENTAL_DATA."""

REFERENCE_ANALYSIS_PROMPT = r"""You are a cell-biology evidence extraction specialist. Analyze one reference paper independently. You receive its full text, a paper-level context, and only the captions of figures already accepted as genuine experimental-data figures.

Important modality limitation: you do not receive image pixels. Therefore, any figure observation must be reconstructed only from the supplied caption and the paper's original textual Results statements. Never claim to have visually inspected the image. Label every such item TEXT_DERIVED_FIGURE_EVIDENCE.

Tasks:
1. For every accepted figure, extract one or more objective paper-reported observations. Preserve groups, samples, treatments, measurements, direction of change, and stated statistical comparisons when present. Do not add outside knowledge.
2. Produce 2-6 atomic paper-local subconclusions. Each subconclusion must be supported by supplied text anchors and/or accepted figure evidence. Do not link to another paper and do not create a new hypothesis.
3. If the paper has no accepted experimental figure, derive subconclusions from verifiable paper text only and make the absence explicit.

Forbidden:
- claiming direct visual observation;
- using excluded figures;
- importing another paper;
- inventing data, mechanisms, or numerical values;
- using the target paper, target hypothesis, or sealed answer.

Return JSON only:
{
  "reference_package_id": "exact package id",
  "evidence_mode": "TEXT_ONLY_DEEPSEEK_NO_IMAGE_PIXELS",
  "accepted_figure_evidence": [
    {
      "figure_uid": "exact accepted figure ID",
      "observation_id": "O1_1",
      "observation": "caption/results-grounded observation",
      "provenance": "TEXT_DERIVED_FIGURE_EVIDENCE",
      "source_text_anchors": ["short supplied-text anchor"]
    }
  ],
  "subconclusions": [
    {
      "subconclusion_id": "P1_1",
      "statement": "one atomic paper-local established claim",
      "supporting_observation_ids": ["O1_1"],
      "supporting_figure_uids": ["figure uid"],
      "supporting_text_anchors": ["short supplied-text anchor"],
      "support_basis": "FIGURE_AND_TEXT or TEXT_ONLY"
    }
  ],
  "warnings": []
}"""

JOINT_REASONING_PROMPT = r"""You are a cell-biology hypothesis-generation researcher. Use only the supplied analyses of the minimal reference papers. Each reference provides paper-local subconclusions supported by accepted experimental-figure captions/results text or text alone.

Construct multiple cross-paper combinations and produce a set of Knowledge Gaps and a corresponding set of testable Hypotheses.

Hard rules:
1. Each reasoning combination must use at least two different reference papers.
2. Use only supplied subconclusion IDs and statements. Do not use excluded figures or external knowledge.
3. A Knowledge Gap must state one precise untested relationship not already established by any single supplied paper.
4. Each Hypothesis must be a directional, falsifiable answer to exactly one Knowledge Gap.
5. Gap and Hypothesis mapping must be one-to-one: every gap has exactly one hypothesis in the same pair; every hypothesis contains corresponding_gap_id; no orphan gaps or hypotheses.
6. Produce 2-6 Gap-Hypothesis pairs.
7. Do not read, reconstruct, quote, or assume the target paper's real hypothesis or findings.
8. Return concise scientific outputs, not hidden chain-of-thought. The reasoning path is represented only by ordered subconclusion IDs and a short evidence-combination statement.

Return JSON only:
{
  "reasoning_combinations": [
    {
      "combination_id": "C1",
      "subconclusion_ids": ["P1_1", "P2_1"],
      "reference_package_ids": ["ref1", "ref2"],
      "evidence_combination": "brief statement of what is jointly established"
    }
  ],
  "gap_hypothesis_pairs": [
    {
      "pair_id": "GH1",
      "source_combination_id": "C1",
      "gap_id": "G1",
      "knowledge_gap": "one precise untested relationship",
      "hypothesis_id": "H1",
      "corresponding_gap_id": "G1",
      "hypothesis": "one directional falsifiable answer",
      "independent_variable": "explicit variable",
      "dependent_variable": "explicit variable",
      "falsification_condition": "observation that would contradict the hypothesis"
    }
  ],
  "gap_set": [
    {"gap_id": "G1", "knowledge_gap": "same text as the paired gap", "paired_hypothesis_id": "H1"}
  ],
  "hypothesis_set": [
    {"hypothesis_id": "H1", "hypothesis": "same text as the paired hypothesis", "corresponding_gap_id": "G1"}
  ],
  "warnings": []
}"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def bounded(text: str, maximum: int) -> str:
    text = text or ""
    if len(text) <= maximum:
        return text
    half = maximum // 2
    return text[:half] + "\n[Middle omitted for API context limit]\n" + text[-half:]


def find_deepseek_key() -> str:
    value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    candidates = [
        PROJECT_ROOT / "minimum_set" / "key" / "DEEPSEEK_API_KEY.txt",
        PROJECT_ROOT / "code" / "key" / "DEEPSEEK_API_KEY.txt",
    ]
    for path in candidates:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    raise RuntimeError("DeepSeek API key was not found.")


class DeepSeekJSONClient:
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=find_deepseek_key(), base_url="https://api.deepseek.com")

    def call(self, system_prompt: str, payload: dict, cache_path: Path, force: bool) -> dict:
        if cache_path.exists() and not force:
            print(f"  Reusing cache: {cache_path.name}", flush=True)
            return read_json(cache_path)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=32768,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}},
                    response_format={"type": "json_object"},
                    stream=False,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("DeepSeek returned empty JSON content")
                result = json.loads(content)
                write_json(cache_path, result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek JSON call failed after 3 attempts: {last_error}") from last_error


def ensure_output_layout(output_root: Path) -> dict[str, Path]:
    paths = {
        "analysis": output_root / "01_reference_analysis",
        "joint": output_root / "02_joint_reasoning",
        "cache": output_root / "cache",
        "snapshot": output_root / "00_input_snapshot",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def validate_screening(result: dict, figures: list[dict]) -> None:
    decisions = result.get("figure_decisions")
    if not isinstance(decisions, list):
        raise ValueError("figure_decisions must be a list")
    expected = {item["figure_uid"] for item in figures}
    received = {item.get("figure_uid") for item in decisions}
    if expected != received:
        raise ValueError(f"Figure decision mismatch: missing={expected-received}, extra={received-expected}")
    for item in decisions:
        valid_include = (
            item.get("classification") == "EXPERIMENTAL_DATA"
            and item.get("contains_real_experimental_data") is True
            and item.get("relevant_to_paper_core_question") is True
        )
        if item.get("decision") == "INCLUDE" and not valid_include:
            raise ValueError(f"Unsafe inclusion decision for {item.get('figure_uid')}")


def normalize_reference_analysis(result: dict, reference_number: int) -> dict:
    """Flatten model variants and assign globally unique, deterministic O/P IDs."""
    normalized = json.loads(json.dumps(result))
    flat_observations: list[dict] = []
    old_to_new_observation_ids: dict[str, list[str]] = {}

    for figure_item in normalized.get("accepted_figure_evidence", []):
        figure_uid = figure_item.get("figure_uid")
        nested = figure_item.get("observations")
        source_items = nested if isinstance(nested, list) else [figure_item]
        for source in source_items:
            item = dict(source)
            item["figure_uid"] = item.get("figure_uid") or figure_uid
            old_id = item.get("observation_id")
            new_id = f"O{reference_number}_{len(flat_observations) + 1}"
            item["observation_id"] = new_id
            item["provenance"] = item.get("provenance") or figure_item.get("provenance")
            flat_observations.append(item)
            if old_id:
                old_to_new_observation_ids.setdefault(old_id, []).append(new_id)
    normalized["accepted_figure_evidence"] = flat_observations

    for index, subconclusion in enumerate(normalized.get("subconclusions", []), start=1):
        subconclusion["subconclusion_id"] = f"P{reference_number}_{index}"
        support_ids = subconclusion.pop(
            "support_observation_ids",
            subconclusion.get("supporting_observation_ids", []),
        )
        remapped: list[str] = []
        for old_id in support_ids if isinstance(support_ids, list) else []:
            remapped.extend(old_to_new_observation_ids.get(old_id, [old_id]))
        subconclusion["supporting_observation_ids"] = list(dict.fromkeys(remapped))
    return normalized


def validate_reference_analysis(result: dict, package_id: str, included_ids: set[str]) -> None:
    if result.get("reference_package_id") != package_id:
        raise ValueError(f"Reference package mismatch for {package_id}")
    if result.get("evidence_mode") != "TEXT_ONLY_DEEPSEEK_NO_IMAGE_PIXELS":
        raise ValueError("The result must disclose the text-only DeepSeek evidence mode")
    observations = result.get("accepted_figure_evidence", [])
    observation_ids = set()
    for item in observations:
        if item.get("figure_uid") not in included_ids:
            raise ValueError(f"Analysis used a non-included figure: {item.get('figure_uid')}")
        if item.get("provenance") != "TEXT_DERIVED_FIGURE_EVIDENCE":
            raise ValueError("Every figure observation must disclose text-derived provenance")
        observation_ids.add(item.get("observation_id"))
    subconclusions = result.get("subconclusions")
    if not isinstance(subconclusions, list) or not 2 <= len(subconclusions) <= 6:
        raise ValueError(f"{package_id} must contain 2-6 subconclusions")
    for item in subconclusions:
        support_ids = set(item.get("supporting_observation_ids", []))
        if support_ids and not support_ids <= observation_ids:
            raise ValueError(f"Unknown observation used by {item.get('subconclusion_id')}")


def validate_joint_result(result: dict, analyses: list[dict]) -> None:
    known_subconclusions = {
        item["subconclusion_id"]
        for analysis in analyses
        for item in analysis["subconclusions"]
    }
    known_packages = {analysis["reference_package_id"] for analysis in analyses}
    combinations = result.get("reasoning_combinations", [])
    pairs = result.get("gap_hypothesis_pairs", [])
    gap_set = result.get("gap_set", [])
    hypothesis_set = result.get("hypothesis_set", [])
    if not 2 <= len(pairs) <= 6:
        raise ValueError("The final result must contain 2-6 Gap-Hypothesis pairs")
    combination_ids = set()
    for combo in combinations:
        combination_ids.add(combo.get("combination_id"))
        ids = set(combo.get("subconclusion_ids", []))
        packages = set(combo.get("reference_package_ids", []))
        if not ids or not ids <= known_subconclusions:
            raise ValueError(f"Unknown subconclusion in {combo.get('combination_id')}")
        if len(packages) < 2 or not packages <= known_packages:
            raise ValueError(f"Every combination must use at least two known papers: {combo.get('combination_id')}")
    gap_ids, hypothesis_ids = set(), set()
    for pair in pairs:
        gap_id = pair.get("gap_id")
        hypothesis_id = pair.get("hypothesis_id")
        if pair.get("source_combination_id") not in combination_ids:
            raise ValueError(f"Unknown source combination for {pair.get('pair_id')}")
        if pair.get("corresponding_gap_id") != gap_id:
            raise ValueError(f"Gap-Hypothesis mismatch in {pair.get('pair_id')}")
        if gap_id in gap_ids or hypothesis_id in hypothesis_ids:
            raise ValueError("Gap IDs and Hypothesis IDs must be unique")
        if not pair.get("knowledge_gap") or not pair.get("hypothesis"):
            raise ValueError(f"Incomplete pair: {pair.get('pair_id')}")
        gap_ids.add(gap_id)
        hypothesis_ids.add(hypothesis_id)
    if {item.get("gap_id") for item in gap_set} != gap_ids:
        raise ValueError("gap_set does not match gap_hypothesis_pairs")
    if {item.get("hypothesis_id") for item in hypothesis_set} != hypothesis_ids:
        raise ValueError("hypothesis_set does not match gap_hypothesis_pairs")
    for item in gap_set:
        if item.get("paired_hypothesis_id") not in hypothesis_ids:
            raise ValueError(f"Orphan gap: {item.get('gap_id')}")
    for item in hypothesis_set:
        if item.get("corresponding_gap_id") not in gap_ids:
            raise ValueError(f"Orphan hypothesis: {item.get('hypothesis_id')}")


def run_pipeline(data_root: Path, output_root: Path, model: str, force: bool) -> Path:
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    paths = ensure_output_layout(output_root)
    shutil.copy2(manifest_path, paths["snapshot"] / "dataset_manifest.json")
    client = DeepSeekJSONClient(model)

    curation_audits: list[dict] = []
    reference_analyses: list[dict] = []
    for reference in manifest["references"]:
        package_id = reference["package_id"]
        curated_dir = data_root / reference["package_path"] / "curated"
        curated_manifest_path = curated_dir / "experimental_figures_manifest.json"
        curation_audit_path = curated_dir / "figure_curation_audit.json"
        if not curated_manifest_path.exists() or not curation_audit_path.exists():
            raise FileNotFoundError(
                f"Missing curated figure data for {package_id}. Run prepare_reasoning_dataset.py first."
            )
        included_figures = read_json(curated_manifest_path)
        curation_audit = read_json(curation_audit_path)
        curation_audits.append(curation_audit)
        included_ids = {item["figure_uid"] for item in included_figures}
        print(
            f"[{package_id}] Using curated experimental figures: "
            f"{len(included_figures)}/{curation_audit['original_figure_count']}",
            flush=True,
        )

        article_text = (data_root / reference["full_text_path"]).read_text(encoding="utf-8")
        context_path = data_root / reference["package_path"] / "derived" / "context.json"
        context = read_json(context_path) if context_path.exists() else {}
        analysis_payload = {
            "reference_package_id": package_id,
            "paper": {
                "title": reference["title"],
                "abstract": reference.get("abstract", ""),
                "full_text": bounded(article_text, 90000),
                "existing_global_context": context,
            },
            "accepted_experimental_figures": [
                {
                    "figure_uid": item["figure_uid"],
                    "figure_label": item["figure_label"],
                    "figure_id_in_article": item["figure_id_in_article"],
                    "caption": item["caption"],
                    "screening_reason": item["curation_decision"]["reason"],
                    "image_path_for_future_vision_model": item["image_path"],
                }
                for item in included_figures
            ],
            "excluded_figure_ids": curation_audit["excluded_figure_ids"],
        }
        print(f"[{package_id}] Text-grounded observations and subconclusions", flush=True)
        analysis = client.call(
            REFERENCE_ANALYSIS_PROMPT,
            analysis_payload,
            paths["cache"] / f"{package_id}_reference_analysis.json",
            force,
        )
        analysis = normalize_reference_analysis(analysis, reference["reference_number"])
        validate_reference_analysis(analysis, package_id, included_ids)
        write_json(paths["analysis"] / f"{package_id}.json", analysis)
        reference_analyses.append(analysis)

    joint_payload = {
        "task": "Combine paper-local subconclusions into paired Knowledge Gaps and Hypotheses.",
        "modality_disclosure": (
            "DeepSeek is text-only. Figure evidence was derived from captions and original paper text; "
            "image pixels were preserved locally but not supplied to the model."
        ),
        "reference_analyses": reference_analyses,
    }
    print("[Joint] Building Gap-Hypothesis pairs", flush=True)
    joint = client.call(
        JOINT_REASONING_PROMPT,
        joint_payload,
        paths["cache"] / "joint_reasoning.json",
        force,
    )
    validate_joint_result(joint, reference_analyses)
    write_json(paths["joint"] / "joint_reasoning.json", joint)

    final = {
        "schema_version": "deepseek-text-grounded-reasoning-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "status": "COMPLETE",
        "input_dataset": manifest["dataset_id"],
        "modality_and_scope": {
            "deepseek_api_accepts_image_pixels": False,
            "image_files_preserved_locally": True,
            "figure_observation_provenance": "TEXT_DERIVED_FIGURE_EVIDENCE",
            "excluded_figure_classes": [
                "SCHEMATIC", "SIMULATION", "NETWORK", "DECORATIVE_OR_IRRELEVANT", "UNCERTAIN"
            ],
            "sealed_target_used": False,
        },
        "data_preprocessing": {
            "figure_curation_location": "references/<package_id>/curated/",
            "curation_audits": curation_audits,
        },
        "reference_analyses": reference_analyses,
        "joint_reasoning": joint,
        "counts": {
            "reference_count": len(reference_analyses),
            "total_figures": sum(item["original_figure_count"] for item in curation_audits),
            "included_experimental_figures": sum(item["included_figure_count"] for item in curation_audits),
            "subconclusion_count": sum(len(item["subconclusions"]) for item in reference_analyses),
            "gap_hypothesis_pair_count": len(joint["gap_hypothesis_pairs"]),
        },
        "audit_warning": (
            "These are structured LLM judgments based on paper text and captions, not direct visual observations. "
            "Replace TEXT_DERIVED_FIGURE_EVIDENCE with outputs from a real multimodal model when available."
        ),
    }
    final_path = output_root / "final_reasoning.json"
    write_json(final_path, final)
    print(f"Saved final result: {final_path}", flush=True)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build text-grounded figure evidence, subconclusions, Gap set, and paired Hypothesis set with DeepSeek."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-csv", type=Path, help="curation_registry.csv for batch mode")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit", type=parse_limit_interval, metavar="N|START-END",
        help="Optional interval over CURATED registry rows; omitted means all",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Ignore cached API responses")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.source_csv:
        output_root = args.output_root or args.data_root / "reasoning_runs" / "deepseek_text_v1"
        run_pipeline(args.data_root.resolve(), output_root.resolve(), args.model, args.force)
    else:
        curated = [row for row in read_csv_rows(args.source_csv) if row.get("status") == "CURATED"]
        try:
            curated = select_interval(curated, args.limit)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        results = []
        for row in tqdm(curated, desc="LLM reasoning complete", unit="paper"):
            data_root = Path(row["data_root"]).resolve()
            output_root = (
                args.output_root / row["pmcid"] / "deepseek_text_v1"
                if args.output_root else data_root / "reasoning_runs" / "deepseek_text_v1"
            )
            try:
                final_path = run_pipeline(data_root, output_root.resolve(), args.model, args.force)
                results.append({**row, "status": "COMPLETE", "final_result": str(final_path), "error": ""})
            except Exception as exc:
                results.append({**row, "status": "ERROR", "final_result": "", "error": f"{type(exc).__name__}: {exc}"})
                tqdm.write(f"ERROR {row.get('pmcid')} {type(exc).__name__}: {exc}")
        output_csv = args.source_csv.parent / "reasoning_registry.csv"
        write_csv_rows(
            output_csv, results,
            ["package_id", "pmcid", "status", "data_root", "final_result", "error"],
        )
        print(f"Saved reasoning registry: {output_csv}")
