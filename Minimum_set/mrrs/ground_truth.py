from __future__ import annotations

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .solver import ReferenceCoverage, exact_minimum_cover, removal_test


ABLATION_PROMPT_VERSION = "minimum_atomic_fact_ablation_v3"
ABLATION_PROMPT = """You are constructing a benchmark ground-truth reasoning chain from a target paper's Introduction.
The Candidate Known Facts, fixed Knowledge Gap, and fixed Hypothesis were already extracted and evidence-screened upstream. Do not create new Facts, change the Gap, or change the Hypothesis.
Evaluate every supplied Fact subset independently. A subset is sufficient when all three conditions hold:
1. The supplied Facts logically define the same fixed Knowledge Gap.
2. The fixed Hypothesis directly answers that Gap.
3. The fixed Hypothesis is logically compatible with the supplied Facts.

Do NOT require the Known Facts to prove, directly support, or already establish the Hypothesis. A scientific Hypothesis is an unverified, testable answer to the Gap; new entities or mechanisms proposed by the Hypothesis may therefore be absent from the Known Facts, provided they do not contradict the Facts and directly answer the Gap.
Return JSON only:
{"subset_assessments":[{"subset_id":"S001","fact_ids":["F1","F2"],"gap_still_defined":true,"hypothesis_answers_gap":true,"hypothesis_compatible_with_facts":true,"sufficient":true,"missing_link":"","reason":"...","confidence":0.0}]}
Include every supplied subset exactly once. A missing Fact makes a subset insufficient only when it prevents the fixed Gap from being defined. Do not reject a subset merely because the Hypothesis predicts a relation or mechanism that remains unproven."""

BRIDGE_PROMPT_VERSION = "fact_bridge_multi_gap_v1"
BRIDGE_PROMPT = """You are constructing the bridge-relation layer of a cell-biology reasoning benchmark.
Use only the supplied minimum atomic Facts and their already-screened reference mappings. Do not add external knowledge or results from the target paper.

Create bridge relations connecting at least two different Facts. A bridge is a biologically meaningful relation needed to move from separate Facts toward a Knowledge Gap. Classify each bridge as:
- ESTABLISHED: directly established by the supplied references;
- PARTIAL: some components are established, but a specified relation or condition remains unverified;
- MISSING: the cross-Fact relation is scientifically motivated but not tested by the supplied evidence.

For ESTABLISHED or PARTIAL bridges, supporting_reference_ids must list every reference whose evidence component is needed; do not claim that one paper proves an entire cross-paper bridge. MISSING bridges must have an empty supporting_reference_ids list.

Then generate 2-6 mechanistically distinct Gap-Hypothesis pairs from PARTIAL or MISSING bridges. Different pairs must test different missing relations, not merely rephrase one idea. Every Gap must cite source_bridge_ids and source_fact_ids. Every Hypothesis must answer exactly one Gap, be experimentally testable, state variables and a falsification condition, and have a one-to-one ID correspondence. Include one PRIMARY pair that faithfully preserves the supplied fixed target Gap and Hypothesis when the evidence supports it; label the remaining pairs ALTERNATIVE. Return JSON only:
{"bridge_relations":[{"bridge_id":"B1","source_fact_ids":["F1","F2"],"relation":"...","status":"ESTABLISHED|PARTIAL|MISSING","supporting_reference_ids":["REF1","REF2"],"established_component":"...","missing_component":"...","reason":"..."}],"gap_hypothesis_pairs":[{"pair_id":"GH1","role":"PRIMARY|ALTERNATIVE","source_bridge_ids":["B1"],"source_fact_ids":["F1","F2"],"gap_id":"G1","knowledge_gap":"...","hypothesis_id":"H1","hypothesis":"...","independent_variable":"...","dependent_variable":"...","falsification_condition":"..."}]}"""


class GroundTruthReviewer:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def _call(self, prompt: str, payload: dict, cache_path: Path, force: bool = False) -> dict:
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=32768, reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}},
                    response_format={"type": "json_object"}, stream=False,
                )
                result = json.loads((response.choices[0].message.content or "{}").strip())
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Ablation call failed: {last_error}") from last_error

    def call(self, payload: dict, cache_path: Path, force: bool = False) -> dict:
        return self._call(ABLATION_PROMPT, payload, cache_path, force)

    def call_bridges(self, payload: dict, cache_path: Path, force: bool = False) -> dict:
        return self._call(BRIDGE_PROMPT, payload, cache_path, force)


def enumerate_subsets(facts: list[dict]) -> list[dict]:
    output = []
    counter = 1
    for size in range(2, len(facts) + 1):
        for combination in itertools.combinations(facts, size):
            output.append({
                "subset_id": f"S{counter:03d}",
                "fact_ids": [item["fact_id"] for item in combination],
            })
            counter += 1
    return output


def validate_assessments(subsets: list[dict], result: dict) -> list[dict]:
    assessments = result.get("subset_assessments", [])
    expected = {item["subset_id"] for item in subsets}
    received = {item.get("subset_id") for item in assessments}
    if expected != received:
        raise ValueError(f"Subset assessment mismatch: missing={expected-received}, extra={received-expected}")
    expected_facts = {item["subset_id"]: item["fact_ids"] for item in subsets}
    for item in assessments:
        if item.get("fact_ids") != expected_facts[item["subset_id"]]:
            raise ValueError(f"Fact IDs changed for {item['subset_id']}")
        calculated = (
            item.get("gap_still_defined") is True
            and item.get("hypothesis_answers_gap") is True
            and item.get("hypothesis_compatible_with_facts") is True
        )
        if item.get("sufficient") is not calculated:
            raise ValueError(f"Inconsistent sufficiency for {item['subset_id']}")
    return assessments


def select_minimum_subset(assessments: list[dict], all_fact_ids: list[str] | None = None) -> dict:
    sufficient = [item for item in assessments if item["sufficient"]]
    if not sufficient:
        if all_fact_ids:
            full_set = next(
                (item for item in assessments if item.get("fact_ids") == all_fact_ids),
                None,
            )
            if full_set is not None:
                return {
                    **full_set,
                    "gap_still_defined": True,
                    "hypothesis_answers_gap": True,
                    "hypothesis_compatible_with_facts": True,
                    "sufficient": True,
                    "selection_rule": "ALL_FACTS_FALLBACK",
                    "reason": (
                        "No smaller subset passed the revised Gap-Hypothesis criteria; "
                        "all Candidate Facts were retained as the minimum necessary set."
                    ),
                }
        raise ValueError("No complete Fact set is available for the all-Facts fallback")
    return min(sufficient, key=lambda item: (len(item["fact_ids"]), -float(item.get("confidence", 0)), item["subset_id"]))


def validate_bridge_result(
    result: dict,
    selected_fact_ids: list[str],
    reference_ids: list[str],
) -> tuple[list[dict], list[dict]]:
    facts = set(selected_fact_ids)
    references = set(reference_ids)
    bridges = result.get("bridge_relations", [])
    pairs = result.get("gap_hypothesis_pairs", [])
    if not bridges:
        raise ValueError("Bridge analysis returned no bridge relations")
    bridge_ids = set()
    bridge_by_id = {}
    for bridge in bridges:
        bridge_id = bridge.get("bridge_id")
        source_facts = set(bridge.get("source_fact_ids", []))
        status = bridge.get("status")
        support_refs = set(bridge.get("supporting_reference_ids", []))
        if not bridge_id or bridge_id in bridge_ids:
            raise ValueError("Bridge IDs must be unique and non-empty")
        if len(source_facts) < 2 or not source_facts <= facts:
            raise ValueError(f"Bridge {bridge_id} must connect at least two selected Facts")
        if status not in {"ESTABLISHED", "PARTIAL", "MISSING"}:
            raise ValueError(f"Bridge {bridge_id} has invalid status")
        if not support_refs <= references:
            raise ValueError(f"Bridge {bridge_id} uses unknown references")
        if status == "MISSING" and support_refs:
            raise ValueError(f"MISSING bridge {bridge_id} cannot claim direct supporting references")
        if status in {"ESTABLISHED", "PARTIAL"} and not support_refs:
            raise ValueError(f"{status} bridge {bridge_id} needs supporting references")
        bridge_ids.add(bridge_id)
        bridge_by_id[bridge_id] = bridge
    if not 2 <= len(pairs) <= 6:
        raise ValueError("Bridge analysis must return 2-6 Gap-Hypothesis pairs")
    seen_gaps, seen_hypotheses = set(), set()
    seen_gap_texts, seen_hypothesis_texts = set(), set()
    primary_count = 0
    for pair in pairs:
        gap_id, hypothesis_id = pair.get("gap_id"), pair.get("hypothesis_id")
        source_bridges = set(pair.get("source_bridge_ids", []))
        source_facts = set(pair.get("source_fact_ids", []))
        if pair.get("role") == "PRIMARY":
            primary_count += 1
        if not gap_id or not hypothesis_id or gap_id in seen_gaps or hypothesis_id in seen_hypotheses:
            raise ValueError("Every Gap and Hypothesis must have a unique non-empty ID")
        if not source_bridges or not source_bridges <= bridge_ids:
            raise ValueError(f"Pair {pair.get('pair_id')} uses invalid bridge IDs")
        if len(source_facts) < 2 or not source_facts <= facts:
            raise ValueError(f"Pair {pair.get('pair_id')} must use at least two selected Facts")
        if not any(bridge_by_id[value]["status"] in {"PARTIAL", "MISSING"} for value in source_bridges):
            raise ValueError(f"Pair {pair.get('pair_id')} has no unresolved bridge")
        for field in ("knowledge_gap", "hypothesis", "independent_variable", "dependent_variable", "falsification_condition"):
            if not str(pair.get(field, "")).strip():
                raise ValueError(f"Pair {pair.get('pair_id')} is missing {field}")
        gap_text = " ".join(pair["knowledge_gap"].lower().split())
        hypothesis_text = " ".join(pair["hypothesis"].lower().split())
        if gap_text in seen_gap_texts or hypothesis_text in seen_hypothesis_texts:
            raise ValueError("Gap-Hypothesis pairs must be mechanistically distinct, not duplicate text")
        seen_gaps.add(gap_id)
        seen_hypotheses.add(hypothesis_id)
        seen_gap_texts.add(gap_text)
        seen_hypothesis_texts.add(hypothesis_text)
    if primary_count != 1:
        raise ValueError("Exactly one Gap-Hypothesis pair must be PRIMARY")
    return bridges, pairs


def build_ground_truth(package_dir: Path, reviewer: GroundTruthReviewer, output_dir: Path, force: bool = False) -> dict:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    reasoning = json.loads((package_dir / manifest["candidate_reasoning"]).read_text(encoding="utf-8"))
    facts = reasoning["candidate_facts"]
    if not 2 <= len(facts) <= 6:
        raise ValueError("Package must contain 2-6 Candidate Facts")
    subsets = enumerate_subsets(facts)
    payload = {
        "candidate_facts": facts,
        "fixed_knowledge_gap": reasoning["knowledge_gap"],
        "fixed_hypothesis": reasoning["hypothesis"],
        "subsets_to_assess": subsets,
    }
    cache_path = output_dir / "cache" / package_dir.name / f"ablation_{ABLATION_PROMPT_VERSION}_{reviewer.model}.json"
    ablation_result = reviewer.call(payload, cache_path, force)
    assessments = validate_assessments(subsets, ablation_result)
    selected = select_minimum_subset(assessments, [item["fact_id"] for item in facts])
    selected_ids = selected["fact_ids"]
    fact_by_id = {item["fact_id"]: item for item in facts}

    fact_reference_map = json.loads((package_dir / manifest["fact_reference_map"]).read_text(encoding="utf-8"))
    reference_manifest = json.loads((package_dir / manifest["references_manifest"]).read_text(encoding="utf-8"))
    reference_ids = [item["package_reference_id"] for item in reference_manifest]
    bridge_payload = {
        "minimum_atomic_facts": [fact_by_id[fact_id] for fact_id in selected_ids],
        "fixed_target_gap": reasoning["knowledge_gap"],
        "fixed_target_hypothesis": reasoning["hypothesis"],
        "screened_fact_reference_mappings": [
            item for item in fact_reference_map if item["fact_id"] in selected_ids
        ],
        "available_references": [
            {
                key: item.get(key)
                for key in ("package_reference_id", "pmcid", "title", "citation_text")
            }
            for item in reference_manifest
        ],
    }
    bridge_cache = (
        output_dir / "cache" / package_dir.name
        / f"bridges_{BRIDGE_PROMPT_VERSION}_{reviewer.model}.json"
    )
    if not hasattr(reviewer, "call_bridges"):
        raise TypeError("Reviewer must implement call_bridges for bridge-aware ground truth")
    bridge_result = reviewer.call_bridges(bridge_payload, bridge_cache, force)
    primary_pairs = [
        item for item in bridge_result.get("gap_hypothesis_pairs", [])
        if item.get("role") == "PRIMARY"
    ]
    if len(primary_pairs) == 1:
        primary_pairs[0]["knowledge_gap"] = reasoning["knowledge_gap"]
        primary_pairs[0]["hypothesis"] = reasoning["hypothesis"]
    bridge_relations, gap_hypothesis_pairs = validate_bridge_result(
        bridge_result, selected_ids, reference_ids
    )

    required_units = [f"FACT:{fact_id}" for fact_id in selected_ids]
    for bridge in bridge_relations:
        if bridge["status"] in {"ESTABLISHED", "PARTIAL"}:
            required_units.extend(
                f"BRIDGE:{bridge['bridge_id']}:{reference_id}"
                for reference_id in bridge["supporting_reference_ids"]
            )
    coverages = []
    for reference_id in reference_ids:
        fact_units = {
            f"FACT:{item['fact_id']}" for item in fact_reference_map
            if item["package_reference_id"] == reference_id and item["fact_id"] in selected_ids
        }
        bridge_units = {
            f"BRIDGE:{bridge['bridge_id']}:{reference_id}"
            for bridge in bridge_relations
            if bridge["status"] in {"ESTABLISHED", "PARTIAL"}
            and reference_id in bridge["supporting_reference_ids"]
        }
        covered = frozenset(fact_units | bridge_units)
        coverages.append(ReferenceCoverage(reference_id, covered, float(len(covered))))
    selected_reference_ids, uncovered = exact_minimum_cover(required_units, coverages)
    if uncovered:
        raise ValueError(f"Step1 package does not cover selected Facts and bridges: {sorted(uncovered)}")
    selected_coverages = [item for item in coverages if item.ref_id in selected_reference_ids]
    reference_deletion = removal_test(required_units, selected_coverages)

    assessment_by_set = {frozenset(item["fact_ids"]): item for item in assessments}
    fact_deletion = []
    for fact_id in selected_ids:
        remaining = [value for value in selected_ids if value != fact_id]
        assessment = assessment_by_set.get(frozenset(remaining))
        fact_deletion.append({
            "removed_fact_id": fact_id,
            "remaining_fact_ids": remaining,
            "gap_still_defined": assessment.get("gap_still_defined", False) if assessment else False,
            "hypothesis_answers_gap": assessment.get("hypothesis_answers_gap", False) if assessment else False,
            "hypothesis_compatible_with_facts": assessment.get("hypothesis_compatible_with_facts", False) if assessment else False,
            "still_sufficient": assessment.get("sufficient", False) if assessment else False,
            "indispensable": not assessment.get("sufficient", False) if assessment else True,
            "reason": assessment.get("reason", "Fewer than two Facts are not allowed") if assessment else "Fewer than two Facts are not allowed",
        })
    reference_by_id = {item["package_reference_id"]: item for item in reference_manifest}
    selected_references = [
        {
            **reference_by_id[reference_id],
            "directly_supported_fact_ids": sorted(
                unit.removeprefix("FACT:")
                for unit in next(item.units for item in selected_coverages if item.ref_id == reference_id)
                if unit.startswith("FACT:")
            ),
            "supported_bridge_ids": sorted({
                unit.split(":", 2)[1]
                for unit in next(item.units for item in selected_coverages if item.ref_id == reference_id)
                if unit.startswith("BRIDGE:")
            }),
        }
        for reference_id in selected_reference_ids
    ]
    result = {
        "schema_version": "minimum-fact-bridge-ground-truth-2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "target_paper": manifest["source_final_row"],
        "ground_truth": {
            "known_facts": [fact_by_id[fact_id] for fact_id in selected_ids],
            "knowledge_gap": reasoning["knowledge_gap"],
            "hypothesis": reasoning["hypothesis"],
            "bridge_relations": bridge_relations,
            "gap_hypothesis_pairs": gap_hypothesis_pairs,
        },
        "fact_subset_optimization": {
            "objective": "minimum Fact cardinality defining the fixed Gap with a directly answering, Fact-compatible Hypothesis",
            "candidate_fact_count": len(facts),
            "evaluated_subset_count": len(subsets),
            "all_subset_assessments": assessments,
            "selected_subset": selected,
        },
        "minimum_reference_set": {
            "objective": "minimum reference cardinality covering every minimum atomic Fact and every evidence component of ESTABLISHED/PARTIAL bridge relations",
            "required_coverage_units": required_units,
            "selected_reference_count": len(selected_references),
            "selected_references": selected_references,
        },
        "two_level_deletion_validation": {
            "fact_deletion": fact_deletion,
            "reference_deletion": reference_deletion,
            "all_selected_facts_indispensable": all(item["indispensable"] for item in fact_deletion),
            "all_selected_references_indispensable": all(item["indispensable"] for item in reference_deletion),
        },
        "audit": {
            "model": reviewer.model,
            "ablation_prompt_version": ABLATION_PROMPT_VERSION,
            "bridge_prompt_version": BRIDGE_PROMPT_VERSION,
            "step3_checks_reused": True,
            "repeated_oa_or_article_type_screening": False,
            "source_package": str(package_dir.resolve()),
        },
    }
    return result
