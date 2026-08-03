import pytest

import json

from mrrs.ground_truth import build_ground_truth, enumerate_subsets, select_minimum_subset, validate_assessments


FACTS = [{"fact_id": f"F{i}"} for i in range(1, 4)]


def test_enumerates_all_allowed_fact_subsets():
    subsets = enumerate_subsets(FACTS)
    assert [item["fact_ids"] for item in subsets] == [
        ["F1", "F2"], ["F1", "F3"], ["F2", "F3"], ["F1", "F2", "F3"]
    ]


def test_selects_smallest_sufficient_subset():
    selected = select_minimum_subset([
        {"subset_id": "S1", "fact_ids": ["F1", "F2", "F3"], "sufficient": True, "confidence": 1.0},
        {"subset_id": "S2", "fact_ids": ["F1", "F2"], "sufficient": True, "confidence": 0.8},
    ])
    assert selected["fact_ids"] == ["F1", "F2"]


def test_rejects_inconsistent_ablation_judgment():
    subsets = [{"subset_id": "S001", "fact_ids": ["F1", "F2"]}]
    result = {"subset_assessments": [{
        "subset_id": "S001", "fact_ids": ["F1", "F2"],
        "gap_still_defined": False, "hypothesis_answers_gap": True,
        "hypothesis_compatible_with_facts": True,
        "sufficient": True,
    }]}
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_assessments(subsets, result)


def test_builds_minimum_fact_and_reference_ground_truth(tmp_path):
    package = tmp_path / "m1_PMC1"
    package.mkdir()
    reasoning = {
        "candidate_facts": [
            {"fact_id": "F1", "statement": "A"},
            {"fact_id": "F2", "statement": "B"},
            {"fact_id": "F3", "statement": "C"},
        ],
        "knowledge_gap": "A-B relation is unknown.",
        "hypothesis": "A regulates B.",
    }
    (package / "candidate_reasoning.json").write_text(json.dumps(reasoning), encoding="utf-8")
    (package / "fact_reference_map.json").write_text(json.dumps([
        {"fact_id": "F1", "package_reference_id": "REF1"},
        {"fact_id": "F2", "package_reference_id": "REF2"},
    ]), encoding="utf-8")
    (package / "references_manifest.json").write_text(json.dumps([
        {"package_reference_id": "REF1"}, {"package_reference_id": "REF2"}
    ]), encoding="utf-8")
    (package / "manifest.json").write_text(json.dumps({
        "candidate_reasoning": "candidate_reasoning.json",
        "fact_reference_map": "fact_reference_map.json",
        "references_manifest": "references_manifest.json",
        "source_final_row": {"pmcid": "PMC1", "title": "T"},
    }), encoding="utf-8")

    class Reviewer:
        model = "test"
        def call(self, payload, cache_path, force):
            return {"subset_assessments": [
                {
                    **subset,
                    "gap_still_defined": set(subset["fact_ids"]) >= {"F1", "F2"},
                    "hypothesis_answers_gap": True,
                    "hypothesis_compatible_with_facts": True,
                    "sufficient": set(subset["fact_ids"]) >= {"F1", "F2"},
                    "reason": "test", "confidence": 1.0,
                }
                for subset in payload["subsets_to_assess"]
            ]}
        def call_bridges(self, payload, cache_path, force):
            return {
                "bridge_relations": [
                    {
                        "bridge_id": "B1", "source_fact_ids": ["F1", "F2"],
                        "relation": "A may regulate B", "status": "PARTIAL",
                        "supporting_reference_ids": ["REF1", "REF2"],
                        "established_component": "A and B are established",
                        "missing_component": "their relation is untested", "reason": "test",
                    }
                ],
                "gap_hypothesis_pairs": [
                    {
                        "pair_id": "GH1", "role": "PRIMARY", "source_bridge_ids": ["B1"],
                        "source_fact_ids": ["F1", "F2"], "gap_id": "G1",
                        "knowledge_gap": "placeholder", "hypothesis_id": "H1", "hypothesis": "placeholder",
                        "independent_variable": "A", "dependent_variable": "B", "falsification_condition": "B does not change",
                    },
                    {
                        "pair_id": "GH2", "role": "ALTERNATIVE", "source_bridge_ids": ["B1"],
                        "source_fact_ids": ["F1", "F2"], "gap_id": "G2",
                        "knowledge_gap": "Whether timing changes the A-B relation", "hypothesis_id": "H2",
                        "hypothesis": "Timing changes B after A", "independent_variable": "timing",
                        "dependent_variable": "B", "falsification_condition": "Timing has no effect",
                    },
                ],
            }

    result = build_ground_truth(package, Reviewer(), tmp_path / "out")
    assert [item["fact_id"] for item in result["ground_truth"]["known_facts"]] == ["F1", "F2"]
    assert result["minimum_reference_set"]["selected_reference_count"] == 2
    assert len(result["ground_truth"]["gap_hypothesis_pairs"]) == 2
    assert result["ground_truth"]["gap_hypothesis_pairs"][0]["knowledge_gap"] == reasoning["knowledge_gap"]
    assert result["two_level_deletion_validation"]["all_selected_facts_indispensable"]


def test_falls_back_to_all_facts_when_no_smaller_subset_passes():
    assessments = [
        {"subset_id": "S1", "fact_ids": ["F1", "F2"], "sufficient": False},
        {"subset_id": "S2", "fact_ids": ["F1", "F3"], "sufficient": False},
        {"subset_id": "S3", "fact_ids": ["F2", "F3"], "sufficient": False},
        {"subset_id": "S4", "fact_ids": ["F1", "F2", "F3"], "sufficient": False},
    ]
    selected = select_minimum_subset(assessments, ["F1", "F2", "F3"])
    assert selected["fact_ids"] == ["F1", "F2", "F3"]
    assert selected["sufficient"] is True
    assert selected["selection_rule"] == "ALL_FACTS_FALLBACK"
