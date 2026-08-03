import pytest

from pmc_m.llm import BinaryReviewer


def valid_review(final="INCLUDE"):
    return {
        "research_question": {"result": "PASS", "reason": "The core question concerns cell-cycle control."},
        "original_experimental_research": {"result": "PASS", "reason": "The study performs original cell experiments."},
        "novel_biological_finding": {"result": "PASS", "reason": "The experiments establish a new regulatory relationship."},
        "experimental_evidence": {"result": "PASS", "reason": "Controls and validation support the main conclusion."},
        "final_decision": final,
    }


def test_valid_review_is_accepted():
    review = BinaryReviewer._validate_review(valid_review())
    assert review["final_decision"] == "INCLUDE"


def test_any_fail_requires_exclusion():
    review = valid_review(final="EXCLUDE")
    review["experimental_evidence"]["result"] = "FAIL"
    assert BinaryReviewer._validate_review(review)["final_decision"] == "EXCLUDE"


def test_inconsistent_final_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        BinaryReviewer._validate_review(valid_review(final="EXCLUDE"))


def test_json_code_fence_is_tolerated():
    import json

    value = "```json\n" + json.dumps(valid_review()) + "\n```"
    assert BinaryReviewer._decode_review_json(value)["final_decision"] == "INCLUDE"


def test_empty_model_content_is_rejected_clearly():
    with pytest.raises(ValueError, match="empty content"):
        BinaryReviewer._decode_review_json("")
