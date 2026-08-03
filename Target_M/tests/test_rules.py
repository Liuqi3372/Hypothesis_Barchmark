from pmc_m.rules import choose_category, hard_exclusion


def test_empty_abstract_excluded():
    assert hard_exclusion("Original study", "", ["research-article"]) == "E_EMPTY_OR_INSUFFICIENT_ABSTRACT"


def test_non_open_access_excluded_first():
    assert hard_exclusion("Original study", "Experimental result " * 20, ["research-article"], False) == "E_NOT_PMC_OPEN_ACCESS"


def test_review_excluded_by_type():
    abstract = "We systematically summarize " + "prior work " * 20
    assert hard_exclusion("Something", abstract, ["Review"]) == "E_EXCLUDED_ARTICLE_TYPE"


def test_review_excluded_by_title():
    abstract = "Experimental details and results " * 10
    assert hard_exclusion("A systematic review of signaling", abstract, ["article"]) == "E_EXCLUDED_TITLE_PATTERN"


def test_preprint_retraction_and_on_hold_are_excluded():
    abstract = "Experimental details and original biological results " * 10
    assert hard_exclusion("Original study", abstract, ["preprint"]) == "E_PREPRINT"
    assert hard_exclusion("Retracted: Original study", abstract, ["article"]) == "E_RETRACTED_OR_WITHDRAWN"
    assert hard_exclusion("Original study", abstract, ["article on hold"]) == "E_ON_HOLD"
    assert hard_exclusion("Original study", abstract, ["article"], journal="Journal On Hold") == "E_ON_HOLD"


def test_category_choice():
    category = choose_category(
        "Mitophagy controls lysosome function",
        "We measured autophagic flux and mitochondrial turnover in cells.",
        {"Autophagy, cell death, and quality control"},
    )
    assert category == "Autophagy, cell death, and quality control"


def test_all_category_queries_use_controlled_and_text_terms():
    from pmc_m.rules import CATEGORIES

    assert len(CATEGORIES) == 9
    for category in CATEGORIES:
        assert category.mesh_terms
        assert category.text_terms
        assert "[mh]" in category.query
        assert "[tiab]" in category.query
        assert category.basis_ids
