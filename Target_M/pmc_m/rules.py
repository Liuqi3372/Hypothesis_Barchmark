from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    name: str
    definition: str
    mesh_terms: tuple[str, ...]
    text_terms: tuple[str, ...]
    keywords: tuple[str, ...]
    basis_ids: tuple[str, ...]

    @property
    def query(self) -> str:
        # MeSH保证概念规范，题名/摘要词保证尚未完成MeSH标引的新论文也能被检出。
        mesh = [f'"{term}"[mh]' for term in self.mesh_terms]
        text = [
            f"{term}[tiab]" if "*" in term else f'"{term}"[tiab]'
            for term in self.text_terms
        ]
        return "(" + " OR ".join(mesh + text) + ")"


# Classification basis:
# - ASCB_EMBO_2026: current professional-society scientific topic taxonomy.
# - NLM_MESH_G04: NLM controlled vocabulary hierarchy for cell physiological phenomena.
# - ALBERTS_MBOC4: canonical cell-biology textbook chapter structure.
CATEGORIES: tuple[Category, ...] = (
    Category(
        name="Membranes, organelles, and intracellular trafficking",
        definition=(
            "Cellular membranes, organelle organization and dynamics, membrane contact sites, "
            "protein sorting, endocytosis, exocytosis, secretion, and vesicular transport."
        ),
        mesh_terms=(
            "Cell Compartmentation", "Organelles", "Organelle Biogenesis",
            "Endocytosis", "Exocytosis", "Secretory Pathway",
        ),
        text_terms=(
            "organelle", "organelle dynamics", "membrane contact site", "vesicle trafficking",
            "protein sorting", "endocytosis", "exocytosis", "secretory pathway",
            "mitochondri*", "lysosom*", "endosom*", "peroxisom*",
            "endoplasmic reticulum", "Golgi apparatus",
        ),
        keywords=(
            "organelle", "membrane contact", "vesicle trafficking", "protein sorting",
            "endocyt", "exocyt", "secretory pathway", "mitochond", "lysosom",
            "endosom", "peroxisom", "endoplasmic reticulum", "golgi",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Cytoskeleton, polarity, and cell motility",
        definition=(
            "Actin, microtubules, intermediate filaments, molecular motors, cell polarity, "
            "cilia, flagella, and the cell-intrinsic machinery that generates movement."
        ),
        mesh_terms=(
            "Cytoskeleton", "Actin Cytoskeleton", "Microtubules", "Molecular Motor Proteins",
            "Cell Polarity", "Cell Movement",
        ),
        text_terms=(
            "cytoskeleton", "actin cytoskeleton", "microtubule dynamics",
            "intermediate filament", "molecular motor", "cell polarity",
            "cell motility", "cilia", "flagella", "lamellipodia", "filopodia",
        ),
        keywords=(
            "cytoskeleton", "actin", "microtubul", "intermediate filament",
            "molecular motor", "cell polarity", "cell motility", "cilia",
            "flagell", "lamellipod", "filopod",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Cell adhesion, migration, and mechanobiology",
        definition=(
            "Cell-cell and cell-matrix adhesion, extracellular matrix, directed migration, "
            "invasion, mechanotransduction, and physical control of cellular behavior."
        ),
        mesh_terms=(
            "Cell Adhesion", "Cell Movement", "Chemotaxis", "Extracellular Matrix",
            "Mechanotransduction, Cellular",
        ),
        text_terms=(
            "cell adhesion", "cell migration", "focal adhesion", "cell-cell adhesion",
            "cell-matrix adhesion", "extracellular matrix", "chemotaxis",
            "mechanotransduction", "mechanobiology", "collective migration",
            "epithelial-mesenchymal transition",
        ),
        keywords=(
            "cell adhesion", "cell migration", "focal adhesion", "cell-cell adhesion",
            "cell-matrix", "extracellular matrix", "chemotaxis", "mechanotransduction",
            "mechanobiolog", "collective migration", "epithelial-mesenchymal",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Signal transduction and cell communication",
        definition=(
            "Receptor activation, intracellular signaling networks, second messengers, "
            "signal integration, and communication between cells."
        ),
        mesh_terms=(
            "Signal Transduction", "Cell Communication", "Receptors, Cell Surface",
            "Second Messenger Systems", "MAP Kinase Signaling System",
        ),
        text_terms=(
            "signal transduction", "cell signaling", "signaling pathway",
            "receptor activation", "second messenger", "kinase cascade",
            "MAPK signaling", "Wnt signaling", "Hippo signaling", "Notch signaling",
        ),
        keywords=(
            "signal transduction", "cell signaling", "signaling pathway",
            "receptor activation", "second messenger", "kinase cascade",
            "mapk", "wnt signaling", "hippo signaling", "notch signaling",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Cell cycle, division, and chromosome segregation",
        definition=(
            "Cell-cycle control, checkpoints, DNA replication linked to cycle progression, "
            "mitosis, meiosis, chromosome segregation, and cytokinesis."
        ),
        mesh_terms=(
            "Cell Cycle", "Cell Cycle Checkpoints", "Cell Division",
            "Mitosis", "Meiosis", "Chromosome Segregation", "Cytokinesis",
        ),
        text_terms=(
            "cell cycle", "cell cycle checkpoint", "mitosis", "meiosis",
            "chromosome segregation", "spindle assembly", "centromere",
            "kinetochore", "cytokinesis", "abscission",
        ),
        keywords=(
            "cell cycle", "checkpoint", "mitosis", "meiotic", "meiosis",
            "chromosome segregation", "spindle", "centromere", "kinetochore",
            "cytokinesis", "abscission",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Genome organization, DNA damage, and repair",
        definition=(
            "Chromatin and nuclear genome organization, DNA replication and recombination, "
            "genome instability, DNA damage responses, and repair mechanisms."
        ),
        mesh_terms=(
            "Chromatin", "DNA Packaging", "DNA Replication", "DNA Recombination",
            "DNA Damage", "DNA Repair", "Genomic Instability",
        ),
        text_terms=(
            "chromatin organization", "chromatin dynamics", "nuclear organization",
            "DNA replication", "DNA recombination", "DNA damage response",
            "DNA repair", "replication stress", "genome instability",
            "double-strand break",
        ),
        keywords=(
            "chromatin", "nuclear organization", "dna replication",
            "dna recombination", "dna damage", "dna repair", "replication stress",
            "genome instability", "double-strand break",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Autophagy, cell death, and quality control",
        definition=(
            "Autophagic pathways, regulated cell-death programs, proteostasis, organelle "
            "quality control, and ubiquitin-proteasome-dependent quality control."
        ),
        mesh_terms=(
            "Autophagy", "Mitophagy", "Regulated Cell Death", "Apoptosis",
            "Proteostasis", "Ubiquitin-Proteasome System",
        ),
        text_terms=(
            "autophagy", "mitophagy", "apoptosis", "necroptosis", "pyroptosis",
            "ferroptosis", "regulated cell death", "proteostasis",
            "protein quality control", "ubiquitin-proteasome system",
        ),
        keywords=(
            "autophag", "mitophagy", "apoptosis", "necroptosis", "pyroptosis",
            "ferroptosis", "regulated cell death", "proteostasis",
            "protein quality control", "ubiquitin-proteasome",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Cell metabolism, stress, and homeostasis",
        definition=(
            "Cellular bioenergetics and metabolism, nutrient sensing, redox control, "
            "stress adaptation, unfolded-protein response, and maintenance of homeostasis."
        ),
        mesh_terms=(
            "Energy Metabolism", "Cell Respiration", "Oxidative Stress",
            "Endoplasmic Reticulum Stress", "Unfolded Protein Response",
            "Cellular Homeostasis",
        ),
        text_terms=(
            "cell metabolism", "cellular metabolism", "bioenergetics",
            "nutrient sensing", "metabolic stress", "oxidative stress",
            "redox regulation", "endoplasmic reticulum stress",
            "unfolded protein response", "cellular homeostasis",
        ),
        keywords=(
            "cell metabolism", "cellular metabolism", "bioenergetic",
            "nutrient sensing", "metabolic stress", "oxidative stress",
            "redox regulation", "endoplasmic reticulum stress",
            "unfolded protein response", "cellular homeostasis",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
    Category(
        name="Stem cells, differentiation, aging, and regeneration",
        definition=(
            "Stem-cell identity and niches, cell-fate decisions, differentiation and "
            "reprogramming, cellular aging and senescence, regeneration, and tissue repair."
        ),
        mesh_terms=(
            "Stem Cells", "Stem Cell Niche", "Cell Differentiation",
            "Cellular Reprogramming", "Cellular Senescence", "Regeneration",
        ),
        text_terms=(
            "stem cell", "stem cell niche", "pluripotency", "cell differentiation",
            "lineage commitment", "cell fate", "cellular reprogramming",
            "cellular senescence", "cellular aging", "regeneration", "tissue repair",
        ),
        keywords=(
            "stem cell", "stem cell niche", "pluripot", "cell differentiation",
            "lineage commitment", "cell fate", "reprogramming",
            "cellular senescence", "cellular aging", "regeneration", "tissue repair",
        ),
        basis_ids=("ASCB_EMBO_2026", "NLM_MESH_G04", "ALBERTS_MBOC4"),
    ),
)

EXCLUDED_TYPES = {
    "review", "systematic review", "meta-analysis", "editorial", "comment", "guideline",
    "case reports", "case report", "news", "letter", "protocol", "correction", "retraction",
    "published erratum", "practice guideline", "consensus development conference",
    "review-article", "case-report", "letter-to-the-editor", "editorial-material",
    "systematic-review", "meta-analysis-article", "correction-article", "retracted-article",
    "preprint", "preprint-article", "retraction", "retraction notice",
    "withdrawn publication", "withdrawn-article", "article on hold", "on-hold",
}

TITLE_EXCLUSION = re.compile(
    r"\b(review|systematic review|scoping review|meta-analysis|meta analysis|editorial|commentary|"
    r"guideline|consensus statement|case report|case series|study protocol|correction|erratum|"
    r"preprint|withdrawn|retracted|retraction|on[ -]hold)\b",
    re.IGNORECASE,
)


def hard_exclusion(
    title: str,
    abstract: str,
    article_types: list[str],
    open_access: bool = True,
    journal: str = "",
) -> str | None:
    if not open_access:
        return "E_NOT_PMC_OPEN_ACCESS"
    if not abstract or len(re.sub(r"\s+", " ", abstract).strip()) < 80:
        return "E_EMPTY_OR_INSUFFICIENT_ABSTRACT"
    normalized = {x.strip().lower() for x in article_types}
    status_text = " ".join(normalized | {title.lower(), journal.lower()})
    if "preprint" in status_text:
        return "E_PREPRINT"
    if re.search(r"\b(retracted|retraction|withdrawn)\b", status_text):
        return "E_RETRACTED_OR_WITHDRAWN"
    if re.search(r"\bon[ -]hold\b", status_text):
        return "E_ON_HOLD"
    if normalized & EXCLUDED_TYPES:
        return "E_EXCLUDED_ARTICLE_TYPE"
    if TITLE_EXCLUSION.search(title):
        return "E_EXCLUDED_TITLE_PATTERN"
    return None


def choose_category(title: str, abstract: str, source_categories: set[str]) -> str:
    text = f"{title} {abstract}".lower()
    scored: list[tuple[int, int, str]] = []
    for order, category in enumerate(CATEGORIES):
        score = sum(text.count(term) for term in category.keywords)
        if category.name in source_categories:
            score += 2
        scored.append((score, -order, category.name))
    return max(scored)[2]
