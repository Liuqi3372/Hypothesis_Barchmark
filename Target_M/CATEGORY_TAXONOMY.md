# Evidence-Based Cell Biology Taxonomy

## Purpose

This taxonomy defines the scope of Step 1. It is an operational retrieval
taxonomy, not a claim that cell biology has one universally accepted,
mutually exclusive classification. Cell-biology papers often span several
processes. Step 1 therefore records every matching source category and assigns
one primary category for balanced sampling and output organization. Step 2
still decides whether the paper's central research question is genuinely cell
biological.

## Topic Coverage Sampling

For a fixed annual budget, Step 1 does not force equal topic counts. It first
guarantees a configurable minimum number of primary papers per topic. Remaining
places are allocated according to the annual candidate distribution. When a
paper matches multiple topic queries, it contributes `1/k` prevalence weight to
each of its `k` matched topics, preventing the same paper from inflating the
estimated literature volume. The PMCID is nevertheless selected once and is
assigned one primary category only.

## Authority and construction method

The categories were constructed from three complementary authoritative
sources:

1. **ASCB|EMBO Cell Bio 2026 scientific topics.** This is the current topic
   taxonomy used by the American Society for Cell Biology and EMBO for their
   main cell-biology meeting. It supplies the contemporary category boundaries
   and subtopics.
2. **NLM Medical Subject Headings (MeSH).** The `Cell Physiological Phenomena`
   hierarchy and related descriptors supply controlled vocabulary for
   reproducible database retrieval.
3. **Alberts et al., Molecular Biology of the Cell, 4th edition.** The chapter
   structure supplies a canonical textbook cross-check that the categories
   cover the core conceptual areas of cell biology.

The PMC query for each category is:

```text
(MeSH descriptors) OR (controlled synonyms in Title/Abstract)
```

Using both components is deliberate. MeSH improves terminological consistency,
while Title/Abstract synonyms retain recent PMC articles that are not yet
MEDLINE-indexed. The official PMC User Guide notes that using `[mh]` alone
restricts retrieval to the MEDLINE-indexed subset of PMC.

## Operational categories

| Primary category | Included scope | Principal authority mapping |
|---|---|---|
| Membranes, organelles, and intracellular trafficking | Membranes, organelles, contact sites, protein sorting, endocytosis, exocytosis, and secretion | ASCB: Membrane Biology and Trafficking; Organelles. Alberts: Chapters 10–14. |
| Cytoskeleton, polarity, and cell motility | Actin, microtubules, intermediate filaments, motors, polarity, cilia, flagella, and intrinsic motility machinery | ASCB: Cytoskeleton and Motility. Alberts: Chapter 16. |
| Cell adhesion, migration, and mechanobiology | Cell-cell/cell-matrix adhesion, ECM, directed migration, and mechanical regulation | ASCB: Cell Adhesion, Migration, and the Extracellular Environment; Biophysics, Structural Biology, & Mechanobiology. Alberts: Chapter 19. |
| Signal transduction and cell communication | Receptors, signaling networks, second messengers, and intercellular communication | ASCB: Signal Transduction. Alberts: Chapter 15. |
| Cell cycle, division, and chromosome segregation | Cycle control, checkpoints, mitosis, meiosis, segregation, cytokinesis, and abscission | ASCB: Cell Division and Cell Cycle Control. Alberts: Chapters 17–18. |
| Genome organization, DNA damage, and repair | Chromatin, nuclear organization, replication, recombination, genome instability, damage response, and repair | ASCB: Genome Biology. Alberts: Chapters 4–5. |
| Autophagy, cell death, and quality control | Autophagy, regulated cell-death programs, proteostasis, and organelle/protein quality control | ASCB: Cell Stress and Quality Control; Cell Death. Alberts: Chapter 17. |
| Cell metabolism, stress, and homeostasis | Bioenergetics, nutrient sensing, redox control, metabolic/ER stress, stress adaptation, and homeostasis | ASCB: Cell Metabolism; Cell Stress and Quality Control. Alberts: Chapters 2 and 14. |
| Stem cells, differentiation, aging, and regeneration | Stem-cell identity/niches, fate choice, differentiation, reprogramming, senescence, regeneration, and repair | ASCB: Developmental and Stem Cell Biology; Cell Aging and Regeneration. Alberts: Chapters 21–22. |

## Boundary rules

- A disease, drug, organism, assay, or technology is not a category by itself.
  It is eligible only when the central question concerns a cellular mechanism.
- "Cytoskeleton, polarity, and cell motility" covers the machinery producing
  movement. "Cell adhesion, migration, and mechanobiology" covers interaction
  with other cells, matrix, gradients, and physical environments.
- DNA damage and repair are separated from general cellular stress because
  ASCB and Alberts place genome maintenance with genome biology. Oxidative,
  metabolic, ER, and proteotoxic stress belong to metabolism/stress/homeostasis
  or quality control, depending on the central mechanism.
- Autophagy is grouped with cell death and quality control because it can
  mediate survival, turnover, or death; the paper's central mechanism determines
  the primary assignment.
- A paper may match multiple retrieval categories. `source_categories`
  preserves all matches; `category` is the single highest-scoring primary label.

## References

- American Society for Cell Biology and EMBO. **Cell Bio 2026: Main Scientific
  Topics/Subtopics.** https://www.ascb.org/cellbio2026/abstracts-2026/
- National Library of Medicine. **MeSH: Cell Physiological Phenomena (G04).**
  https://www.ncbi.nlm.nih.gov/mesh/G04
- National Library of Medicine. **PMC User Guide: Article Search and MeSH
  fields.** https://pmc.ncbi.nlm.nih.gov/about/userguide/
- Alberts B, Johnson A, Lewis J, et al. **Molecular Biology of the Cell.**
  4th ed. Garland Science; 2002. NCBI Bookshelf:
  https://www.ncbi.nlm.nih.gov/books/NBK21054/
