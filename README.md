# Opinion Network Formation from Class Survey Responses

DPCN Assignment 1 — Team **Titan Mandal**. The final report is [output/report.pdf](output/report.pdf).

We analyse a 60-statement Likert survey covering four domains (Technology / AI **T**, Education **E**, Ethics / Society **S**, Environment **V**) as two networks:

| | Respondent network | Statement network |
|---|---|---|
| Nodes | 87 respondents (after cleaning) | 60 statements |
| Similarity | Row-centred cosine over shared answers | Pearson r across respondents |
| Edges | k-nearest neighbours, k = 8 | \|r\| ≥ 0.30, signed |
| Path distance | 1 / similarity | 1 / r, positive edges only |

---

## Key findings

- **A consensus class.** 79% of answers are Agree/Strongly Agree. Environment and Ethics are near-unanimous.
- **No distinct opinion camps.** Respondent communities are not significantly stronger than in shuffled data (null-model p ≈ 0.2) and are unstable across runs.
- **One clearest axis of disagreement.** Regulation and traditional structure (T12, E02, E03) versus open, flexible learning (E04, E01, E09). All negative statement correlations involve Education.
- **Themes cut across domains.** 57% of statement edges link different domains, forming four cross-domain clusters (e.g. digital rights & AI regulation).
- **Uncertain topics stay disconnected.** The most-Neutral statements (E13, T08, T09) correlate with nothing.

---

## Data preparation

- Likert encoding: Strongly Disagree = −2 … Strongly Agree = +2. Missing answers stay `NaN`, never Neutral.
- Missingness is mostly dropout. All 505 blank cells come from respondents who abandoned the form; only 37 cells are deliberate "No Comments".
- The **9 respondents with fewer than 30 answers** are excluded: 5 blank forms and 4 who answered only the Technology page. Left in, they would create isolated nodes and fake communities.
- Response-style flags (straight-lining, net dissent, mostly Neutral) are reported but those respondents are kept.
- Raw cosine similarity mostly measures overall agreement level, so each respondent's mean answer is subtracted first.

## Why these methods

Section 2 of the report has a **design-choice table**. Each row compares a choice with the obvious alternative, using evidence computed during the run:

- A global threshold would isolate 15 respondents; k-NN connects everyone.
- Label propagation finds a single community, which supports the finding that there are no camps.
- ARI is chance-corrected, whereas NMI scores unrelated partitions above zero.
- The same comparison is made for missing-data handling, path cost, the null model, Pearson vs Spearman, the edge threshold, centrality, the split index, η², α and PCA.

## Analyses

- **Network metrics:** density, clustering, components, hop and weighted path lengths.
- **Communities:** Louvain communities, tested against a null model built from 50 datasets with each statement's answers shuffled across respondents. Stability is measured across seeds and 90% subsamples. η² shows which statements separate communities.
- **Statement network:** negative (opposition) edges, cross-domain share, Louvain clusters on positive edges, unconnected statements, betweenness.
- **Polarization:** a split index, min(% agree, % disagree), rather than variance alone, plus consensus and rejected statements.
- **Domain structure:** domain-score correlations, Cronbach's α, and PCA (PC1 = general agreement, PC2 = contested axis).
- **Robustness:** sensitivity to k and threshold, expected chance edges per threshold, and raw vs centred cosine.

---

## Repository structure

```text
Titan_Mandal/
├── data/Survey_Results_UC.csv
├── notebooks/full_pipeline.py     # end-to-end run; writes the report from computed values
├── src/
│   ├── data_prep.py               # loading, encoding, missing-data audit, filtering, style flags
│   ├── build_networks.py          # centred cosine, k-NN graph, signed statement network
│   ├── analysis.py                # metrics, Louvain, η², centrality, split index
│   ├── advanced_analysis.py       # null model, stability, sensitivity, clusters, α, PCA
│   ├── visualize.py               # figures 1–9
│   └── generate_pdf.py            # report.md → report.pdf
├── tests/test_network_math.py
├── output/
│   ├── figures/
│   ├── report.md
│   ├── report.pdf
│   └── results.json               # every computed result
└── requirements.txt
```

## How to run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python notebooks/full_pipeline.py # ≈ 40 s; regenerates figures, report and results.json
pytest -q                         # unit tests for the similarity, η², split-index and filtering maths
```

Every number and table in the report is computed during the run; nothing is hardcoded. Team name, repository link and contributions are set at the top of `notebooks/full_pipeline.py`.
