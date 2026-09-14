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
- **No distinct opinion camps.** Respondent communities are not significantly stronger than in shuffled data (null-model p ≈ 0.2), are unstable across runs, and opinion scores are unimodal: a continuum, not blocs.
- **A leading axis of disagreement, not a single one.** Regulation and traditional structure (T12, E02, E03) versus open, flexible learning (E04, E01, E09). Parallel analysis finds several small dimensions above noise. All 4 negative statement correlations involve Education and have bootstrap 95% intervals excluding zero.
- **Domains matter, but themes cross them.** Within-domain links are 1.8× more common than chance, yet 57% of links cross domains, and 3 of 6 reproducible statement clusters (bootstrap consensus cores) mix domains.
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
- Two single Louvain runs on bootstrap resamples agree only at ARI ≈ 0.1, so statement clusters are reported as bootstrap consensus cores.
- The same comparison is made for missing-data handling, path cost, the null model, Pearson vs Spearman, the edge threshold (vs FDR control), centrality, the split index, η² and PCA with parallel analysis.

## Analyses

Every structural claim is checked against a baseline.

- **Network metrics:** density, clustering (vs shuffled-data and degree-preserving rewired baselines), components, hop and weighted path lengths.
- **Respondent communities:** Louvain, tested against 50 shuffled datasets (each statement's answers permuted across respondents). Stability is measured across seeds and 90% subsamples. Bimodality of opinion scores and assortativity test whether the structure is camps or a continuum.
- **Statement network:** negative edges with bootstrap confidence intervals and FDR/Bonferroni correction, within-domain edge enrichment, modularity vs rewired graphs, bootstrap consensus cores, unconnected statements, betweenness.
- **Polarization:** a split index, min(% agree, % disagree), rather than variance alone, plus consensus and rejected statements.
- **Domain structure:** domain-score correlations, Cronbach's α, and PCA with Horn's parallel analysis.
- **Robustness:** sensitivity to k and threshold, expected chance edges per threshold, raw vs centred cosine, Pearson vs Spearman.

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
│   ├── advanced_analysis.py       # null models, bootstraps, FDR, consensus cores, α, PCA + parallel analysis
│   ├── visualize.py               # figures 1–7
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

python notebooks/full_pipeline.py # ≈ 2 min; regenerates figures, report and results.json
pytest -q                         # unit tests for similarity, η², split index, FDR, bimodality and filtering
```

Every number and table in the report is computed during the run; nothing is hardcoded. Team name, repository link and contributions are set at the top of `notebooks/full_pipeline.py`.
