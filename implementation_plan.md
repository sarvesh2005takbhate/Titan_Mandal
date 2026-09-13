# Network Analysis of Class Survey Opinions — Implementation Plan

## Goal

Build two complementary networks from the 97-respondent, 60-statement Likert survey, perform full network analysis (community detection, centrality, polarization), generate 6+ publication-quality visualizations, and produce an 8-page PDF report.

## Dataset Summary

- **File**: `Survey_Results_UC.csv` (currently at repo root — will be moved to `/data`)
- **Rows**: 97 respondents (Response IDs 11–126, non-contiguous)
- **Columns**: 1 ID + 60 statements (T01–T15, E01–E15, S01–S15, V01–V15)
- **Values**: Strongly Disagree / Disagree / Neutral / Agree / Strongly Agree / No Comments / blank

---

## Proposed Changes

### Project Structure Setup

Create the full directory structure:

```
Titan_Mandal/
├── data/                    → Survey_Results_UC.csv
├── src/
│   ├── data_prep.py         → [NEW] Load CSV, encode Likert, handle missing data
│   ├── build_networks.py    → [NEW] Construct both networks
│   ├── analysis.py          → [NEW] All network metrics & community interpretation
│   └── visualize.py         → [NEW] All 6+ figure-generation functions
├── notebooks/
│   └── full_pipeline.py     → [NEW] End-to-end script that runs everything
├── output/
│   ├── figures/             → All saved PNGs
│   └── report.md            → Report content (render to PDF separately)
├── README.md                → [NEW] Setup & reproduction instructions
└── requirements.txt         → [NEW] Python dependencies
```

---

### Component 1: Data Preparation (`src/data_prep.py`)

#### [NEW] [`data_prep.py`](file:///c:/Users/Sarvesh/Titan_Mandal/src/data_prep.py)

- Load CSV, extract statement codes (T01–T15, E01–E15, S01–S15, V01–V15) from column headers
- Encode: Strongly Disagree=-2, Disagree=-1, Neutral=0, Agree=+1, Strongly Agree=+2
- `No Comments` and blank → `NaN` (not treated as Neutral)
- Report missing data statistics: count per statement, per respondent
- Output: clean numeric DataFrame (97 × 60) + missing data summary

---

### Component 2: Network Construction (`src/build_networks.py`)

#### [NEW] [`build_networks.py`](file:///c:/Users/Sarvesh/Titan_Mandal/src/build_networks.py)

**Network 1 — Respondent Similarity Network**:
- Compute pairwise cosine similarity (pairwise-complete, ignoring NaN pairs)
- Also compute Pearson correlation as robustness check
- Plot similarity distribution histogram → choose sparsification threshold
- Strategy: top-k nearest neighbors (k ≈ 8) OR similarity cutoff — will choose based on distribution
- Build `networkx` undirected weighted graph

**Network 2 — Statement Co-Endorsement Network**:
- Transpose data: compute Pearson correlation between all 60 statement columns
- Sparsify: keep edges where |r| > threshold (start with 0.3, adjust)
- Store edge sign (positive/negative) as attribute
- Build `networkx` graph with T/E/S/V category as node attribute

---

### Component 3: Analysis (`src/analysis.py`)

#### [NEW] [`analysis.py`](file:///c:/Users/Sarvesh/Titan_Mandal/src/analysis.py)

1. **Descriptive stats**: nodes, edges, density, avg degree, avg path length (LCC), diameter, clustering coefficient
2. **Community detection**: Louvain on Network 1 → modularity, community sizes
3. **Community interpretation**: mean encoded response per T/E/S/V category per community
4. **Centrality**: eigenvector centrality (Network 1), betweenness centrality (Network 2)
5. **Polarization**: variance per statement → rank top-5 most polarizing + top-5 consensus
6. **Missing data report**: concentration analysis

---

### Component 4: Visualization (`src/visualize.py`)

#### [NEW] [`visualize.py`](file:///c:/Users/Sarvesh/Titan_Mandal/src/visualize.py)

Six required figures:
1. Similarity distribution histogram (with threshold line)
2. Network 1 graph — nodes colored by Louvain community, sized by eigenvector centrality
3. Network 2 graph — nodes colored by T/E/S/V category, positive/negative edges in different colors
4. Bar chart — Top-5 polarizing + top-5 consensus statements
5. Heatmap — community × category mean opinion score
6. Degree distribution — Network 1

---

### Component 5: Pipeline Script

#### [NEW] [`notebooks/full_pipeline.py`](file:///c:/Users/Sarvesh/Titan_Mandal/notebooks/full_pipeline.py)

Single script that imports all modules and runs end-to-end: load → build → analyze → visualize → save all figures.

---

### Component 6: Report

#### [NEW] [`output/report.md`](file:///c:/Users/Sarvesh/Titan_Mandal/output/report.md)

8-page report with sections:
1. Team Name (placeholder)
2. GitHub Link (placeholder)
3. Dataset Documentation
4. Pipeline Followed (deep justification)
5. Analysis and Visualizations (all 6 figures + metrics in text)
6. Results and Discussion
7. Individual Contribution (placeholder table)

All numbers, metrics, and figure references will be from the **real computed data**.

---

## Execution Order

1. Set up project directories and move CSV to `/data`
2. Write `data_prep.py` → run to validate encoding
3. Write `build_networks.py` → run to build both networks
4. Write `analysis.py` → run all analyses
5. Write `visualize.py` → generate all 6 figures
6. Write `full_pipeline.py` → verify end-to-end
7. Write `report.md` with real data
8. Generate PDF from report
9. Write `README.md` and `requirements.txt`

## Verification Plan

### Automated Tests
- Run `full_pipeline.py` end-to-end and verify all 6 figures are generated in `/output/figures/`
- Sanity checks: network is not fully connected blob, not fully disconnected, modularity > 0

### Manual Verification
- Visual inspection of all 6 figures
- Report cross-check: all numbers match computed output
