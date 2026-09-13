# Network Analysis of Class Survey Opinions (Titan Mandal)

An end-to-end Python network science project that constructs and analyzes two complementary networks from a 97-respondent, 60-statement Likert survey across four thematic domains: **Technology/AI (T)**, **Education (E)**, **Ethics/Society (S)**, and **Environment (V)**.

---

## 📌 Project Overview

This codebase implements full network analysis pipelines:
1. **Network 1 — Respondent Similarity Network**:
   - Nodes: 97 survey respondents
   - Edges: Pairwise cosine similarity calculated over encoded Likert responses (−2 to +2), ignoring missing values pairwise.
   - Graph Sparsification: Top-8 nearest neighbors ($k=8$) per respondent.
   - Analysis: Community detection using Louvain modularity optimization, eigenvector centrality (opinion leaders), degree distribution, and homophily/variance analysis across categories.

2. **Network 2 — Statement Co-Endorsement Network**:
   - Nodes: 60 survey statements (15 per domain: T, E, S, V)
   - Edges: Pairwise Pearson correlation between statement opinion vectors across respondents, thresholded at $|r| \ge 0.3$.
   - Analysis: Domain clustering, signed correlation edges (positive vs negative endorsement), betweenness centrality (identifying bridge statements that connect distinct thematic domains), and polarization analysis.

---

## 📁 Repository Structure

```
Titan_Mandal/
├── data/
│   └── Survey_Results_UC.csv      # Cleaned raw survey responses
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_prep.py               # Data loading, Likert encoding (−2..+2), missing data handling
│   ├── build_networks.py          # Construction of Respondent (N1) & Statement (N2) networks
│   ├── analysis.py                # Network metrics, Louvain community detection, centrality, polarization
│   ├── visualize.py               # Publication-quality figure generation (6 figures)
│   └── generate_pdf.py            # PDF report generator from markdown report & figures
├── notebooks/
│   └── full_pipeline.py           # End-to-end execution script
├── output/
│   ├── figures/                   # Rendered PNG figures (fig1 .. fig6)
│   ├── report.md                  # Comprehensive Markdown report
│   └── report.pdf                 # Generated publication-ready 8-page PDF report
├── implementation_plan.md         # Detailed design specification
├── requirements.txt               # Dependencies specification
└── README.md                      # Setup and reproduction guide
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Titan_Mandal.git
   cd Titan_Mandal
   ```

2. **Set up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Reproduction

To execute the full end-to-end pipeline (loads data, constructs networks, runs graph metrics, and generates all 6 figures & results):

```bash
python notebooks/full_pipeline.py
```

To compile the 8-page final PDF report from `output/report.md`:

```bash
python src/generate_pdf.py
```

---

## 📊 Visualizations Generated

| Figure | Output File | Description |
| :--- | :--- | :--- |
| **Figure 1** | `output/figures/fig1_similarity_distribution.png` | Histogram of pairwise respondent cosine similarities with sparsification cutoff. |
| **Figure 2** | `output/figures/fig2_respondent_network.png` | Network 1 graph layout colored by Louvain community and sized by eigenvector centrality. |
| **Figure 3** | `output/figures/fig3_statement_network.png` | Network 2 graph layout colored by domain category (T/E/S/V) with signed correlation edges. |
| **Figure 4** | `output/figures/fig4_polarization_bars.png` | Horizontal bar charts of top-5 polarizing and top-5 consensus statements. |
| **Figure 5** | `output/figures/fig5_community_heatmap.png` | Heatmap of mean opinion scores by community across all 4 categories. |
| **Figure 6** | `output/figures/fig6_degree_distribution.png` | Degree distribution histogram for Network 1. |

---

## 📄 License & Attribution
Designed for Network Science / Data Processing & Complex Networks coursework assignment.
