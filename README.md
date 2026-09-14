# Opinion Network Formation from Class Survey Responses

This project is a complete network-science analysis of a class opinion survey, implemented as part of the DPCN coursework assignment on opinion network formation. The objective is to treat each respondent and each survey statement as a node in a relational system, then examine how opinion patterns cluster, polarize, and connect across thematic domains.

The repository analyzes a 96-respondent, 60-statement Likert survey covering four domains:

- Technology / AI (T)
- Education (E)
- Ethics / Society (S)
- Environment (V)

The task is not simply to compute descriptive statistics. Instead, the goal is to model the social architecture of opinions: which respondents are close to one another, which statements co-move together, which communities emerge, and which issues act as bridges or polarizing points within the class.

---

## 1. Assignment objective

The assignment is based on the principle that opinions are relational rather than isolated. A raw mean response or frequency table cannot reveal the network structure of agreement, disagreement, or clustering.

This project therefore builds two complementary graphs:

1. Network 1 — Respondent network
   - Nodes = respondents
   - Edge strength = similarity between opinion patterns
   - Purpose = detect communities, central respondents, and local clusters of like-minded individuals

2. Network 2 — Statement network
   - Nodes = statements
   - Edge strength = correlation between statements across respondents
   - Purpose = reveal which issues co-occur, which are antagonistic, and which statements act as bridges between thematic domains

The assignment explicitly requires careful methodological discipline: similarity values are used for interpretation, but shortest-path computations must use a transformed distance metric. In other words, the model must not incorrectly treat cosine similarity itself as a path distance.

---

## 2. Dataset

The raw data are stored in:

- data/Survey_Results_UC.csv

The dataset contains:

- 96 respondents
- 60 statements
- 15 statements per domain
- coded Likert values from -2 to +2
- missing values stored as NaN instead of neutral values

Response encoding:

- Strongly Disagree = -2
- Disagree = -1
- Neutral = 0
- Agree = +1
- Strongly Agree = +2

Missing responses are left as NaN so that pairwise-complete operations can be applied without silently converting missingness into a neutral opinion.

---

## 3. Data preparation and preprocessing

The preprocessing pipeline in src/data_prep.py handles:

- loading the raw CSV
- parsing the statement codes
- mapping each statement to its domain (T, E, S, V)
- converting raw Likert text into numeric ordinal responses
- preserving NaN values for incomplete responses
- summarising missingness at both statement and respondent level

This preserves the structure of the survey and avoids distorting the opinion patterns before network construction begins.

---

## 4. Network construction methodology

### 4.1 Network 1: respondent similarity network

The respondent network is built from pairwise-complete cosine similarity between each respondent's 60-item opinion vector. This measures the alignment of opinion profiles while ignoring missing entries when both respondents answered the same statement.

Key points:

- Similarity is computed across all statements for each pair of respondents
- Only pairwise-complete responses are used
- The graph is sparsified using a k-nearest-neighbor rule with k = 8
- Edge weights retain the original similarity value
- For shortest-path calculations, a distance transform is applied

The critical methodological correction is:

- similarity is not treated as a path length
- path distance uses d = 1 / similarity for positive similarities
- non-positive similarities are treated as non-connecting for shortest-path operations

This ensures the graph remains mathematically valid: the weight is interpreted as relationship strength, while the distance is used only as a graph-cost measure.

### 4.2 Network 2: statement correlation network

The statement network is derived from Pearson correlation across respondents for each pair of statements. A signed graph is then formed using an absolute-correlation threshold of |r| >= 0.30.

The graph includes:

- positive edges: meaning the two statements are co-endorsed
- negative edges: meaning the statements tend to move in opposite directions
- nodes grouped by domain category

This network reveals whether certain statements are conceptually aligned or structurally opposed.

---

## 5. Analysis methods

The analysis pipeline includes the following components:

### 5.1 Descriptive graph statistics

For both networks, we compute:

- node and edge counts
- density
- average degree
- clustering coefficient
- connected components and largest connected component
- average shortest-path length and diameter using the distance attribute

### 5.2 Community detection

Louvain community detection is used on the respondent graph to find groups of respondents sharing similar opinion patterns. This highlights structural clustering that would be invisible in a raw distribution table.

### 5.3 Centrality analysis

We compute centrality measures to determine which respondents or statements are most structurally influential:

- degree centrality for local connectivity
- betweenness centrality for bridging positions
- weighted centrality on the proper distance-aware graph

### 5.4 Polarization and consensus

Statement-level variance is used to identify:

- most polarizing statements (highest variance)
- most consensual statements (lowest variance)

This distinguishes issues on which the class is divided from issues on which the class is broadly aligned.

### 5.5 Domain-level analysis

We compare average opinion profiles across T, E, S, and V to see whether disagreement is concentrated in one domain or spread across the spectrum. This helps distinguish domain-specific conflict from general class consensus.

### 5.6 Robustness checks

The pipeline also evaluates sensitivity to key modeling choices:

- varying k in the respondent network
- varying correlation thresholds in the statement network
- community stability across repeated Louvain runs
- consistency between cosine similarity and Pearson-based respondent similarity

These checks ensure the conclusions are not artifacts of one arbitrary parameter choice.

---

## 6. Repository structure

```text
Titan_Mandal/
├── data/
│   └── Survey_Results_UC.csv
├── notebooks/
│   └── full_pipeline.py
├── output/
│   ├── figures/
│   ├── report.md
│   ├── report.pdf
│   └── results.json
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── advanced_analysis.py
│   ├── build_networks.py
│   ├── data_prep.py
│   ├── generate_pdf.py
│   └── visualize.py
├── tests/
│   └── test_network_math.py
├── README.md
├── requirements.txt
└── Survey_Results_UC.csv
```

---

## 7. How to run the project

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis pipeline:

```bash
python notebooks/full_pipeline.py
```

This script will:

- load and encode the survey data
- construct both networks
- compute graph metrics and communities
- generate all figures
- write the Markdown report
- export the final PDF
- save structured results to output/results.json

Generate the PDF independently if needed:

```bash
python src/generate_pdf.py
```

---

## 8. Main findings

The current project results indicate that:

- the class is not uniformly polarized across all questions
- strong agreement exists on environmental and ethical themes
- educational statements show the largest structural polarization
- some respondents form distinct communities based on their educational views rather than on general social values
- the statement network is dominated by positive associations, but negative edges capture important conceptual tensions

This shows that simple questionnaire summaries miss the relational structure of opinion formation. Network analysis reveals the architecture of agreement and disagreement, which is exactly the aim of the DPCN opinion-network assignment.

---

## 9. Interpretation

The essential value of this project is that it moves beyond tabular opinion summaries and reconstructs the hidden structure of the class mindset. Rather than asking only “what was the average response?”, it asks:

- Which respondents are close to one another?
- Which statements cluster together?
- Which issues divide the class?
- Which nodes are bridges between communities?
- How stable are these structures under different modeling assumptions?

These questions are central to opinion-network analysis and are exactly aligned with the DPCN assignment.

---

