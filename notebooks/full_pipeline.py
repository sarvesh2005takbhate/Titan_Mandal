"""
full_pipeline.py — End-to-end survey network analysis.

    python notebooks/full_pipeline.py

Regenerates output/figures/*.png, output/report.md, output/report.pdf and
output/results.json. Every number in the report is computed here.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# Runtime warnings are deliberately NOT suppressed: the analysis is warning-clean on
# the committed data (verified), and a numerical issue introduced by a future edit
# should surface rather than pass silently.

import networkx as nx
import numpy as np
import pandas as pd

from src.data_prep import CATEGORY_LABELS, DOMAINS, filter_respondents, load_and_encode, response_style_flags
from src.build_networks import build_respondent_network, build_statement_network, positive_subgraph
from src.analysis import (
    centrality_table, descriptive_stats, detect_communities, domain_eta_squared,
    eta_squared, polarizing_consensus_rejected, statement_eta_squared, statement_summary,
)
from src.advanced_analysis import (
    agreement_bias_check, bimodality_coefficient, bootstrap_edge_ci, community_stability, correlation_tests,
    design_choice_evidence, domain_edge_enrichment, domain_reliability, domain_scores, item_pair_correlations,
    modularity_null_model, negative_edges, parallel_analysis, pca_imputation_sensitivity, principal_components,
    respondent_k_sensitivity, rewired_null, statement_robust_cores, statement_threshold_sensitivity,
)
from src import visualize as viz
from src.generate_pdf import build_pdf

# ── Configuration ────────────────────────────────────────────────────────────
TEAM_NAME = "Titan Mandal"
REPO_URL = "https://github.com/sarvesh2005takbhate/Titan_Mandal"
CONTRIBUTIONS = {
    "Hrishikesh": "Data preparation, Likert encoding, missing-data handling, respondent similarity construction.",
    "Sarvesh": "Network construction, Louvain community detection, centrality analysis, robustness experiments.",
    "Shourya": ("Visualization, statement-level analysis, post-review revisions including threshold/FDR analysis, "
                "PCA sensitivity, robustness and null-model updates, refined statement/community interpretations, "
                "dynamics discussion, and final verification of the results and report."),
}
K = 8                     # nearest neighbours in the respondent network
CORR_THRESHOLD = 0.30     # |r| cut-off in the statement network
MIN_ANSWERED = 30         # respondents answering fewer statements are excluded
NULL_RUNS = 500  # permutation null model; 500 runs give a p-value resolution of 1/501 ≈ 0.002
# Near-duplicate statements placed in different domains (answer-consistency check).
DUPLICATE_PAIRS = [("T10", "S05"), ("T02", "E12")]
# A robust statement core is named after the anchor statement it contains ("Mixed" otherwise).
THEME_ANCHORS = {
    "T10": "Regulation & digital rights",
    "T02": "AI-ready, industry-linked education",
    "E01": "Active, interdisciplinary learning",
    "S10": "Evidence & civic participation",
    "V03": "Sustainable habits & education",
    "V06": "Resource conservation",
}


def short(code, text, width=70):
    return f"{code} ({textwrap.shorten(text[code], width, placeholder='…')})"


def pct(x, digits=0):
    return f"{x:.{digits}%}"


def main():
    fig_dir = viz.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("[1/5] Loading data ...")
    df_all, text, missing = load_and_encode("data/Survey_Results_UC.csv")
    df, dropped = filter_respondents(df_all, MIN_ANSWERED)
    flags = response_style_flags(df)
    n_all, n = len(df_all), len(df)
    answers = df.stack()
    share_agree, share_neutral, share_disagree = (answers > 0).mean(), (answers == 0).mean(), (answers < 0).mean()
    partial = df.notna().sum(axis=1).loc[lambda s: s < df.shape[1] - 5]
    nc_after = int(df.isna().sum().sum())

    # ── 2. Networks ──────────────────────────────────────────────────────────
    print("[2/5] Building networks ...")
    G1, sim = build_respondent_network(df, k=K, center=True)
    G2, corr = build_statement_network(df, CORR_THRESHOLD)
    G2_pos = positive_subgraph(G2)
    stats1, stats2 = descriptive_stats(G1), descriptive_stats(G2_pos)

    # ── 3. Analysis ──────────────────────────────────────────────────────────
    print("[3/5] Analysing (null models and bootstraps take a minute or two) ...")
    partition, Q = detect_communities(G1)
    sizes = pd.Series(partition).value_counts().sort_index()
    null = modularity_null_model(df, Q, k=K, runs=NULL_RUNS)
    stability = community_stability(G1, df, partition, k=K)
    bias = agreement_bias_check(df, k=K)
    k_sens = respondent_k_sensitivity(df, partition)
    cent1 = centrality_table(G1)
    eta = statement_eta_squared(df, partition)
    eta_domain = domain_eta_squared(df, partition)
    top_eta = list(eta.head(4).index)

    summary = statement_summary(df)
    polar, consensus, rejected = polarizing_consensus_rejected(summary)
    tests = correlation_tests(df)
    neg = bootstrap_edge_ci(df, negative_edges(G2)).merge(tests[["u", "v", "fdr", "bonferroni"]], on=["u", "v"], how="left")
    thr_sens = statement_threshold_sensitivity(df)
    thr_row = thr_sens.set_index("threshold").loc[CORR_THRESHOLD]
    cent2 = centrality_table(G2_pos)
    signed_isolates = sorted(n for n in G2 if G2.degree(n) == 0)
    positive_only_isolates = sorted(set(stats2["isolates"]) - set(signed_isolates))
    enrich = domain_edge_enrichment(G2)
    rewired = rewired_null(G2_pos)
    cores = statement_robust_cores(df, CORR_THRESHOLD)
    for core in cores["cores"]:
        names = [name for anchor, name in THEME_ANCHORS.items() if anchor in core["members"]]
        core["theme"] = " / ".join(names) if names else "Mixed"
        core["domains"] = sorted({c[0] for c in core["members"]}, key=DOMAINS.index)
    mixed_cores = [c for c in cores["cores"] if len(c["domains"]) > 1]
    privacy_core = next((c for c in cores["cores"] if "T10" in c["members"]), None)

    dom_corr = domain_scores(df).corr()
    alpha = domain_reliability(df)
    pca = principal_components(df, n_components=3)
    parallel = parallel_analysis(df)
    pc2 = pca["loadings"]["PC2"]
    pc2_scores = pca["scores"]["PC2"]
    groups = df.index.to_series().map(partition)
    pc2_by_comm = pc2_scores.groupby(groups).mean()
    pc2_eta = eta_squared(pc2_scores, groups)
    pc2_bimodality = bimodality_coefficient(pc2_scores)
    nx.set_node_attributes(G1, pc2_scores.to_dict(), "pc2")
    nx.set_node_attributes(G1, df.mean(axis=1).to_dict(), "agreement")
    assort_pc2 = nx.numeric_assortativity_coefficient(G1, "pc2")
    assort_agree = nx.numeric_assortativity_coefficient(G1, "agreement")
    # Negative edges whose two statements sit on opposite poles of PC2.
    axis_edges = [f"{r.u}–{r.v}" for r in neg.itertuples()
                  if pc2[r.u] * pc2[r.v] < 0 and min(abs(pc2[r.u]), abs(pc2[r.v])) >= 0.1]
    dup = item_pair_correlations(corr, DUPLICATE_PAIRS)
    ev = design_choice_evidence(df_all, df, G1, sim, partition, summary)
    split_vs_strength = summary["split"].corr(cent2["strength"])
    neutral_rank = summary["neutral"].rank(ascending=False)
    iu = np.triu_indices(len(dom_corr), 1)
    pairs = [(dom_corr.index[i], dom_corr.columns[j], dom_corr.iat[i, j]) for i, j in zip(*iu)]
    strongest, weakest = max(pairs, key=lambda p: p[2]), min(pairs, key=lambda p: p[2])

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("[4/5] Drawing figures ...")
    viz.plot_data_overview(df_all, dropped)
    viz.plot_respondent_network(G1, partition, cent1)
    viz.plot_statement_network(G2)
    viz.plot_polarization(df, {
        "Most split": list(polar.index),
        "Rejected by the class": [c for c in rejected.index if c not in polar.index],
        "Strongest consensus": list(consensus.index),
    }, text)
    viz.plot_domain_structure(dom_corr, alpha)
    viz.plot_pc2(pc2, text)
    viz.plot_robustness(null, k_sens, thr_sens, CORR_THRESHOLD)

    # ── 5. Report ────────────────────────────────────────────────────────────
    print("[5/5] Writing report ...")
    significant = null["p"] < 0.05
    pc2_neg, pc2_pos = pc2.nsmallest(3).index, pc2.nlargest(3).index
    open_comm, control_comm = pc2_by_comm.idxmin(), pc2_by_comm.idxmax()
    top_split = polar.iloc[0]
    top_rejected = rejected.iloc[0]
    reliab_low = alpha[alpha < 0.7]
    straightliners = flags[flags["flags"].str.contains("straight")]["respondent"].tolist()
    dissenters = flags[flags["flags"].str.contains("dissenter")]["respondent"].tolist()
    neutrals = flags[flags["flags"].str.contains("neutral")]["respondent"].tolist()
    split_20 = int((summary["split"].round(2) >= 0.20).sum())
    sv_consensus = sum(c[0] in "SV" for c in consensus.index)
    domain_agree = {d: float((df[[c for c in df if c[0] == d]].stack() > 0).mean()) for d in DOMAINS}
    null_clust_z = (stats1["clustering_unweighted"] - null["null_clustering_mean"]) / null["null_clustering_std"]
    rewired_q_z = (rewired["q_unweighted"] - rewired["null_q_mean"]) / rewired["null_q_std"]
    uncertain = [c for c in signed_isolates if neutral_rank[c] <= 5]
    n_tests = len(tests)
    neg_fdr = int(neg["fdr"].sum())
    neg_bonf = [f"{r.u}–{r.v}" for r in neg.itertuples() if r.bonferroni]
    neg_ci_ok = bool((neg["ci_high"] < 0).all())

    # Threshold-specific negative-edge context (the |r| >= 0.30 network stays primary):
    # how many negative correlations exist at the report's own FDR significance boundary.
    fdr_neg = tests[tests["fdr"] & (tests["r"] < 0)]
    neg_fdr_all = len(fdr_neg)
    neg_fdr_min_r = float(tests.loc[tests["fdr"], "r"].abs().min())
    neg_fdr_with_e = int((fdr_neg["u"].str.startswith("E") | fdr_neg["v"].str.startswith("E")).sum())
    e03_neg_fdr = int(((fdr_neg["u"] == "E03") | (fdr_neg["v"] == "E03")).sum())
    t01t12_r = float(tests.loc[((tests["u"] == "T01") & (tests["v"] == "T12"))
                               | ((tests["u"] == "T12") & (tests["v"] == "T01")), "r"].iloc[0])
    # Strongest association of each signed isolate — just below the edge cut-off.
    iso_best = {}
    for c in signed_isolates:
        row = corr[c].drop(c).dropna().abs().sort_values(ascending=False)
        iso_best[c] = (row.index[0], float(corr.loc[c, row.index[0]]))
    iso_abs = [abs(v[1]) for v in iso_best.values()]
    iso_abs_uncertain = [abs(iso_best[c][1]) for c in uncertain]
    # Pairwise-complete sample sizes behind the per-threshold p-values.
    pair_n_values = [int(df[[a, b]].notna().all(axis=1).sum())
                     for i, a in enumerate(df.columns) for b in df.columns[i + 1:]]
    median_pair_n = int(df.notna().sum().median())
    # Complete-case check on the mean imputation used by the PCA.
    pca_sens = pca_imputation_sensitivity(df)

    L = []
    add = L.append
    add("# Opinion Network Formation from Class Survey Responses")
    add("")
    add(f"**Team Name:** {TEAM_NAME}")
    add("")
    add(f"**GitHub Repository:** [{REPO_URL.split('github.com/')[1]}]({REPO_URL})")
    add("")
    add("---")
    add("")
    add("## Summary")
    add("")
    add(f"We model a {df_all.shape[1]}-statement Likert survey (Technology, Education, Ethics, Environment) as two networks: "
        f"a **respondent network** (who thinks alike) and a **statement network** (which opinions travel together). Key findings:")
    add("")
    add(f"- **A consensus class.** {pct(share_agree)} of all answers are Agree/Strongly Agree (Environment {pct(domain_agree['V'])}, Ethics {pct(domain_agree['S'])}); disagreement is concentrated in a handful of Education and Technology statements.")
    add(f"- **No distinct opinion camps.** Respondent communities (Q = {Q:.3f}) are {'not significantly' if not significant else 'significantly'} stronger than in shuffled data "
        f"(null Q = {null['null_mean']:.3f} ± {null['null_std']:.3f}, p = {null['p']:.2f}), change between runs, and the main opinion scores are unimodal. Opinions show diffuse structure rather than stable camps.")
    add(f"- **A leading axis of disagreement:** regulation and traditional structure ({', '.join(pc2_pos)}) versus open, flexible learning ({', '.join(pc2_neg)}). "
        f"At the |r| ≥ {CORR_THRESHOLD:.2f} cut-off all {len(neg)} negative statement correlations involve Education"
        f"{' and their bootstrap 95% intervals exclude zero' if neg_ci_ok else ''}; "
        f"relaxing to the FDR boundary (|r| ≥ {neg_fdr_min_r:.2f}) admits {neg_fdr_all} negative pairs, "
        f"so opposition is Education-centred but not exclusive to it.")
    add(f"- **Domains matter, but themes cross them.** Within-domain links occur at {enrich['within_enrichment']:.1f}× the proportion expected under a uniform-edge baseline, yet {pct(enrich['cross_share'])} of links cross domains "
        f"and {len(mixed_cores)} of the {len(cores['cores'])} reproducible statement clusters mix domains"
        + (f" (e.g. *{privacy_core['theme']}*)." if privacy_core else "."))
    add("")

    # 1. Dataset
    add("## 1. Dataset Documentation")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    add(f"| Raw responses | {n_all} respondents × {df_all.shape[1]} statements (15 per domain: T, E, S, V) |")
    add("| Encoding | Strongly Disagree = −2, Disagree = −1, Neutral = 0, Agree = +1, Strongly Agree = +2 |")
    add(f"| Missing cells | {missing['total_missing']} of {missing['total_cells']} ({pct(missing['total_missing'] / missing['total_cells'], 1)}): {missing['blank']} blank, {missing['no_comments']} “No Comments” |")
    add(f"| Excluded respondents | {len(dropped)} answered < {MIN_ANSWERED} statements: {int((dropped['answered'] == 0).sum())} blank forms, {int((dropped['domains_answered'] == 'T').sum())} answered only the Technology block |")
    add(f"| Analysed | **{n} respondents**, {nc_after} missing cells ({pct(nc_after / df.size, 1)}) |")
    add(f"| Answer mix (analysed) | {pct(share_agree)} agree · {pct(share_neutral)} neutral · {pct(share_disagree)} disagree |")
    add("")
    add(f"- **Missingness is dropout, not item refusal.** All {missing['blank']} blank cells come from respondents who abandoned the form ({int((dropped['domains_answered'] == 'T').sum())} stopped exactly after the Technology block, consistent with a form paged by domain). "
        f"Only the {missing['no_comments']} “No Comments” are interpreted as deliberate item-level skips. Both are kept as NaN, never as Neutral; similarities use only statements both respondents answered. "
        f"{len(partial)} partially complete respondents with ≥ {MIN_ANSWERED} answers are retained.")
    add(f"- **Strong agreement bias.** Mean answers are high in every domain, so raw similarity would mainly measure *how much* someone agrees. This motivates centring (Section 2).")
    add(f"- **Response-style flags (retained, reported).** {len(straightliners)} respondents gave the same answer to ≥ 75% of statements; "
        + (f"#{', #'.join(map(str, dissenters))} is a net dissenter (mean below zero); " if dissenters else "")
        + (f"#{', #'.join(map(str, neutrals))} answered mostly Neutral. " if neutrals else "")
        + "These may be genuine views or low-effort answers.")
    add(f"- **Consistency check.** Near-duplicate statements in different domains correlate positively: "
        + "; ".join(f"{r.u}–{r.v} r = {r.r:.2f}" for r in dup.itertuples()) + ", suggesting answers are coherent.")
    add("")
    add(f"![Figure 1: (a) Answer distribution by domain. (b) Statements answered per respondent; the {len(dropped)} excluded respondents are orange.](figures/fig1_data_overview.png)")
    add("")

    # 2. Pipeline
    add("## 2. Pipeline Followed")
    add("")
    add("| | Network 1: Respondents | Network 2: Statements |")
    add("|---|---|---|")
    add(f"| Nodes | {n} respondents | {df.shape[1]} statements |")
    add("| Similarity | Row-centred cosine over shared statements (≥ 5) | Pearson r across respondents (pairwise complete) |")
    add(f"| Edges | k-nearest neighbours, k = {K} (positive similarity only) | abs(r) ≥ {CORR_THRESHOLD:.2f}, signed (+ co-endorsement, − opposition) |")
    add("| Edge weight | similarity | abs(r) |")
    add("| Path distance | 1 / similarity | 1 / r, positive edges only |")
    add("")
    add(f"1. **Clean.** Encode Likert answers, keep missing as NaN, and drop the {len(dropped)} respondents with < {MIN_ANSWERED} answers (otherwise they appear as isolated nodes and fake communities).")
    add(f"2. **Respondent similarity.** Subtract each respondent's mean answer, then take cosine similarity over shared statements. Centring also weakens the link between a node's degree and its agreement level (r = {bias['raw']['degree_vs_mean_answer_r']:.2f} → {bias['centred']['degree_vs_mean_answer_r']:.2f}).")
    add(f"3. **Sparsify.** Link each respondent to its {K} most similar peers. Similarity stays the edge weight; shortest paths use distance = 1/similarity, so strong ties are short.")
    add(f"4. **Statement network.** Keep abs(r) ≥ {CORR_THRESHOLD:.2f} (p ≈ {thr_row['p_value']:.3f} at the median pairwise n = {median_pair_n}). Negative edges are kept for interpretation but excluded from path metrics and community detection, since opposition is not proximity.")
    add(f"5. **Test and analyse.** Every structural claim is checked against a baseline: shuffled answers ({NULL_RUNS}×) for respondent communities and clustering, degree-preserving rewiring for the statement network, "
        f"bootstrap resamples of respondents for edge confidence intervals and statement clusters, and parallel analysis for PCA. Descriptive tools: centrality, η², a split index for polarization, Cronbach's α.")
    add("")
    add("### Design choices and why")
    add("")
    add("Each choice was compared with the obvious alternative on this dataset. The last column gives the evidence.")
    add("")
    thr_iso = thr_sens.set_index("threshold")["isolates"]
    k_conn = ev["min_connected_k"]
    add("| Decision | Chosen | Instead of | Why / benefit (evidence from this data) |")
    add("|---|---|---|---|")
    add(f"| Missing answers | NaN, pairwise-complete | Code as Neutral | A blank is not an opinion. Coding blanks as Neutral would raise the Neutral share from {pct(ev['neutral_share_true'])} to {pct(ev['neutral_share_if_blank_neutral'])} and invent {missing['blank']} answers. |")
    add(f"| Incomplete forms | Exclude < {MIN_ANSWERED} answers | Impute | Imputing 45–60 of 60 answers fabricates a profile; kept as-is, the {int((dropped['answered'] == 0).sum())} blank forms become isolated nodes and fake communities. |")
    add(f"| Respondent similarity | Row-centred cosine | Raw cosine, Euclidean | Compares *which* statements a person favours, not how much they agree: mean similarity {bias['raw']['mean_similarity']:.2f} → {bias['centred']['mean_similarity']:.2f}. Euclidean distance also grows with the number of shared answers. |")
    add(f"| Sparsification | k-NN, k = {K} | Global similarity threshold | A threshold with the same edge count leaves {ev['threshold_isolates']} respondents isolated ({ev['threshold_components']} components); k-NN gives every respondent ≥ k neighbours"
        + (f" and is connected from k = {k_conn}" if k_conn is not None else " (no tested k ≤ 19 connects the graph on this data)")
        + f". k ≈ √n, mid-range of the tested 5–12. |")
    add(f"| Path cost | 1 / similarity | 1 − similarity, hops | Keeps strong ties short while penalising weak ties; betweenness ranks barely change with 1 − similarity (ρ = {ev['betweenness_rank_rho_inv_vs_one_minus']:.2f}), so results do not hinge on it. |")
    add(f"| Communities | Louvain | Greedy, label propagation, Girvan–Newman | Fast, weighted modularity optimisation. Greedy modularity reaches similar Q ({ev['greedy_q']:.2f}, {ev['greedy_n']} groups) but a different split (ARI {ev['greedy_ari']:.2f}); the margin over Louvain is comparable to Louvain's own seed-to-seed variation (Q = {stability['seed_q_min']:.3f}–{stability['seed_q_max']:.3f} across {stability['seed_runs']} seeds), so Louvain — whose stability is characterized in Section 3.2 — is retained. Label propagation finds {ev['lpa_n']} community — consistent with weak structure. Girvan–Newman is O(m²n) with no natural stopping point. |")
    add(f"| Significance | Shuffled-answer null model | Raw Q, random-graph baseline | k-NN graphs are modular and clustered by construction (null Q = {null['null_mean']:.2f}, clustering {null['null_clustering_mean']:.2f}). Shuffling keeps every statement's answer distribution and the full pipeline, so it tests exactly whether *combinations* of opinions cluster. |")
    add(f"| Statement clusters | Bootstrap consensus cores | One Louvain run | Two single runs on bootstrap resamples agree only at ARI {cores['single_run_ari']:.2f}; keeping statements co-clustered in ≥ {pct(cores['min_coassign'])} of {cores['runs']} resamples reports only reproducible groups. |")
    add(f"| Partition agreement | ARI | NMI | Chance-corrected: unrelated partitions of the same sizes give ARI ≈ {abs(ev['random_ari']):.2f} but NMI {ev['random_nmi']:.2f}, which rewards chance overlap. |")
    add(f"| Statement association | Signed Pearson r | Spearman; abs(r) | Spearman agrees closely (r = {ev['pearson_spearman_r']:.2f} over all pairs). Keeping the sign separates opposition from co-endorsement; abs(r) would treat the {len(neg)} opposing pairs as allies when clustering. |")
    add(f"| Edge threshold | abs(r) ≥ {CORR_THRESHOLD:.2f} | FDR cut-off, 0.20, 0.40 | Slightly stricter than Benjamini–Hochberg FDR control at 5% ({int(tests['fdr'].sum())} edges, abs(r) ≥ {tests.loc[tests['fdr'], 'r'].abs().min():.2f}). Under a simple all-null approximation, roughly {thr_row['expected_false_edges']:.0f} threshold-exceeding pairs would be expected by chance at |r| ≥ {CORR_THRESHOLD:.2f}; this is not an estimate of the actual false-discovery proportion. 0.40 isolates {int(thr_iso[0.40])} statements. |")
    add(f"| Centrality | Betweenness (on distance) | Degree, eigenvector | In k-NN every degree ≥ k by construction; eigenvector centrality tracks agreement level more (r = {ev['eigenvector_vs_agreement_r']:.2f}) than betweenness does (r = {ev['betweenness_vs_agreement_r']:.2f}). |")
    add(f"| Polarization | Split index | Variance | Variance ranks {top_rejected.name} #{ev['variance_rank'][top_rejected.name]} most polarizing although {pct(top_rejected.disagree)} reject it; the split index is high only when both camps are large and reads directly in %. |")
    add(f"| Community profile | η² per statement | Domain averages | Size-weighted 0–1 effect size, comparable across statements; domain averages smooth over heterogeneous items (Education α = {alpha['E']:.2f}). |")
    add(f"| Dimensions | PCA + parallel analysis | Eyeballing variance explained | Parallel analysis compares each component with shuffled data; it shows {parallel['n_above_noise']} components above noise, so no single axis can be over-claimed. |")
    add("")
    add(f"**Reproducibility.** One command (`python notebooks/full_pipeline.py`) regenerates every number, table and figure from the raw CSV with fixed seeds. Unit tests check the similarity, η², split-index, FDR and filtering maths, and all results are exported to `results.json`.")
    add("")

    # 3. Analysis
    add("## 3. Analysis and Visualizations")
    add("")
    add("### 3.1 Network-level metrics")
    add("")
    add("| Metric | Respondent network | Statement network (positive edges) |")
    add("|---|---:|---:|")
    for label, key in [("Nodes", "nodes"), ("Edges", "edges"), ("Density", "density"), ("Average degree", "avg_degree"),
                       ("Weighted clustering", "clustering")]:
        add(f"| {label} | {stats1[key]} | {stats2[key]} |")
    add(f"| Clustering (unweighted) vs baseline | {stats1['clustering_unweighted']:.2f} (shuffled {null['null_clustering_mean']:.2f}) | {stats2['clustering_unweighted']:.2f} (rewired {rewired['null_clustering']:.2f}) |")
    for label, key in [("Connected components", "components"),
                       ("Avg. shortest path (hops)", "avg_path_hops"), ("Diameter (hops)", "diameter_hops"),
                       ("Avg. path length (distance units)", "avg_path_distance")]:
        add(f"| {label} | {stats1[key]} | {stats2[key]} |")
    add("")
    add(f"Paths are short in both networks (about {stats1['avg_path_hops']:.0f} hops). Both have more triangles than their baselines: respondent clustering exceeds shuffled-data k-NN graphs by {null_clust_z:.0f} standard deviations, "
        f"and statement clustering exceeds degree-preserving rewired graphs. Local neighbourhoods of like-minded respondents are therefore real even though global communities are not (Section 3.2). "
        f"The respondent network is a single component; in the statement network {len(stats2['isolates'])} statements have no positive link (Section 3.3).")
    add("")

    add("### 3.2 Respondent communities")
    add("")
    add(f"- Louvain finds **{len(sizes)} communities** (sizes {', '.join(map(str, sizes.values))}), Q = {Q:.3f}.")
    add(f"- **Null model:** networks from shuffled answers give Q = {null['null_mean']:.3f} ± {null['null_std']:.3f} (z = {null['z']:.1f}, p = {null['p']:.2f}). "
        + ("The observed structure is **not significantly stronger than chance**." if not significant else "The observed structure is significantly stronger than chance."))
    add(f"- **Stability:** agreement with the main partition is ARI = {stability['seed_ari_mean']:.2f} across Louvain seeds and {stability['subsample_ari_mean']:.2f} ± {stability['subsample_ari_std']:.2f} when 10% of respondents are removed.")
    add(f"- **Diffuse structure rather than stable camps:** respondents' PC2 scores (Section 3.5) are unimodal (bimodality coefficient {pc2_bimodality:.2f}; > 0.555 would suggest two camps). "
        f"Neighbours share their position on this axis (assortativity {assort_pc2:.2f}) but not their overall agreement level ({assort_agree:.2f}). "
        f"Communities show separation along this leading axis, although the opinion space remains multidimensional: they split along PC2 (η² = {pc2_eta:.2f}; C{open_comm} {pc2_by_comm[open_comm]:+.2f} vs C{control_comm} {pc2_by_comm[control_comm]:+.2f}), only modestly more than cutting shuffled data would (η² = {null['null_pc2_eta_mean']:.2f}).")
    add(f"- **What differs between them:** membership explains most variance for {', '.join(top_eta)} (η² = {eta.iloc[3]:.2f}–{eta.iloc[0]:.2f}). No community disagrees with the class overall; they differ in emphasis.")
    add(f"- **Centrality:** #{', #'.join(map(str, cent1.index[:3]))} lie on the most shortest paths (highest betweenness). This is a structural position between similar-minded neighbourhoods, not a measure of influence.")
    add("")
    add("![Figure 2: Respondent network (k = 8, centred cosine). Colour and shape show Louvain community; node size shows betweenness.](figures/fig2_respondent_network.png)")
    add("")

    add("### 3.3 Statement network")
    add("")
    add(f"- **Signs:** {int(stats2['edges'])} positive and {len(neg)} negative edges. At the |r| ≥ {CORR_THRESHOLD:.2f} cut-off every negative edge involves an Education statement:")
    add("")
    add("| Statement A | Statement B | r | 95% bootstrap CI |")
    add("|---|---|---:|---:|")
    for r in neg.itertuples():
        add(f"| {short(r.u, text, 50)} | {short(r.v, text, 50)} | {r.r:.2f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] |")
    add("")
    add(f"- **Are they real?** {'All' if neg_fdr == len(neg) else neg_fdr} {len(neg)} survive FDR correction over all {n_tests:,} statement pairs"
        + (" and every interval excludes zero" if neg_ci_ok else "")
        + (f"; only {', '.join(neg_bonf)} also survives the much stricter Bonferroni correction." if neg_bonf else "."))
    add(f"- **Threshold context:** the “all involve Education” pattern is specific to the {CORR_THRESHOLD:.2f} cut-off: at the FDR boundary (|r| ≥ {neg_fdr_min_r:.2f}) there are {neg_fdr_all} negative correlations, {neg_fdr_with_e} involving Education (E03 alone carries {e03_neg_fdr}), including the pure Technology pair T01–T12 (r = {t01t12_r:.2f}) — opposition is Education-centred but not exclusive.")
    add(f"- **Domains vs themes:** if links ignored domains, {pct(enrich['expected_cross_share'])} would cross domains; only {pct(enrich['cross_share'])} do, so within-domain links occur at {enrich['within_enrichment']:.1f}× the proportion expected under a uniform-edge baseline. Domains matter, but most links still cross them.")
    add(f"- **Clusters:** the positive network is more modular than degree-preserving rewired graphs (Q = {rewired['q_unweighted']:.2f} vs {rewired['null_q_mean']:.2f} ± {rewired['null_q_std']:.2f}, z = {rewired_q_z:.0f}), "
        f"but any single partition is unreliable (two Louvain runs on bootstrap resamples agree at ARI {cores['single_run_ari']:.2f}). "
        f"We therefore report **robust cores**: connected components of the graph that links two statements whenever they cluster together in ≥ {pct(cores['min_coassign'])} of {cores['runs']} bootstrap resamples. "
        f"Because connectivity can chain through intermediate statements, an individual pair inside a core may fall below {pct(cores['min_coassign'])}; the last column reports each core's average, and a typical pair of statements clusters together in only {pct(cores['overall_coassign'])}.")
    add("")
    add("| Core | Theme | Statements | Avg. pair co-clustered |")
    add("|---|---|---|---:|")
    for i, core in enumerate(cores["cores"], 1):
        add(f"| {i} | {core['theme']} | {', '.join(core['members'])} | {pct(core['coassign'])} |")
    add("")
    in_cores = sum(len(c["members"]) for c in cores["cores"])
    add(f"- {in_cores} of {df.shape[1]} statements belong to a core; {len(mixed_cores)} of {len(cores['cores'])} cores mix domains"
        + (f"; e.g. core {cores['cores'].index(privacy_core) + 1} spans {', '.join(CATEGORY_LABELS[d] for d in privacy_core['domains'])} ({', '.join(privacy_core['members'])})." if privacy_core else "."))
    iso_notes = ", ".join(f"{c}–{p} {r:+.2f}" for c, (p, r) in iso_best.items())
    add(f"- **Unconnected statements:** {', '.join(signed_isolates)} have no edge at the {CORR_THRESHOLD:.2f} cut-off; their strongest associations ({iso_notes}) sit just below it, so they are weakly integrated rather than truly unconnected. "
        + (f"{', '.join(c + ' (' + pct(summary.loc[c, 'neutral']) + ' Neutral)' for c in uncertain)} are among the five most-Neutral statements, i.e. statements of weaker directional consensus with higher Neutral response shares. " if uncertain else "")
        + (f"{', '.join(positive_only_isolates)} connect only through negative edges." if positive_only_isolates else ""))
    add(f"- **Bridges:** highest betweenness: {', '.join(short(c, text, 45) for c in cent2.index[:3])}. "
        f"Divisive statements sit at the periphery (split index vs weighted degree r = {split_vs_strength:.2f}).")
    add("")
    add("![Figure 3: Statement network (abs(r) ≥ 0.30). Colour shows domain (also the code letter); red edges are negative correlations; unconnected statements are shown along the bottom.](figures/fig3_statement_network.png)")
    add("")

    add("### 3.4 Polarization and consensus")
    add("")
    add("Variance mixes a split class with a lopsided one, so we rank polarization by the **split index = min(% agree, % disagree)**, which is high only when both camps are large.")
    add("")
    add("| Group | Statement | Disagree | Neutral | Agree | Mean |")
    add("|---|---|---:|---:|---:|---:|")
    for title, frame in (("Most split", polar), ("Rejected", rejected.drop(polar.index, errors="ignore")), ("Consensus", consensus)):
        for code, r in frame.iterrows():
            add(f"| {title} | {short(code, text, 60)} | {pct(r.disagree)} | {pct(r.neutral)} | {pct(r.agree)} | {r['mean']:+.2f} |")
    add("")
    add(f"- Only {split_20} statements have ≥ 20% on both sides; the most divisive is {short(top_split.name, text, 60)} ({pct(top_split.agree)} agree vs {pct(top_split.disagree)} disagree).")
    add(f"- **{top_rejected.name} is a consensus, not a split:** {pct(top_rejected.disagree)} disagree. Ranking by variance alone would have mislabelled it as polarizing.")
    add("")
    add("![Figure 4: Answer distributions for the most split, rejected and consensus statements.](figures/fig4_polarization.png)")
    add("")

    add("### 3.5 Domain structure and opinion dimensions")
    add("")
    add(f"- **Domain scores correlate positively.** Strongest {strongest[0]}–{strongest[1]} (r = {strongest[2]:.2f}), weakest {weakest[0]}–{weakest[1]} (r = {weakest[2]:.2f}).")
    add(f"- **Reliability:** Cronbach's α = " + ", ".join(f"{d} {alpha[d]:.2f}" for d in DOMAINS) + ". "
        + (f"{' and '.join(CATEGORY_LABELS[d] for d in reliab_low.index)} fall below 0.70 — lower internal consistency and greater item heterogeneity — so a single domain average smooths over that heterogeneity." if len(reliab_low) else ""))
    add(f"- **PCA with parallel analysis:** {parallel['n_above_noise']} components exceed the 95th percentile of shuffled data. PC1 ({pct(pca['explained'][0], 1)}) is general agreement ({pct(pca['pc1_same_sign_share'])} of loadings share a sign; r = {pca['pc1_vs_mean_answer_r']:.2f} with mean answer). "
        f"**PC2 ({pct(pca['explained'][1], 1)}) is the largest substantive dimension:** {', '.join(short(c, text, 40) for c in pc2_pos)} versus {', '.join(short(c, text, 40) for c in pc2_neg)}. "
        f"It is only slightly larger than PC3 ({pct(pca['explained'][2], 1)}), so disagreement is multi-dimensional; we interpret PC2 because its loadings give a coherent descriptive contrast that is also reflected in several statement-level associations"
        + (f" — the significant negative edge{'s' if len(axis_edges) > 1 else ''} {', '.join(axis_edges)} join{'' if len(axis_edges) > 1 else 's'} its opposite poles." if axis_edges else ".")
        + " PCA and the correlations draw on the same response matrix, so this is descriptive alignment rather than independent evidence.")
    add(f"- **Imputation note:** PCA needs complete rows, so the {nc_after} residual missing cells ({pct(nc_after / df.size, 1)}) are imputed with column means before decomposing; a complete-case PCA (n = {pca_sens['n_complete']}) reproduces the same second axis (PC2 loading correlation r = {pca_sens['pc2_loading_r']:.2f}), so this choice does not drive the result.")
    add("")
    add("![Figure 5: (a) Correlation between respondents' domain scores. (b) Internal consistency of each domain.](figures/fig5_domain_structure.png)")
    add("")
    add(f"![Figure 6: Statements with the largest loadings on the second principal component ({pct(pca['explained'][1], 1)} of variance).](figures/fig6_pc2_loadings.png)")
    add("")

    add("### 3.6 Robustness")
    add("")
    k_q = k_sens.set_index("k")["modularity"]
    add(f"- **k:** modularity falls steadily from {k_q.iloc[0]:.2f} (k = {k_q.index[0]}) to {k_q.iloc[-1]:.2f} (k = {k_q.index[-1]}); partitions at other k agree only moderately with k = {K} (ARI {k_sens.loc[k_sens['k'] != K, 'ari_vs_k8'].min():.2f}–{k_sens.loc[k_sens['k'] != K, 'ari_vs_k8'].max():.2f}), consistent with weak communities.")
    add(f"- **Threshold:** edges drop from {int(thr_sens['edges'].iloc[0])} at 0.20 to {int(thr_sens['edges'].iloc[-1])} at 0.50. "
        f"As a simple all-null approximation, approximately {thr_row['expected_false_edges']:.0f} threshold-exceeding pairs would be expected by chance at |r| ≥ {CORR_THRESHOLD:.2f} (at 0.20, ~{thr_sens['expected_false_edges'].iloc[0]:.0f}); this is not an estimate of the actual false-discovery proportion. "
        f"{CORR_THRESHOLD:.2f} keeps most statements non-isolated. "
        f"Per-threshold p-values are computed at the median pairwise n = {median_pair_n}; pairwise sample sizes actually vary {min(pair_n_values)}–{max(pair_n_values)}.")
    add(f"- **Similarity choice:** raw and centred cosine give different respondent partitions (ARI = {bias['partition_ari']:.2f}), another sign that respondent communities depend on modelling choices.")
    add(f"- **Statement-level results hold:** negative edges have bootstrap intervals excluding zero, Spearman gives nearly the same correlations (r = {ev['pearson_spearman_r']:.2f}), and the statement pairs that define each core in Section 3.3 meet the ≥ {pct(cores['min_coassign'])} co-assignment threshold, with average pairwise co-clustering reported separately in the core table.")
    add("")
    add("![Figure 7: (a) Observed modularity vs shuffled-data null model. (b) Respondent network across k. (c) Statement network across thresholds (log scale).](figures/fig7_robustness.png)")
    add("")

    # 4. Discussion
    add("## 4. Results and Discussion")
    add("")
    add(f"1. **The class largely agrees.** {sv_consensus} of the 5 strongest-consensus statements are about Ethics or Environment, which are also the most internally consistent domains (α ≥ {alpha[['S', 'V']].min():.2f}).")
    add(f"2. **Disagreement has a leading axis, not a single one.** The largest substantive dimension contrasts regulation and traditional structure (stricter AI rules, exams, compulsory attendance) with open, flexible learning (online, project-based, collaborative). "
        f"It is supported by the significant negative edges, but several comparably small dimensions also exceed noise.")
    add(f"3. **The network shows diffuse structure rather than stable camps.** Communities are no stronger than in shuffled data, are unstable, and PC2 scores are unimodal. Respondents are embedded among neighbours with similar views (assortativity {assort_pc2:.2f}), but these neighbourhoods blend into each other rather than forming blocs. Averages alone could not show this; the null model makes it explicit.")
    add(f"4. **Domains shape opinions but do not contain them.** Within-domain links occur at {enrich['within_enrichment']:.1f}× the proportion expected under a uniform-edge baseline, yet most links cross domains and {len(mixed_cores)} of {len(cores['cores'])} reproducible clusters mix them"
        + (f" (regulation and rights: {', '.join(privacy_core['members'])})." if privacy_core else "."))
    add(f"5. **High-Neutral statements are weakly integrated.** The most-Neutral statements ({', '.join(uncertain)}) have no association strong enough to enter the network (strongest |r| = {min(iso_abs_uncertain):.2f}–{max(iso_abs_uncertain):.2f}); they are statements of weaker directional consensus and higher Neutral response shares rather than two-sided splits.")
    add("")
    add(f"**A dynamics lens (conceptual).** Our data are a single snapshot and cannot test dynamics; they can only suggest questions for opinion-dynamics models. The observed combination — neighbours aligned on the main opinion axis (assortativity {assort_pc2:.2f}) with global community structure indistinguishable from the shuffled baseline — is the kind of configuration that averaging-style models (e.g. DeGroot) are standard tools for exploring; whether repeated averaging would reinforce local alignment without consolidating global camps is a question for simulation, not for a snapshot. The E03-centred opposition on the PC2 axis is a candidate fault line that bounded-confidence or threshold models could investigate. This is framing for future work, not a prediction; no simulation is performed.")
    add("")
    add("### Limitations")
    add("")
    add(f"- Small class sample (n = {n}); strong agreement bias compresses the usable scale.")
    add("- Likert data are ordinal; Pearson r and cosine treat them as interval (Spearman gives near-identical results).")
    add(f"- {n_tests:,} pairwise tests: the actual false-discovery proportion of the reported edges is not estimated; only the strongest survive Bonferroni.")
    add("- Respondent communities and statement clusters outside the robust cores are not reproducible.")
    add("- η² for communities is optimistic (communities are built from the same answers); the shuffled baseline shows how much.")
    add("- Edges are associations, not causal or social ties; centrality is not real-world influence.")
    add("")
    add("### Conclusion")
    add("")
    add("The survey describes a broadly like-minded class whose disagreement is spread thinly over several dimensions, the largest contrasting structure and regulation with openness and flexibility, mostly through Education. "
        "Network analysis reveals robust cross-domain themes and negatively associated pairs of statements. Null models and bootstraps show where the network is informative (edges, local neighbourhoods, statement cores) and where it is not (respondent communities).")
    add("")

    # 5. Contributions
    add("## 5. Individual Contributions")
    add("")
    for name, task in CONTRIBUTIONS.items():
        add(f"- **{name}:** {task}")
    add("")

    report_path = Path("output/report.md")
    report_path.write_text("\n".join(L), encoding="utf-8")
    build_pdf(report_path, Path("output/report.pdf"))

    results = {
        "respondents_raw": n_all, "respondents_analysed": n, "statements": df.shape[1],
        "missing": missing, "dropped_respondents": dropped.reset_index().to_dict("records"),
        "response_style_flags": flags.to_dict("records"),
        "answer_shares": {"agree": share_agree, "neutral": share_neutral, "disagree": share_disagree},
        "respondent_network": {**stats1, "k": K},
        "statement_network": {**stats2, "threshold": CORR_THRESHOLD, "negative_edges": neg.to_dict("records"),
                              "signed_isolates": signed_isolates, "domain_enrichment": enrich, "rewired_null": rewired,
                              "fdr_edges": int(tests["fdr"].sum()), "bonferroni_edges": int(tests["bonferroni"].sum()),
                              "fdr_negative_pairs": neg_fdr_all, "fdr_negative_min_abs_r": round(neg_fdr_min_r, 4),
                              "fdr_negative_edges": fdr_neg[["u", "v", "r"]].round(4).to_dict("records"),
                              "isolate_strongest_r": {c: {"partner": p, "r": round(r, 4)} for c, (p, r) in iso_best.items()},
                              "pairwise_n_range": [int(min(pair_n_values)), int(max(pair_n_values))],
                              "median_pairwise_n": median_pair_n},
        "statement_robust_cores": {k: v for k, v in cores.items()},
        "communities": {"modularity": Q, "sizes": sizes.to_dict(), "partition": partition,
                        "null_model": {k: v for k, v in null.items() if k != "null_q"}, "stability": stability,
                        "eta_squared_by_domain": eta_domain.to_dict(), "top_eta_squared": eta.head(10).to_dict(),
                        "pc2_mean_by_community": pc2_by_comm.to_dict(), "pc2_eta_squared": pc2_eta,
                        "pc2_bimodality": pc2_bimodality, "assortativity_pc2": assort_pc2, "assortativity_agreement": assort_agree},
        "agreement_bias": bias,
        "k_sensitivity": k_sens.to_dict("records"),
        "threshold_sensitivity": thr_sens.to_dict("records"),
        "respondent_centrality_top10": cent1.head(10).reset_index(names="respondent").to_dict("records"),
        "statement_centrality": cent2.reset_index(names="statement").to_dict("records"),
        "statement_summary": summary.reset_index(names="statement").to_dict("records"),
        "domain_correlation": dom_corr.to_dict(), "cronbach_alpha": alpha.to_dict(),
        "design_choice_evidence": {k: v for k, v in ev.items() if k != "variance_rank"},
        "pca": {"explained": pca["explained"], "parallel_analysis": parallel,
                "pc1_vs_mean_answer_r": pca["pc1_vs_mean_answer_r"], "loadings": pca["loadings"].to_dict(),
                "imputation_sensitivity": pca_sens},
    }
    Path("output/results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("Done: output/report.pdf, output/results.json, output/figures/")


if __name__ == "__main__":
    main()
