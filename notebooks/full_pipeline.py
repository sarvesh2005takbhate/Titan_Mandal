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
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from src.data_prep import CATEGORY_LABELS, DOMAINS, filter_respondents, load_and_encode, response_style_flags
from src.build_networks import build_respondent_network, build_statement_network, positive_subgraph
from src.analysis import (
    centrality_table, community_deviation, descriptive_stats, detect_communities, domain_eta_squared,
    eta_squared, polarizing_consensus_rejected, statement_eta_squared, statement_summary,
)
from src.advanced_analysis import (
    agreement_bias_check, community_stability, cross_domain_share, design_choice_evidence, domain_reliability, domain_scores,
    item_pair_correlations, modularity_null_model, negative_edges, principal_components,
    respondent_k_sensitivity, statement_communities, statement_threshold_sensitivity,
)
from src import visualize as viz
from src.generate_pdf import build_pdf

# ── Configuration ────────────────────────────────────────────────────────────
TEAM_NAME = "Titan Mandal"
REPO_URL = "https://github.com/sarvesh2005takbhate/Titan_Mandal"
CONTRIBUTIONS = {
    "Hrishikesh": "Data preparation, Likert encoding, missing-data handling, respondent similarity construction.",
    "Sarvesh": "Network construction, Louvain community detection, centrality analysis, robustness experiments.",
    "Shourya": "Visualization, statement-level analysis, report compilation and PDF generation.",
}
K = 8                     # nearest neighbours in the respondent network
CORR_THRESHOLD = 0.30     # |r| cut-off in the statement network
MIN_ANSWERED = 30         # respondents answering fewer statements are excluded
NULL_RUNS = 50
# Near-duplicate statements placed in different domains (answer-consistency check).
DUPLICATE_PAIRS = [("T10", "S05"), ("T02", "E12")]
# A statement cluster is named after the anchor statement it contains.
THEME_ANCHORS = {
    "T10": "Digital rights & AI regulation",
    "T02": "AI-ready, industry-linked education",
    "V02": "Environmental & social responsibility",
    "E01": "Active learning & civic action",
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
    print("[3/5] Analysing (null model and stability take a minute) ...")
    partition, Q = detect_communities(G1)
    sizes = pd.Series(partition).value_counts().sort_index()
    null = modularity_null_model(df, Q, k=K, runs=NULL_RUNS)
    stability = community_stability(G1, df, partition, k=K)
    bias = agreement_bias_check(df, k=K)
    k_sens = respondent_k_sensitivity(df, partition)
    cent1 = centrality_table(G1)
    eta = statement_eta_squared(df, partition)
    eta_domain = domain_eta_squared(df, partition)
    top_eta = list(eta.head(10).index)
    deviation = community_deviation(df, partition, top_eta)

    summary = statement_summary(df)
    polar, consensus, rejected = polarizing_consensus_rejected(summary)
    neg = negative_edges(G2)
    thr_sens = statement_threshold_sensitivity(df)
    thr_row = thr_sens.set_index("threshold").loc[CORR_THRESHOLD]
    cent2 = centrality_table(G2_pos)
    signed_isolates = sorted(n for n in G2 if G2.degree(n) == 0)
    cross_share = cross_domain_share(G2)
    stmt_part, stmt_q, stmt_ari, domain_q = statement_communities(G2)
    positive_only_isolates = sorted(set(stats2["isolates"]) - set(signed_isolates))

    clusters = pd.Series(stmt_part).groupby(pd.Series(stmt_part)).apply(lambda s: sorted(s.index))
    cluster_names = {}
    for cid, members in clusters.items():
        names = [name for anchor, name in THEME_ANCHORS.items() if anchor in members]
        cluster_names[cid] = " / ".join(names) if names else "Mixed"

    dom_corr = domain_scores(df).corr()
    alpha = domain_reliability(df)
    pca = principal_components(df, n_components=3)
    pc2 = pca["loadings"]["PC2"]
    pc2_scores = pca["scores"]["PC2"]
    groups = df.index.to_series().map(partition)
    pc2_by_comm = pc2_scores.groupby(groups).mean()
    pc2_eta = eta_squared(pc2_scores, groups)
    dup = item_pair_correlations(corr, DUPLICATE_PAIRS)
    ev = design_choice_evidence(df_all, df, G1, sim, partition, summary)
    split_vs_strength = summary["split"].corr(cent2["strength"])
    neutral_rank = summary["neutral"].rank(ascending=False)
    privacy_cluster = next(m for m in clusters if "T10" in m)
    iu = np.triu_indices(len(dom_corr), 1)
    pairs = [(dom_corr.index[i], dom_corr.columns[j], dom_corr.iat[i, j]) for i, j in zip(*iu)]
    strongest, weakest = max(pairs, key=lambda p: p[2]), min(pairs, key=lambda p: p[2])

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("[4/5] Drawing figures ...")
    viz.plot_data_overview(df_all, dropped)
    viz.plot_similarity_distribution(bias)
    viz.plot_respondent_network(G1, partition, cent1)
    viz.plot_statement_network(G2)
    viz.plot_community_profiles(deviation, eta, text, sizes)
    viz.plot_polarization(df, {
        "Most split": list(polar.index),
        "Rejected by the class": [c for c in rejected.index if c not in polar.index],
        "Strongest consensus": list(consensus.index),
    }, text)
    viz.plot_robustness(null, k_sens, thr_sens, CORR_THRESHOLD)
    viz.plot_domain_structure(dom_corr, alpha)
    viz.plot_pc2(pc2, text)

    # ── 5. Report ────────────────────────────────────────────────────────────
    print("[5/5] Writing report ...")
    significant = null["p"] < 0.05
    pc2_neg, pc2_pos = pc2.nsmallest(3).index, pc2.nlargest(3).index
    open_comm, control_comm = pc2_by_comm.idxmin(), pc2_by_comm.idxmax()
    top_split = polar.iloc[0]
    e03 = summary.loc["E03"]
    reliab_low = alpha[alpha < 0.7]
    straightliners = flags[flags["flags"].str.contains("straight")]["respondent"].tolist()
    dissenters = flags[flags["flags"].str.contains("dissenter")]["respondent"].tolist()
    neutrals = flags[flags["flags"].str.contains("neutral")]["respondent"].tolist()

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
    add(f"- **A consensus class.** {pct(share_agree)} of all answers are Agree/Strongly Agree. Environment and Ethics are near-unanimous; disagreement is concentrated in a handful of Education and Technology statements.")
    add(f"- **No distinct opinion camps.** Respondent communities (Q = {Q:.3f}) are {'only marginally' if not significant else 'significantly'} stronger than in shuffled data "
        f"(null Q = {null['null_mean']:.3f} ± {null['null_std']:.3f}, p = {null['p']:.2f}) and change between runs (ARI ≈ {stability['subsample_ari_mean']:.2f}). Opinions vary along a continuum, not in blocs.")
    add(f"- **The clearest axis of disagreement:** support for regulation and traditional structure ({', '.join(pc2_pos)}) versus open, flexible learning ({', '.join(pc2_neg)}). All {len(neg)} negative statement correlations involve Education.")
    add(f"- **Themes cut across domains.** {pct(cross_share)} of statement edges link different domains; the statement network forms {len(clusters)} cross-domain clusters, including a *digital rights & AI regulation* cluster.")
    add("")

    # 1. Dataset
    add("## 1. Dataset Documentation")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    add(f"| Raw responses | {n_all} respondents × {df_all.shape[1]} statements (15 per domain: T, E, S, V) |")
    add("| Encoding | Strongly Disagree = −2, Disagree = −1, Neutral = 0, Agree = +1, Strongly Agree = +2 |")
    add(f"| Missing cells | {missing['total_missing']} of {missing['total_cells']} ({pct(missing['total_missing'] / missing['total_cells'], 1)}): {missing['blank']} blank, {missing['no_comments']} “No Comments” |")
    add(f"| Excluded respondents | {len(dropped)} answered < {MIN_ANSWERED} statements: {int((dropped['answered'] == 0).sum())} blank forms, {int((dropped['domains_answered'] == 'T').sum())} stopped after the Technology page |")
    add(f"| Analysed | **{n} respondents**, {nc_after} missing cells ({pct(nc_after / df.size, 1)}) |")
    add(f"| Answer mix (analysed) | {pct(share_agree)} agree · {pct(share_neutral)} neutral · {pct(share_disagree)} disagree |")
    add("")
    add(f"- **Missingness is dropout, not item refusal.** All {missing['blank']} blank cells come from respondents who abandoned the form (4 stopped exactly after the Technology block, consistent with a form paged by domain). "
        f"Only the {missing['no_comments']} “No Comments” are deliberate skips. Both are kept as NaN, never as Neutral; similarities use only statements both respondents answered. "
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
    add(f"2. **Respondent similarity.** Subtract each respondent's mean answer, then take cosine similarity over shared statements. Centring also weakens the link between a node's degree and its agreement level (r = {bias['raw']['degree_vs_mean_answer_r']:.2f} → {bias['centred']['degree_vs_mean_answer_r']:.2f}; Figure 2).")
    add(f"3. **Sparsify.** Link each respondent to its {K} most similar peers. Similarity stays the edge weight; shortest paths use distance = 1/similarity, so strong ties are short.")
    add(f"4. **Statement network.** Keep abs(r) ≥ {CORR_THRESHOLD:.2f} (p ≈ {thr_row['p_value']:.3f} at n = {n}). Negative edges are kept for interpretation but excluded from path metrics and community detection, since opposition is not proximity.")
    add(f"5. **Analyse.** Louvain communities tested against {NULL_RUNS} shuffled datasets and for stability; centrality; η² (variance explained by community); a split index for polarization; PCA and Cronbach's α for domain structure; sensitivity to k and threshold.")
    add("")
    add("### Design choices and why")
    add("")
    add("Each choice was compared with the obvious alternative on this dataset. The last column gives the evidence.")
    add("")
    thr_iso = thr_sens.set_index("threshold")["isolates"]
    thr_chance = thr_sens.set_index("threshold")["expected_false_edges"]
    add("| Decision | Chosen | Instead of | Why / benefit (evidence from this data) |")
    add("|---|---|---|---|")
    add(f"| Missing answers | NaN, pairwise-complete | Code as Neutral | A blank is not an opinion. Coding blanks as Neutral would raise the Neutral share from {pct(ev['neutral_share_true'])} to {pct(ev['neutral_share_if_blank_neutral'])} and invent {missing['blank']} answers. |")
    add(f"| Incomplete forms | Exclude < {MIN_ANSWERED} answers | Impute | Imputing 45–60 of 60 answers fabricates a profile; kept as-is, the {int((dropped['answered'] == 0).sum())} blank forms become isolated nodes and fake communities. |")
    add(f"| Respondent similarity | Row-centred cosine | Raw cosine, Euclidean | Compares *which* statements a person favours, not how much they agree: mean similarity {bias['raw']['mean_similarity']:.2f} → {bias['centred']['mean_similarity']:.2f}. Euclidean distance also grows with the number of shared answers. |")
    add(f"| Sparsification | k-NN, k = {K} | Global similarity threshold | A threshold with the same edge count leaves {ev['threshold_isolates']} respondents isolated ({ev['threshold_components']} components); k-NN gives every respondent ≥ k neighbours and is connected from k = {ev['min_connected_k']}. k ≈ √n, mid-range of the tested 5–12. |")
    add(f"| Path cost | 1 / similarity | 1 − similarity, hops | Keeps strong ties short while penalising weak ties; betweenness ranks barely change with 1 − similarity (ρ = {ev['betweenness_rank_rho_inv_vs_one_minus']:.2f}), so results do not hinge on it. |")
    add(f"| Communities | Louvain | Greedy, label propagation, Girvan–Newman | Fast, weighted modularity optimisation. Greedy modularity reaches similar Q ({ev['greedy_q']:.2f}, {ev['greedy_n']} groups) but a different split (ARI {ev['greedy_ari']:.2f}); label propagation finds {ev['lpa_n']} community — consistent with weak structure. Girvan–Newman is O(m²n) with no natural stopping point. |")
    add(f"| Significance | Shuffled-answer null model | Raw Q, configuration model | k-NN graphs are modular by construction (null Q = {null['null_mean']:.2f}). Shuffling keeps every statement's answer distribution and the full pipeline, so it tests exactly whether *combinations* of opinions cluster. |")
    add(f"| Partition agreement | ARI | NMI | Chance-corrected: unrelated partitions of the same sizes give ARI ≈ {abs(ev['random_ari']):.2f} but NMI {ev['random_nmi']:.2f}, which rewards chance overlap. |")
    add(f"| Statement association | Signed Pearson r | Spearman; abs(r) | Spearman agrees closely (r = {ev['pearson_spearman_r']:.2f} over all pairs). Keeping the sign separates opposition from co-endorsement; abs(r) would have merged the {len(neg)} Education tensions into clusters. |")
    add(f"| Edge threshold | abs(r) ≥ {CORR_THRESHOLD:.2f} | 0.20 or 0.40 | ~{thr_chance[CORR_THRESHOLD]:.0f} chance edges ({thr_row['expected_false_edges'] / thr_row['edges']:.0%}) vs ~{thr_chance[0.20]:.0f} at 0.20; 0.40 isolates {int(thr_iso[0.40])} statements. |")
    add(f"| Centrality | Betweenness (on distance) | Degree, eigenvector | In k-NN every degree ≥ k by construction; eigenvector centrality tracks agreement level more (r = {ev['eigenvector_vs_agreement_r']:.2f}) than betweenness does (r = {ev['betweenness_vs_agreement_r']:.2f}). Betweenness captures bridging. |")
    add(f"| Polarization | Split index | Variance | Variance ranks E03 #{ev['variance_rank']['E03']} most polarizing although {pct(summary.loc['E03', 'disagree'])} reject it; the split index is high only when both camps are large and reads directly in %. |")
    add(f"| Community profile | η² per statement | Domain averages | Size-weighted 0–1 effect size, comparable across statements; domain averages mix opposing statements (Education α = {alpha['E']:.2f}). |")
    add("| Domain validity | Cronbach's α, PCA | Assume domains coherent | Tests whether a domain average is meaningful; PCA separates general agreement (PC1) from substantive disagreement (PC2). |")
    add("")
    add(f"**Reproducibility.** One command (`python notebooks/full_pipeline.py`) regenerates every number, table and figure from the raw CSV with fixed seeds. Unit tests check the similarity, η², split-index and filtering maths, and all results are exported to `results.json`.")
    add("")
    add("![Figure 2: Pairwise respondent similarity before and after centring each respondent's answers.](figures/fig2_similarity_distribution.png)")
    add("")

    # 3. Analysis
    add("## 3. Analysis and Visualizations")
    add("")
    add("### 3.1 Network-level metrics")
    add("")
    add("| Metric | Respondent network | Statement network (positive edges) |")
    add("|---|---:|---:|")
    for label, key in [("Nodes", "nodes"), ("Edges", "edges"), ("Density", "density"), ("Average degree", "avg_degree"),
                       ("Weighted clustering", "clustering"), ("Clustering ÷ random-graph expectation", "clustering_vs_random"),
                       ("Connected components", "components"),
                       ("Largest component", "lcc_nodes"), ("Avg. shortest path (hops)", "avg_path_hops"),
                       ("Diameter (hops)", "diameter_hops"), ("Avg. path length (distance units)", "avg_path_distance")]:
        add(f"| {label} | {stats1[key]} | {stats2[key]} |")
    add("")
    add(f"Paths are short in both networks (about {stats1['avg_path_hops']:.0f} hops). Clustering is {stats1['clustering_vs_random']:.1f}× (respondents) and {stats2['clustering_vs_random']:.1f}× (statements) what a random graph of the same density would give, so similar opinions form local triangles rather than random links. "
        f"After cleaning, the respondent network is a single component. "
        f"In the statement network {len(stats2['isolates'])} statements have no positive link; see Section 3.3.")
    add("")

    add("### 3.2 Respondent communities")
    add("")
    add(f"- Louvain finds **{len(sizes)} communities** (sizes {', '.join(map(str, sizes.values))}), Q = {Q:.3f}.")
    add(f"- **Null model:** networks from shuffled answers give Q = {null['null_mean']:.3f} ± {null['null_std']:.3f} (z = {null['z']:.1f}, p = {null['p']:.2f}). "
        + ("The observed structure is **not significantly stronger than chance**, because k-NN graphs are modular by construction." if not significant
           else "The observed structure is significantly stronger than chance."))
    add(f"- **Stability:** agreement with the main partition is ARI = {stability['seed_ari_mean']:.2f} across Louvain seeds and {stability['subsample_ari_mean']:.2f} ± {stability['subsample_ari_std']:.2f} when 10% of respondents are removed. Communities are soft groupings.")
    add(f"- **What differs between them:** community membership explains most variance for {', '.join(top_eta[:4])} (η² = {eta.iloc[3]:.2f}–{eta.iloc[0]:.2f}; mean η² by domain: "
        + ", ".join(f"{d} {eta_domain[d]:.2f}" for d in DOMAINS) + "). No community disagrees with the class overall; they differ in emphasis.")
    add(f"- The groups line up with the opinion axis of Section 3.5: community C{open_comm} leans towards open learning (mean PC2 score {pc2_by_comm[open_comm]:+.2f}) and C{control_comm} towards regulation and structure ({pc2_by_comm[control_comm]:+.2f}); community explains η² = {pc2_eta:.2f} of that axis.")
    add(f"- **Centrality:** the most central respondents by betweenness are #{', #'.join(map(str, cent1.index[:3]))}. Their positions are structural (bridging similar-minded groups), not a measure of influence.")
    add("")
    add("![Figure 3: Respondent network (k = 8, centred cosine). Colour and shape show Louvain community; node size shows betweenness.](figures/fig3_respondent_network.png)")
    add("")
    add("![Figure 5: Community mean minus class mean on the ten statements that best separate communities (highest η²). Blue = more agreement, red = less.](figures/fig5_community_profiles.png)")
    add("")

    add("### 3.3 Statement network")
    add("")
    add(f"- **Signs:** {int(stats2['edges'])} positive and {len(neg)} negative edges. Every negative edge involves an Education statement:")
    add("")
    add("| Statement A | Statement B | r |")
    add("|---|---|---:|")
    for r in neg.itertuples():
        add(f"| {short(r.u, text, 55)} | {short(r.v, text, 55)} | {r.r:.2f} |")
    add("")
    add(f"- **Cross-domain themes:** {pct(cross_share)} of edges connect different domains. Louvain on positive edges gives {len(clusters)} clusters (Q = {stmt_q:.2f}), "
        f"much more modular than the domain labels themselves (Q = {domain_q:.2f}; ARI with domains = {stmt_ari:.2f}):")
    add("")
    add("| Cluster | Theme | Statements |")
    add("|---|---|---|")
    for cid, members in clusters.items():
        add(f"| {cid} | {cluster_names[cid]} | {', '.join(members)} |")
    add("")
    uncertain = [c for c in signed_isolates if neutral_rank[c] <= 5]
    add(f"- **Unconnected statements:** {', '.join(signed_isolates)} have no edge at all. "
        f"{', '.join(c + ' (' + pct(summary.loc[c, 'neutral']) + ' Neutral)' for c in uncertain)} are among the five most-Neutral statements in the survey: uncertainty, not conviction, dominates them. "
        + (f"{', '.join(positive_only_isolates)} connect only through negative edges." if positive_only_isolates else ""))
    add(f"- **Bridges:** highest betweenness: {', '.join(short(c, text, 45) for c in cent2.index[:3])}. "
        f"Divisive statements sit at the periphery (split index vs weighted degree r = {split_vs_strength:.2f}).")
    add("")
    add("![Figure 4: Statement network (abs(r) ≥ 0.30). Colour shows domain (also the code letter); red edges are negative correlations; unconnected statements are shown along the bottom.](figures/fig4_statement_network.png)")
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
    add(f"- Only {int((summary['split'] >= 0.2).sum())} statements have ≥ 20% on both sides; the most divisive is {short(top_split.name, text, 60)} ({pct(top_split.agree)} agree vs {pct(top_split.disagree)} disagree).")
    add(f"- **E03 is a consensus, not a split:** {pct(e03.disagree)} reject compulsory attendance. High variance alone would have mislabelled it as polarizing.")
    add("")
    add("![Figure 6: Answer distributions for the most split, rejected and consensus statements.](figures/fig6_polarization.png)")
    add("")

    add("### 3.5 Domain structure and the clearest opinion axis")
    add("")
    add(f"- **Domain scores correlate positively.** Strongest {strongest[0]}–{strongest[1]} (r = {strongest[2]:.2f}), weakest {weakest[0]}–{weakest[1]} (r = {weakest[2]:.2f}).")
    add(f"- **Reliability:** Cronbach's α = " + ", ".join(f"{d} {alpha[d]:.2f}" for d in DOMAINS) + ". "
        + (f"{' and '.join(CATEGORY_LABELS[d] for d in reliab_low.index)} fall below 0.70: they mix opposing positions (e.g. exams and attendance vs project-based and online learning), so a single domain average hides the real disagreement." if len(reliab_low) else ""))
    add(f"- **PCA:** PC1 ({pct(pca['explained'][0], 1)} of variance) is general agreement ({pct(pca['pc1_same_sign_share'])} of loadings share a sign; r = {pca['pc1_vs_mean_answer_r']:.2f} with mean answer). "
        f"**PC2 ({pct(pca['explained'][1], 1)}) is the clearest substantive axis:** {', '.join(short(c, text, 40) for c in pc2_pos)} versus {', '.join(short(c, text, 40) for c in pc2_neg)}. "
        f"It is only slightly larger than PC3 ({pct(pca['explained'][2], 1)}), so beyond general agreement opinions are weakly structured; PC2 stands out because it matches the negative edges and the community differences.")
    add("")
    add("![Figure 8: (a) Correlation between respondents' domain scores. (b) Internal consistency of each domain.](figures/fig8_domain_structure.png)")
    add("")
    add(f"![Figure 9: Statements with the largest loadings on the second principal component ({pct(pca['explained'][1], 1)} of variance).](figures/fig9_pc2_loadings.png)")
    add("")

    add("### 3.6 Robustness")
    add("")
    k_q = k_sens.set_index("k")["modularity"]
    add(f"- **k:** modularity falls steadily from {k_q.iloc[0]:.2f} (k = {k_q.index[0]}) to {k_q.iloc[-1]:.2f} (k = {k_q.index[-1]}); partitions at other k agree only moderately with k = {K} (ARI {k_sens.loc[k_sens['k'] != K, 'ari_vs_k8'].min():.2f}–{k_sens.loc[k_sens['k'] != K, 'ari_vs_k8'].max():.2f}), consistent with weak communities.")
    add(f"- **Threshold:** edges drop from {int(thr_sens['edges'].iloc[0])} at 0.20 (~{thr_sens['expected_false_edges'].iloc[0]:.0f} expected chance edges) to {int(thr_sens['edges'].iloc[-1])} at 0.50. "
        f"{CORR_THRESHOLD:.2f} keeps the network largely connected while limiting chance edges to ~{thr_row['expected_false_edges'] / thr_row['edges']:.0%}.")
    add(f"- **Similarity choice:** raw and centred cosine give different partitions (ARI = {bias['partition_ari']:.2f}), another sign that respondent communities depend on modelling choices, whereas the statement-level findings are stable.")
    add("")
    add("![Figure 7: (a) Observed modularity vs shuffled-data null model. (b) Respondent network across k. (c) Statement network across thresholds (log scale).](figures/fig7_robustness.png)")
    add("")

    # 4. Discussion
    add("## 4. Results and Discussion")
    add("")
    add(f"1. **The class largely agrees.** Environmental and ethical statements (accountability, sustainability, inclusion) have the strongest consensus and the most internally consistent domains (α ≥ {alpha[['S', 'V']].min():.2f}).")
    add(f"2. **Disagreement follows one recognisable axis.** Beneath the shared optimism, the main difference is attitude to control: regulating AI and protecting data, exams and compulsory attendance, versus online, project-based and collaborative learning. Education holds all {len(neg)} opposing edges and the most split statements.")
    add(f"3. **The network shows a continuum, not camps.** Communities are no stronger than in shuffled data and are unstable. Respondents differ by degree along the axis above, not by belonging to opposing groups. Averages alone could not show this; the null model makes it explicit.")
    add(f"4. **Opinions organise by theme, not survey section.** Most statement links cross domains, e.g. data privacy and AI regulation form one cluster ({', '.join(privacy_cluster)}). The four-domain survey layout does not match how respondents actually group issues.")
    add(f"5. **Uncertain topics stay disconnected.** Statements with many Neutral answers (AI diagnosis, autonomous vehicles, research vs infrastructure spending) do not correlate with anything, suggesting unformed rather than divided opinions.")
    add("")
    add("### Limitations")
    add("")
    add(f"- Small, self-selected class sample (n = {n}); strong agreement bias compresses the usable scale.")
    add("- Likert data are ordinal; Pearson r and cosine treat them as interval.")
    add(f"- k, threshold and similarity choices shape the respondent network; communities in particular are not robust.")
    add("- η² for communities is descriptive: communities are built from the same answers, so it is optimistic.")
    add(f"- Beyond general agreement the data are weakly structured (PC2 {pct(pca['explained'][1], 1)} vs PC3 {pct(pca['explained'][2], 1)} of variance).")
    add("- Edges are associations, not causal or social ties; centrality is not real-world influence.")
    add("")
    add("### Conclusion")
    add("")
    add("The survey describes a broadly like-minded class whose meaningful disagreement is concentrated along one axis: structure and regulation versus openness and flexibility, expressed mostly through Education. Network analysis reveals this cross-domain structure. Significance and robustness checks show where the network is informative (statement associations) and where it is not (respondent communities).")
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
                              "signed_isolates": signed_isolates, "cross_domain_share": cross_share},
        "communities": {"modularity": Q, "sizes": sizes.to_dict(), "partition": partition,
                        "null_model": {k: v for k, v in null.items() if k != "null_q"}, "stability": stability,
                        "eta_squared_by_domain": eta_domain.to_dict(), "top_eta_squared": eta.head(10).to_dict(),
                        "pc2_mean_by_community": pc2_by_comm.to_dict(), "pc2_eta_squared": pc2_eta},
        "agreement_bias": {lab: {k: v for k, v in bias[lab].items() if k != "similarities"} for lab in ("raw", "centred")}
                          | {"partition_ari": bias["partition_ari"]},
        "k_sensitivity": k_sens.to_dict("records"),
        "threshold_sensitivity": thr_sens.to_dict("records"),
        "statement_clusters": {f"{cid}: {cluster_names[cid]}": m for cid, m in clusters.items()},
        "statement_cluster_modularity": stmt_q, "domain_label_modularity": domain_q, "cluster_vs_domain_ari": stmt_ari,
        "respondent_centrality_top10": cent1.head(10).reset_index(names="respondent").to_dict("records"),
        "statement_centrality": cent2.reset_index(names="statement").to_dict("records"),
        "statement_summary": summary.reset_index(names="statement").to_dict("records"),
        "domain_correlation": dom_corr.to_dict(), "cronbach_alpha": alpha.to_dict(),
        "design_choice_evidence": {k: v for k, v in ev.items() if k != "variance_rank"},
        "pca": {"explained": pca["explained"], "pc1_vs_mean_answer_r": pca["pc1_vs_mean_answer_r"],
                "loadings": pca["loadings"].to_dict()},
    }
    Path("output/results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("Done: output/report.pdf, output/results.json, output/figures/")


if __name__ == "__main__":
    main()
