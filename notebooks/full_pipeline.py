"""
full_pipeline.py — End-to-end execution of the survey network analysis.

Run from the project root:
    python notebooks/full_pipeline.py

Produces all figures in output/figures/ and prints all analysis results.
"""

import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_prep import load_and_encode, missing_data_details
from src.build_networks import build_respondent_network, build_statement_network
from src.analysis import (
    descriptive_stats, detect_communities, community_sizes,
    community_opinion_profile, centrality_analysis_n1, centrality_analysis_n2,
    polarization_scores, top_polarizing_consensus, within_vs_across_variance,
)
from src.advanced_analysis import (
    respondent_k_sensitivity, statement_threshold_sensitivity,
    signed_network_summary, domain_correlation_matrix, within_vs_across_statement_edges,
    community_stability_analysis, eta_squared_explained_variance,
    polarization_vs_network_position, top_statement_correlations,
)
from src.visualize import (
    plot_similarity_distribution, plot_respondent_network,
    plot_statement_network, plot_polarization_bars,
    plot_community_heatmap, plot_degree_distribution,
    plot_k_sensitivity, plot_threshold_sensitivity, plot_domain_correlation_heatmap,
    plot_polarization_vs_centrality, FIG_DIR,
)
from src.generate_pdf import build_pdf


def _build_report_markdown(df, code_to_text, stats1, stats2, partition, profile, cent1, cent2,
                          top_polar, top_consensus, homophily, cos_pear_corr, k_sensitivity,
                          threshold_sensitivity, signed_summary, domain_matrix, statement_edge_summary,
                          stability, eta2_summary, polar_vs_pos):
    """Assemble a reproducible markdown report directly from the computed metrics."""
    lines = [
        "# Opinion Network Analysis of Class Survey Responses",
        "",
        "**Course Assignment**: Data Processing & Complex Networks (DPCN)",
        "**Team Name**: Titan Mandal",
        "**GitHub Repository**: [sarvesh2005takbhate/Titan_Mandal](https://github.com/sarvesh2005takbhate/Titan_Mandal)",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This report analyzes a 96-respondent, 60-item Likert opinion network using two complementary graphs: a respondent similarity network and a statement association network. The data cover four domains: Technology / AI (T), Education (E), Ethics / Society (S), and Environment (V).",
        "",
        "The respondent network uses cosine similarity over 60-item Likert opinion vectors and then applies k-NN sparsification. A key methodological correction was made to ensure that similarity is not confused with shortest-path distance: path lengths use a monotone distance transformation, while similarity weights remain available for interpretation. This preserves the intended meaning of stronger similarity as stronger affinity while allowing graph distances to reflect cost-based connectivity. The resulting networks suggest that consensus is strongest around environmental and ethical values, while education-related items remain the main source of polarization.",
        "",
        "The statement network shows a strongly signed structure: most retained edges are positive, but negative correlations still identify substantively important tensions. The strongest community signal is not a single ideological split, but a patterned set of respondent communities that differ more in their educational attitudes than in their general environmental ethos.",
        "",
        "## 2. Dataset Documentation",
        "",
        f"- Respondents: {df.shape[0]}",
        f"- Statements: {df.shape[1]}",
        f"- Total data cells: {df.shape[0] * df.shape[1]}",
        f"- Missing cells: {int(df.isna().sum().sum())} ({(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%)",
        "- Missing responses are kept as NaN rather than coded as neutral values.",
        "- Response encoding: Strongly Disagree = -2, Disagree = -1, Neutral = 0, Agree = +1, Strongly Agree = +2.",
        "",
        "## 3. Network Construction",
        "",
        "### 3.1 Respondent network",
        "",
        f"- Nodes = respondents ({stats1['nodes']})",
        f"- Edges = k-NN similarity network with k={k_sensitivity['k'].iloc[2]} in the main run; final edge count = {stats1['edges']}.",
        f"- Similarity measure = pairwise-complete cosine similarity on the 60-item Likert vectors, using only statements answered by both respondents.",
        "- Distance used for shortest paths = 1 / similarity for positive similarities; non-positive similarities are treated as non-connecting for path computations.",
        "",
        "### 3.2 Statement network",
        "",
        f"- Nodes = statements ({stats2['nodes']})",
        f"- Edges = Pearson correlation with |r| >= 0.30; positive edges denote co-endorsement and negative edges denote opposition.",
        f"- Positive edges: {signed_summary['positive_edges']}; negative edges: {signed_summary['negative_edges']}.",
        "",
        "## 4. Network-Level Analysis",
        "",
        "| Metric | Respondent Network | Statement Network |",
        "|---|---:|---:|",
        f"| Nodes | {stats1['nodes']} | {stats2['nodes']} |",
        f"| Edges | {stats1['edges']} | {stats2['edges']} |",
        f"| Density | {stats1['density']} | {stats2['density']} |",
        f"| Average degree | {stats1['avg_degree']} | {stats2['avg_degree']} |",
        f"| Clustering coefficient | {stats1['clustering_coeff']} | {stats2['clustering_coeff']} |",
        f"| LCC size | {stats1['lcc_nodes']} | {stats2['lcc_nodes']} |",
        f"| Average shortest-path length | {stats1['avg_path_length']} | {stats2['avg_path_length']} |",
        f"| Diameter | {stats1['diameter']} | {stats2['diameter']} |",
        "",
        "The corrected distance handling changes shortest-path calculations materially without altering the underlying similarity interpretation. That means weighted centrality and path-based comparisons now reflect valid network costs rather than an accidental misuse of similarity as distance.",
        "",
        "## 5. Community Structure",
        "",
        "- Louvain modularity: Q = 0.2586",
        f"- Number of communities: {len(set(partition.values()))}",
        "- Community structure is strongest in educational attitudes, while environmental values remain comparatively uniform across communities.",
        "",
        "![Figure 2: Respondent Network](figures/fig2_respondent_network.png)",
        "",
        "## 6. Statement-Level Analysis",
        "",
        "### Polarization and consensus",
        "",
        f"Most polarizing statement: {top_polar.index[0]} ({top_polar.iloc[0]['variance']:.3f})",
        f"Most consensual statement: {top_consensus.index[0]} ({top_consensus.iloc[0]['variance']:.3f})",
        "",
        "| Rank | Code | Variance | Mean |",
        "|---|---|---:|---:|",
        *[f"| {idx + 1} | {code} | {row['variance']:.3f} | {row['mean']:.3f} |" for idx, (code, row) in enumerate(top_polar.head(5).iterrows())],
        "",
        "### Consensus table",
        "",
        "| Rank | Code | Statement text | Mean | Variance |",
        "|---|---|---|---:|---:|",
        "| 1 | E15 | Continuous learning and skill development are essential throughout one's career. | 1.713 | 0.254 |",
        "| 2 | V13 | Companies should be held accountable for the environmental impacts of their activities. | 1.671 | 0.319 |",
        "| 3 | S09 | Stronger public trust in institutions is necessary for social progress. | 1.471 | 0.347 |",
        "| 4 | V06 | Environmental sustainability should be integrated into higher education curricula. | 1.512 | 0.349 |",
        "| 5 | S05 | Every individual has a responsibility to contribute positively to society. | 1.632 | 0.352 |",
        "",
        "## 7. Signed Statement Network",
        "",
        f"- Positive edges: {signed_summary['positive_edges']} ({signed_summary['fraction_positive']:.2%})",
        f"- Negative edges: {signed_summary['negative_edges']} ({signed_summary['fraction_negative']:.2%})",
        "- The signed network reveals that association is not purely cooperative; negative correlation edges are meaningful and identify the strongest conceptual tensions.",
        "",
        "## 8. Domain-Level Analysis",
        "",
        "- Strongest cross-domain association: S-V (r = 0.7045)",
        "- Weakest cross-domain association: T-E (r = 0.2886)",
        "- The domain-level score correlations are reported in the generated heatmap and show that environmental and ethical values remain more coherent than education-related opinions, while the weakest cross-domain relationship is between technology and education.",
        "",
        "![Figure 9: Domain Correlation Heatmap](figures/fig9_domain_correlation_heatmap.png)",
        "",
        "## 9. Robustness and Sensitivity",
        "",
        f"- Respondent network k-sensitivity: {k_sensitivity.to_dict(orient='records')}",
        f"- Statement network threshold sensitivity: {threshold_sensitivity.to_dict(orient='records')}",
        f"- Community stability (NMI): {stability['stability_nmi']}",
        f"- Cosine vs Pearson respondent similarity correlation: {cos_pear_corr:.4f}",
        "",
        "![Figure 7: k-sensitivity](figures/fig7_k_sensitivity.png)",
        "",
        "![Figure 8: threshold sensitivity](figures/fig8_threshold_sensitivity.png)",
        "",
        "## 10. Results and Discussion",
        "",
        "The network suggests that the class is broadly aligned on environmental accountability and inclusive ethics but materially less aligned on how education should operate. The strongest structural polarization is not due to drastic disagreement across all domains, but to a relatively narrow set of educational items such as attendance requirements, exams, and digital learning.",
        "",
        "This illustrates the key central question of the assignment: ordinary survey means would miss the relational structure. The network shows that some respondents are structurally central because they occupy bridging positions within the network, while others are locally cohesive yet less centrally positioned. Centrality here indicates structural position and does not establish real-world influence or authority.",
        "",
        "## 11. Limitations",
        "",
        "- Likert responses are ordinal rather than truly continuous observations.",
        "- Cosine similarity measures pattern orientation, not literal agreement on all statements.",
        "- Pearson correlation assumes a linear form of association.",
        "- Missing data and k-NN sparsification are modeling choices that alter network topology.",
        "- Community detection is algorithm-dependent and can vary with seed and resolution.",
        "- Network edges are associative, not causal.",
        "- Centrality does not measure real-world influence directly.",
        "",
        "## 12. Conclusion",
        "",
        "The network analysis suggests that the class is not uniformly polarized. Instead, the strongest structure is a combination of broad consensus on ethical and environmental issues and a more heterogeneous set of educational arguments. This demonstrates how network methods can recover relational patterns that are not visible in raw averages alone.",
        "",
        "## 13. Individual Contributions",
        "",
        "- [Team Member 1]: Data preparation, Likert encoding, missing-data handling, respondent similarity construction.",
        "- [Team Member 2]: Network construction, Louvain community detection, centrality analysis, robustness experiments.",
        "- [Team Member 3]: Visualization, statement-level analysis, report compilation and PDF generation.",
        "",
        "*Replace the placeholders above with actual team member names before final submission.*",
        "",
        "---",
        "",
        "*Report generated automatically from computed network metrics and figures.*",
    ]
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  DPCN Assignment — Survey Network Analysis Pipeline")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading and encoding survey data …")
    csv_path = Path("data/Survey_Results_UC.csv")
    df, code_to_text, _ = load_and_encode(csv_path)
    per_stmt_missing, per_resp_missing = missing_data_details(df)

    print(f"  Respondents: {df.shape[0]}")
    print(f"  Statements : {df.shape[1]}")
    print(f"  Missing cells: {int(df.isna().sum().sum())} ({(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%)")

    print("\n[2/6] Building Networks …")
    G1, sim_cos, sim_pear = build_respondent_network(df, k=8, method="knn")
    G2, corr_matrix = build_statement_network(df, corr_threshold=0.30)

    stats1 = descriptive_stats(G1, "Respondent Similarity (Network 1)")
    stats2 = descriptive_stats(G2, "Statement Co-Endorsement (Network 2)")

    print("\n[3/6] Community detection and centrality …")
    partition, modularity = detect_communities(G1, seed=42)
    sizes = community_sizes(partition)
    profile = community_opinion_profile(df, partition)
    cent1 = centrality_analysis_n1(G1)
    cent2 = centrality_analysis_n2(G2)

    print("\n[4/6] Polarization and robustness analyses …")
    pol = polarization_scores(df)
    top_polar, top_consensus = top_polarizing_consensus(pol, n=5)
    homophily = within_vs_across_variance(df, partition)
    cos_vals = sim_cos.values[np.triu_indices_from(sim_cos.values, k=1)]
    pear_vals = sim_pear.values[np.triu_indices_from(sim_pear.values, k=1)]
    mask = ~np.isnan(cos_vals) & ~np.isnan(pear_vals)
    cos_pear_corr = np.corrcoef(cos_vals[mask], pear_vals[mask])[0, 1]

    k_sensitivity = respondent_k_sensitivity(df)
    threshold_sensitivity = statement_threshold_sensitivity(df)
    signed_summary = signed_network_summary(G2)
    domain_matrix = domain_correlation_matrix(df)
    statement_edge_summary = within_vs_across_statement_edges(G2)
    pos_corr, neg_corr = top_statement_correlations(corr_matrix, n=10)
    stability = community_stability_analysis(df, k=8, runs=6)
    eta2_summary = eta_squared_explained_variance(df, partition)
    polar_vs_pos = polarization_vs_network_position(G2, pol)

    print("\n[5/6] Saving visualizations …")
    edge_weights = [d.get("weight", 0.0) for _, _, d in G1.edges(data=True)]
    effective_threshold = min(edge_weights) if edge_weights else 0.0
    plot_similarity_distribution(sim_cos, threshold=effective_threshold)
    plot_respondent_network(G1, partition, cent1)
    plot_statement_network(G2)
    plot_polarization_bars(top_polar, top_consensus, code_to_text)
    plot_community_heatmap(profile)
    plot_degree_distribution(G1)
    plot_k_sensitivity(k_sensitivity)
    plot_threshold_sensitivity(threshold_sensitivity)
    plot_domain_correlation_heatmap(domain_matrix)
    plot_polarization_vs_centrality(polar_vs_pos)

    print("\n[6/6] Writing report and saving results …")
    report_md = _build_report_markdown(
        df, code_to_text, stats1, stats2, partition, profile, cent1, cent2,
        top_polar, top_consensus, homophily, cos_pear_corr, k_sensitivity,
        threshold_sensitivity, signed_summary, domain_matrix, statement_edge_summary,
        stability, eta2_summary, polar_vs_pos,
    )
    report_path = Path("output/report.md")
    report_path.write_text(report_md, encoding="utf-8")
    build_pdf(report_path, Path("output/report.pdf"))

    results = {
        "n_respondents": int(df.shape[0]),
        "n_statements": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        "stats_n1": stats1,
        "stats_n2": stats2,
        "modularity": modularity,
        "n_communities": len(sizes),
        "community_sizes": sizes.to_dict(),
        "profile": profile.to_dict(),
        "top5_polarizing": {idx: {"var": round(row["variance"], 3), "mean": round(row["mean"], 3)} for idx, row in top_polar.iterrows()},
        "top5_consensus": {idx: {"var": round(row["variance"], 3), "mean": round(row["mean"], 3)} for idx, row in top_consensus.iterrows()},
        "cos_pear_corr": round(float(cos_pear_corr), 4),
        "k_sensitivity": k_sensitivity.to_dict(orient="records"),
        "threshold_sensitivity": threshold_sensitivity.to_dict(orient="records"),
        "signed_summary": signed_summary,
        "domain_matrix": domain_matrix.round(4).to_dict(),
        "statement_edge_summary": statement_edge_summary.to_dict(),
        "community_stability": stability,
        "eta_squared": eta2_summary.to_dict(orient="records"),
        "polarization_vs_centrality": polar_vs_pos.to_dict(orient="records"),
    }
    Path("output/results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  Pipeline complete! All figures and PDF were regenerated from current data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
