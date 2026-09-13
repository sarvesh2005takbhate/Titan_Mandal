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

from src.data_prep import load_and_encode, missing_data_details, get_category, CATEGORY_LABELS
from src.build_networks import build_respondent_network, build_statement_network
from src.analysis import (
    descriptive_stats, detect_communities, community_sizes,
    community_opinion_profile, community_statement_means,
    centrality_analysis_n1, centrality_analysis_n2,
    polarization_scores, top_polarizing_consensus,
    within_vs_across_variance,
)
from src.visualize import (
    plot_similarity_distribution, plot_respondent_network,
    plot_statement_network, plot_polarization_bars,
    plot_community_heatmap, plot_degree_distribution,
    FIG_DIR,
)


def main():
    print("=" * 70)
    print("  DPCN Assignment — Survey Network Analysis Pipeline")
    print("=" * 70)

    # Ensure output dirs exist
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load and encode data ─────────────────────────────────────
    print("\n[1/7] Loading and encoding survey data …")
    csv_path = Path("data/Survey_Results_UC.csv")
    df, code_to_text, missing_summary = load_and_encode(csv_path)
    per_stmt_missing, per_resp_missing = missing_data_details(df)

    print(f"  Respondents: {df.shape[0]}")
    print(f"  Statements : {df.shape[1]}")
    print(f"  Total cells : {df.shape[0] * df.shape[1]}")
    print(f"  Missing cells: {int(df.isna().sum().sum())} "
          f"({df.isna().sum().sum() / (df.shape[0]*df.shape[1]) * 100:.2f}%)")

    print("\n  Top statements with most missing data:")
    print(per_stmt_missing[per_stmt_missing > 0].to_string())

    print("\n  Respondents with most missing data:")
    top_resp_missing = per_resp_missing[per_resp_missing > 0].head(10)
    print(top_resp_missing.to_string())

    # ── Step 2: Build Networks ───────────────────────────────────────────
    print("\n[2/7] Building Network 1: Respondent Similarity …")
    G1, sim_cos, sim_pear = build_respondent_network(df, k=8, method="knn")

    # Compute effective threshold for reporting
    # Get the min weight of edges actually added
    edge_weights = [d["weight"] for _, _, d in G1.edges(data=True)]
    effective_threshold = min(edge_weights) if edge_weights else 0
    print(f"  Effective min similarity in edges: {effective_threshold:.4f}")

    print("\n[3/7] Building Network 2: Statement Co-Endorsement …")
    G2, stmt_corr = build_statement_network(df, corr_threshold=0.3)

    # ── Step 3: Descriptive stats ────────────────────────────────────────
    print("\n[4/7] Computing descriptive network statistics …")
    stats1 = descriptive_stats(G1, "Respondent Similarity (Network 1)")
    stats2 = descriptive_stats(G2, "Statement Co-Endorsement (Network 2)")
    for s in [stats1, stats2]:
        print(f"\n  === {s['name']} ===")
        for k, v in s.items():
            if k != "name":
                print(f"    {k:25s} = {v}")

    # ── Step 4: Community detection ──────────────────────────────────────
    print("\n[5/7] Detecting communities (Louvain) …")
    partition, modularity = detect_communities(G1)
    sizes = community_sizes(partition)
    print(f"  Modularity: {modularity}")
    print(f"  Communities found: {len(sizes)}")
    print(f"  Community sizes:\n{sizes.to_string()}")

    # Community opinion profiles
    profile = community_opinion_profile(df, partition)
    print("\n  Community opinion profiles (mean Likert per category):")
    print(profile.to_string())

    # ── Step 5: Centrality analysis ──────────────────────────────────────
    print("\n[6/7] Centrality analysis …")
    cent1 = centrality_analysis_n1(G1)
    print("\n  Network 1 — Top-5 by eigenvector centrality:")
    print(cent1.head(5).to_string())
    print("\n  Network 1 — Bottom-5 (most isolated):")
    print(cent1.tail(5).to_string())

    cent2 = centrality_analysis_n2(G2)
    print("\n  Network 2 — Top-10 bridge statements (betweenness):")
    print(cent2.head(10)[["betweenness", "category"]].to_string())

    # Polarization
    pol = polarization_scores(df)
    top_polar, top_consensus = top_polarizing_consensus(pol)
    print("\n  Top-5 most polarizing statements:")
    for idx, row in top_polar.iterrows():
        print(f"    {idx} (var={row['variance']:.3f}, mean={row['mean']:.3f}): "
              f"{code_to_text.get(idx, idx)[:70]}")
    print("\n  Top-5 most consensus statements:")
    for idx, row in top_consensus.iterrows():
        print(f"    {idx} (var={row['variance']:.3f}, mean={row['mean']:.3f}): "
              f"{code_to_text.get(idx, idx)[:70]}")

    # Homophily / within-vs-across variance
    homophily = within_vs_across_variance(df, partition)
    print("\n  Within-community vs overall variance by category:")
    print(homophily.to_string(index=False))

    # Robustness check: compare cosine vs Pearson
    cos_vals = sim_cos.values[np.triu_indices_from(sim_cos.values, k=1)]
    pear_vals = sim_pear.values[np.triu_indices_from(sim_pear.values, k=1)]
    mask = ~np.isnan(cos_vals) & ~np.isnan(pear_vals)
    cos_pear_corr = np.corrcoef(cos_vals[mask], pear_vals[mask])[0, 1]
    print(f"\n  Robustness: Correlation between cosine and Pearson similarity = {cos_pear_corr:.4f}")

    # ── Step 6: Generate all figures ─────────────────────────────────────
    print("\n[7/7] Generating visualizations …")
    plot_similarity_distribution(sim_cos, threshold=effective_threshold)
    plot_respondent_network(G1, partition, cent1)
    plot_statement_network(G2)
    plot_polarization_bars(top_polar, top_consensus, code_to_text)
    plot_community_heatmap(profile)
    plot_degree_distribution(G1)

    print("\n" + "=" * 70)
    print("  Pipeline complete! All figures saved to output/figures/")
    print("=" * 70)

    # ── Save results to JSON for report generation ───────────────────────
    results = {
        "n_respondents": int(df.shape[0]),
        "n_statements": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(df.isna().sum().sum() / (df.shape[0]*df.shape[1]) * 100, 2),
        "stats_n1": stats1,
        "stats_n2": stats2,
        "modularity": modularity,
        "n_communities": len(sizes),
        "community_sizes": sizes.to_dict(),
        "profile": profile.to_dict(),
        "top5_central_n1": cent1.head(5).index.tolist(),
        "top5_central_n1_values": cent1.head(5)["eigenvector"].round(4).to_dict(),
        "bottom5_n1": cent1.tail(5).index.tolist(),
        "top10_bridge_n2": cent2.head(10)[["betweenness", "category"]].to_dict(),
        "top5_polarizing": {idx: {"var": round(row["variance"], 3), "mean": round(row["mean"], 3),
                                   "text": code_to_text.get(idx, idx)}
                            for idx, row in top_polar.iterrows()},
        "top5_consensus": {idx: {"var": round(row["variance"], 3), "mean": round(row["mean"], 3),
                                  "text": code_to_text.get(idx, idx)}
                           for idx, row in top_consensus.iterrows()},
        "homophily": homophily.to_dict(orient="records"),
        "cos_pear_corr": round(cos_pear_corr, 4),
        "effective_threshold": round(effective_threshold, 4),
        "per_stmt_missing": per_stmt_missing[per_stmt_missing > 0].to_dict(),
    }
    results_path = Path("output/results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
