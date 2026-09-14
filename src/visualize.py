"""
visualize.py — Generate all publication-quality figures for the report.

Six required figures:
  1. Similarity distribution histogram
  2. Network 1 — respondent graph colored by community
  3. Network 2 — statement graph colored by T/E/S/V category
  4. Polarization bar chart (top-5 polarizing + top-5 consensus)
  5. Heatmap — community × category mean opinion
  6. Degree distribution — Network 1
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")                            # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.data_prep import get_category, CATEGORY_LABELS

# ── Style defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

COMMUNITY_COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12",
                     "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
CATEGORY_COLORS = {"T": "#3498db", "E": "#2ecc71", "S": "#e74c3c", "V": "#f39c12"}

FIG_DIR = Path("output/figures")


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Similarity distribution histogram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_similarity_distribution(sim_matrix: pd.DataFrame, threshold: float | None = None):
    """Plot histogram of off-diagonal cosine similarities."""
    vals = sim_matrix.values[np.triu_indices_from(sim_matrix.values, k=1)]
    vals = vals[~np.isnan(vals)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Pairwise Respondent Cosine Similarities")
    if threshold is not None:
        ax.axvline(threshold, color="#e74c3c", ls="--", lw=2,
                   label=f"Threshold = {threshold:.2f}")
        ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig1_similarity_distribution")
    return vals


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Network 1 graph (respondents, colored by community)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_respondent_network(G: nx.Graph, partition: dict, centrality: pd.DataFrame):
    """Spring-layout graph of Network 1, colored by community, sized by eigenvector centrality."""
    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(G.number_of_nodes()), weight="weight", iterations=80)

    # Node colors & sizes
    node_list = list(G.nodes())
    colors = [COMMUNITY_COLORS[partition.get(n, 0) % len(COMMUNITY_COLORS)] for n in node_list]
    eigen_vals = [centrality.loc[n, "eigenvector"] if n in centrality.index else 0.01 for n in node_list]
    sizes = [300 + 2000 * v for v in eigen_vals]

    # Edge widths
    edges = G.edges(data=True)
    weights = [d.get("weight", 0.5) for _, _, d in edges]
    max_w = max(weights) if weights else 1
    edge_widths = [0.3 + 2.0 * (w / max_w) for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=edge_widths, edge_color="#95a5a6")
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=0.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="black")

    # Legend
    comm_ids = sorted(set(partition.values()))
    patches = [mpatches.Patch(color=COMMUNITY_COLORS[c % len(COMMUNITY_COLORS)],
                              label=f"Community {c}") for c in comm_ids]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.8)
    ax.set_title("Network 1: Respondent Similarity Network\n(colored by Louvain community, sized by eigenvector centrality)")
    ax.axis("off")
    _save(fig, "fig2_respondent_network")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Network 2 graph (statements, colored by T/E/S/V)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_statement_network(G: nx.Graph):
    """
    Spring-layout graph of Network 2, colored by category.
    Positive-correlation edges = solid blue, negative = dashed red.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(G.number_of_nodes(), 1)),
                           weight="weight", iterations=100)

    node_list = list(G.nodes())
    colors = [CATEGORY_COLORS.get(get_category(n), "#aaa") for n in node_list]

    # Separate positive/negative edges
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign", 1) < 0]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax,
                           alpha=0.25, edge_color="#3498db", width=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                           alpha=0.4, edge_color="#e74c3c", style="dashed", width=1.0)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=colors, node_size=350,
                           edgecolors="white", linewidths=0.5, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="black")

    # Legend for categories
    cat_patches = [mpatches.Patch(color=c, label=f"{k}: {CATEGORY_LABELS[k]}")
                   for k, c in CATEGORY_COLORS.items()]
    # Edge legend
    cat_patches.append(plt.Line2D([0], [0], color="#3498db", lw=1.5, label="Positive correlation"))
    cat_patches.append(plt.Line2D([0], [0], color="#e74c3c", lw=1.5, ls="--", label="Negative correlation"))
    ax.legend(handles=cat_patches, loc="upper left", fontsize=8, framealpha=0.8)
    ax.set_title("Network 2: Statement Co-Endorsement Network\n(colored by T/E/S/V category)")
    ax.axis("off")
    _save(fig, "fig3_statement_network")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Top-5 polarizing + top-5 consensus
# ═══════════════════════════════════════════════════════════════════════════════

def plot_polarization_bars(top_polar: pd.DataFrame, top_consensus: pd.DataFrame,
                           code_to_text: dict):
    """Horizontal bar chart of the 10 extreme statements."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Polarizing
    ax = axes[0]
    labels = [f"{idx}" for idx in top_polar.index]
    ax.barh(labels, top_polar["variance"], color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Variance")
    ax.set_title("Top-5 Most Polarizing Statements")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    # Consensus
    ax = axes[1]
    labels = [f"{idx}" for idx in top_consensus.index]
    ax.barh(labels, top_consensus["variance"], color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Variance")
    ax.set_title("Top-5 Most Consensus Statements")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig4_polarization_bars")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Heatmap: community × category
# ═══════════════════════════════════════════════════════════════════════════════

def plot_community_heatmap(profile: pd.DataFrame):
    """Heatmap of mean opinion score by community and T/E/S/V category."""
    data = profile[["T", "E", "S", "V"]].copy()
    data.index = [f"Community {i}\n(n={int(profile.loc[i, 'n'])})" for i in data.index]
    data.columns = [f"{c}: {CATEGORY_LABELS[c]}" for c in data.columns]

    fig, ax = plt.subplots(figsize=(8, max(3, len(data) * 0.8 + 1)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=1, linecolor="white", ax=ax, vmin=-2, vmax=2,
                cbar_kws={"label": "Mean Likert Score (−2 to +2)"})
    ax.set_title("Mean Opinion Score by Community × Category")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, "fig5_community_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — Degree distribution
# ═══════════════════════════════════════════════════════════════════════════════

def plot_degree_distribution(G: nx.Graph):
    """Degree distribution histogram for Network 1."""
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degrees, bins=range(min(degrees), max(degrees) + 2),
            color="#9b59b6", edgecolor="white", alpha=0.85, align="left")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Degree Distribution — Network 1 (Respondent Similarity)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(np.mean(degrees), color="#e74c3c", ls="--", lw=1.5,
               label=f"Mean = {np.mean(degrees):.1f}")
    ax.legend()
    _save(fig, "fig6_degree_distribution")


def plot_k_sensitivity(df: pd.DataFrame):
    """Plot how respondent-network metrics change with k."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = df["k"]
    axes[0, 0].plot(x, df["edges"], marker="o")
    axes[0, 0].set_title("Edge count")
    axes[0, 1].plot(x, df["avg_degree"], marker="o", color="#20a39e")
    axes[0, 1].set_title("Average degree")
    axes[1, 0].plot(x, df["clustering"], marker="o", color="#f4a261")
    axes[1, 0].set_title("Clustering coefficient")
    axes[1, 1].plot(x, df["modularity"], marker="o", color="#e76f51")
    axes[1, 1].set_title("Louvain modularity")
    for ax in axes.flat:
        ax.set_xlabel("k")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig7_k_sensitivity")


def plot_threshold_sensitivity(df: pd.DataFrame):
    """Plot statement-network sensitivity across absolute-correlation thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = df["threshold"]
    axes[0, 0].plot(x, df["edges"], marker="o")
    axes[0, 0].set_title("Edge count")
    axes[0, 1].plot(x, df["density"], marker="o", color="#3a86ff")
    axes[0, 1].set_title("Density")
    axes[1, 0].plot(x, df["lcc_size"], marker="o", color="#ff7b00")
    axes[1, 0].set_title("LCC size")
    axes[1, 1].plot(x, df["clustering"], marker="o", color="#8338ec")
    axes[1, 1].set_title("Clustering coefficient")
    for ax in axes.flat:
        ax.set_xlabel("|r| threshold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig8_threshold_sensitivity")


def plot_domain_correlation_heatmap(matrix: pd.DataFrame):
    """Heatmap of domain-level score correlation matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Domain-level score correlations")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Domain")
    fig.tight_layout()
    _save(fig, "fig9_domain_correlation_heatmap")


def plot_polarization_vs_centrality(summary: pd.DataFrame):
    """Scatter of statement variance against weighted degree and betweenness."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(summary["weighted_degree"], summary["variance"], s=25, alpha=0.7)
    axes[0].set_xlabel("Weighted degree")
    axes[0].set_ylabel("Variance")
    axes[0].set_title("Polarization vs statement degree")
    axes[1].scatter(summary["betweenness"], summary["variance"], s=25, alpha=0.7)
    axes[1].set_xlabel("Betweenness centrality")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Polarization vs statement betweenness")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig10_polarization_vs_centrality")
